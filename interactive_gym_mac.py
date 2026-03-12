# ============================================================================
# IMPORTANT: 此脚本必须使用 mjpython 运行以解决 macOS NSWindow 线程问题
# 运行方式: mjpython interactive_gym.py --policy.path=... --env.type=aloha
# ============================================================================
import os
# 使用 osmesa 软件渲染，避免 GLFW 窗口创建
# 注意：这需要 mujoco-py 或 osmesa 库支持
# 如果 osmesa 不可用，可以尝试 'egl' 或 'glfw'
os.environ.setdefault('MUJOCO_GL', 'glfw')

import gymnasium as gym
import mujoco
import mujoco.viewer
import torch
import importlib
import time
import numpy as np
from lerobot.policies.utils import get_device_from_parameters
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.utils.utils import get_safe_torch_device


class DummyViewer:
    """虚拟 viewer，用于在 mujoco.viewer 不可用时提供替代方案"""
    def __init__(self, max_steps=1000):
        self._running = True
        self._step_count = 0
        self._max_steps = max_steps

    def is_running(self):
        if self._step_count >= self._max_steps:
            return False
        return self._running

    def sync(self):
        self._step_count += 1

    def close(self):
        self._running = False


# 全局渲染器缓存，避免重复创建
_renderer_cache = {}


def patch_gym_aloha_rendering():
    """
    修补 gym_aloha 的渲染问题。

    问题：gym_aloha 使用 dm_control 的 physics.render()，这会尝试创建 OpenGL 上下文。
    在 macOS 上，这与 mujoco.viewer 的 GLFW 上下文冲突，导致 NSWindow 线程错误。

    解决方案：使用 mujoco 的官方 Renderer 类替代 dm_control 的渲染器。
    """
    try:
        # 尝试导入 dm_control 的 mujoco
        from dm_control import mujoco as dm_mujoco

        # 保存原始的 Physics.render 方法
        original_render = dm_mujoco.Physics.render

        def patched_render(self, height=240, width=320, camera_id=-1, scene_option=None):
            """
            使用官方 mujoco 的 Renderer 替代 dm_control 的渲染。
            这避免了 OpenGL 上下文冲突。
            """
            # 获取底层模型和数据
            model = self.model.ptr
            data = self.data.ptr

            # 创建缓存键 - 必须包含 camera_id，否则不同相机会共用同一个渲染器！
            # 这是导致图像视角错乱的关键问题
            cache_key = (id(model), height, width, camera_id)

            # 获取或创建渲染器
            if cache_key not in _renderer_cache:
                _renderer_cache[cache_key] = mujoco.Renderer(model, height=height, width=width)

            renderer = _renderer_cache[cache_key]

            # 更新场景并渲染
            mujoco.mj_forward(model, data)
            # 重要修复：MuJoCo 的 update_scene 支持字符串类型的 camera_id
            # 当 camera_id 是字符串（如 "top"、"angle"）时，直接使用字符串
            # 当 camera_id 是整数时，使用整数
            # 之前的代码错误地将字符串 camera_id 转换为 -1，导致使用了错误的相机视角！
            renderer.update_scene(data, camera=camera_id)

            return renderer.render()

        # 应用补丁
        dm_mujoco.Physics.render = patched_render
        print("Successfully patched dm_control rendering for macOS compatibility.")

    except ImportError:
        print("dm_control not found, skipping render patch.")
    except Exception as e:
        print(f"Warning: Could not patch dm_control rendering: {e}")


# To run, use commands like:
# $ mjpython interactive_gym.py --policy.path=lerobot/act_aloha_sim_insertion_human --env.type=aloha
# $ mjpython interactive_gym.py --policy.path=lerobot/act_aloha_sim_transfer_cube_human --env.type=aloha

@parser.wrap()
def make_env_and_policy(cfg: EvalPipelineConfig):
    """
    Initializes the gymnasium environment and the lerobot policy based on the provided configuration.
    """
    package_name = f"gym_{cfg.env.type}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"Error: {package_name} is not installed.")
        print(f"Please install it with: pip install 'lerobot[{cfg.env.type}]'")
        raise e

    print(f"Making environment: {cfg}")
    gym_handle = f"{package_name}/{cfg.env.task}"

    # 确保环境不使用自己的渲染窗口，避免线程冲突
    # 我们使用 MuJoCo 的 passive viewer 代替
    gym_kwargs = cfg.env.gym_kwargs.copy() if cfg.env.gym_kwargs else {}
    if 'render_mode' not in gym_kwargs:
        gym_kwargs['render_mode'] = None  # 禁用环境自带渲染

    env = gym.make(gym_handle, disable_env_checker=True, **gym_kwargs)

    print(f"Loading policy: {cfg.policy}")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()
    policy.reset()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    return env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor


def main(env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor):
    """
    Runs the main interactive simulation loop.
    """
    device = get_device_from_parameters(policy)
    print(f"Running policy on device: {device}")

    # 启动 MuJoCo passive viewer
    # 注意：必须使用 mjpython 运行此脚本，以确保 GLFW 在主线程初始化
    try:
        viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
        print("MuJoCo viewer launched successfully.")
    except Exception as e:
        print(f"Error launching MuJoCo viewer: {e}")
        print("Falling back to headless mode.")
        viewer = DummyViewer()

    # Reset the environment to get the initial observation.
    observation, info = env.reset(seed=42)

    # --- Helper for debugging ---
    # Print the keys of the observation dictionary to help diagnose mismatches.
    if isinstance(observation, dict):
        print(f"Initial observation keys: {list(observation.keys())}")
    # --------------------------

    # Use viewer.is_running() for a more interactive loop that ends when the window is closed.
    while viewer.is_running():
        start_time = time.time()

        # IMPORTANT: The state synchronization issue you observed is due to the nature of this loop.
        # User interactions in the viewer directly modify the physics state (env.unwrapped.data).
        # However, the `observation` variable used by the policy is only updated by `env.step()`.
        # This means the policy is always acting on the state from *before* your interaction in that step.
        # A true fix requires modifying the environment to refetch observations on demand, which is non-standard.
        # For now, this script treats user interaction as a "perturbation" to the policy's execution.

        try:
            # Preprocess the observation from the *previous* step.
            observation = preprocess_observation(observation)

            observation = {
                key: observation[key].to(device) for key in observation
            }

            # Infer "task" from attributes of environments.
            # TODO: works with SyncVectorEnv but not AsyncVectorEnv
            if hasattr(env, "task_description"):
                observation["task"] = env.unwrapped.task_description
            elif hasattr(env, "task"):
                observation["task"] = env.unwrapped.task
            else:  #  For envs without language instructions, e.g. aloha transfer cube and etc.
                observation["task"] = ""

            # Infer "task" from attributes of environments.
            # TODO: works with SyncVectorEnv but not AsyncVectorEnv
            # observation = add_envs_task(env, observation)

            # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
            observation = env_preprocessor(observation)

            observation = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)

            action_transition = {"action": action}
            action_transition = env_postprocessor(action_transition)
            action = action_transition["action"]

            # Convert to CPU / numpy.
            action_numpy: np.ndarray = action.to("cpu").numpy()
            assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

            # Apply the action to the environment and get the next observation.
            observation, reward, terminated, truncated, info = env.step(action_numpy[0])

            # Sync the viewer to reflect the new state.
            viewer.sync()

            # If the episode is over, reset the environment and the policy's internal state.
            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                observation, info = env.reset()
                policy.reset()
                viewer.sync()

        except AttributeError as e:
            print("\n--- ATTRIBUTE ERROR ---")
            print(f"Caught an error: {e}")
            print("This usually means the policy is incompatible with the current environment.")
            print("Check the 'Initial observation keys' printed at the start.")
            print("The policy likely expects a key that this environment does not provide.")
            print("Stopping simulation loop.\n")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Stopping simulation loop.\n")
            break

        # Optional: Cap the loop frequency to avoid overwhelming the CPU.
        #time.sleep(max(0, 0.01 - (time.time() - start_time)))


    viewer.close()
    env.close()
    print("Viewer closed. Exiting.")


if __name__ == "__main__":
    # These settings can improve performance on NVIDIA GPUs.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ============================================================================
    # IMPORTANT for macOS users:
    # This script MUST be run with `mjpython` instead of regular `python`:
    #
    #   mjpython interactive_gym.py --policy.path=lerobot/act_aloha_sim_insertion_human --env.type=aloha
    #
    # The `mjpython` command ensures that all UI operations (including GLFW window
    # creation) happen on the main thread, which is required by macOS.
    #
    # If you don't have mjpython, you can install MuJoCo which includes it:
    #   pip install mujoco
    #
    # Then find mjpython at: <python_env>/bin/mjpython
    # ============================================================================

    # 关键步骤：在创建环境之前应用补丁，替换 dm_control 的渲染方法
    # 这确保所有渲染都使用 mujoco 的官方 Renderer，避免 OpenGL 上下文冲突
    patch_gym_aloha_rendering()

    # Parse arguments and initialize the environment and policy.
    env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor = make_env_and_policy()

    # Run the main simulation.
    main(env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor)
