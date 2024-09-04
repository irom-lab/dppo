"""
Test Robomimic rendering, no GUI

"""

import os
import time
from gym import spaces
import robosuite as suite

os.environ["MUJOCO_GL"] = "egl"
if __name__ == "__main__":
    env = suite.make(
        env_name="TwoArmTransport",
        robots=["Panda", "Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_heights=96,
        camera_widths=96,
        camera_names="shouldercamera0",
        render_gpu_device_id=0,
        horizon=20,
    )
    obs, done = env.reset(), False
    print("Finished resetting!")
    low, high = env.action_spec
    action_space = spaces.Box(low=low, high=high)
    steps, time_stamp = 0, time.time()
    while True:
        while not done:
            obs, reward, done, info = env.step(action_space.sample())
            steps += 1
        obs, done = env.reset(), False
        print(f"FPS: {steps / (time.time() - time_stamp)}")
        steps, time_stamp = 0, time.time()
