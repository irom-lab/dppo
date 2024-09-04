"""
Visualize Avoid environment from D3IL in MuJoCo GUI

"""

import gym
import gym_avoiding
import imageio

# from gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
# from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from gym.envs import make as make_
import numpy as np

# env = ObstacleAvoidanceEnv(render=False)
env = make_("avoiding-v0", render=True)
# env.start()   # no need to start() any more, already run in init() now
env.reset()
print(env.action_space)

# video_writer = imageio.get_writer("test_d3il.mp4", fps=30)
while 1:
    obs, reward, done, info = env.step(np.array([0.02, 0.1]))
    print("Reward:", reward)
    # video_img = env.render(
    #     mode="rgb_array",
    #     # height=640,
    #     # width=480,
    #     # camera_name=self.render_camera_name,
    # )
    # video_writer.append_data(video_img)
    if input("Press space to stop, or any other key to continue") == " ":
        break
# video_writer.close()
