import argparse
from env import FourRoomsEnv
from wrappers import gym_wrapper
import imageio
import numpy as np
from pyvirtualdisplay import Display
import dill
import gymnasium as gym
import d3rlpy

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

with Display(visible=False) as disp:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                           agent_pos=train_config['agent positions'],
                           goal_pos=train_config['goal positions'],
                           doors_pos=train_config['topologies'],
                           agent_dir=train_config['agent directions'],
                           render_mode="rgb_array"))
    images = []

    bc = d3rlpy.load_learnable('bc_random.d3')
    for i in range(len(train_config['topologies'])):
        obs, _ = env.reset()
        img = env.render()
        images.append(img)
        done = False
        steps = 0
        while not done and steps < 20:
            steps += 1
            obs_flattened = obs.flatten()[None, :]
            action = bc.predict(obs_flattened)
            obs, reward, done, truncated, info = env.step(action[0])
            img = env.render()
            images.append(img)

    gif_name = 'bc_random.gif'

    # Use the determined gif name when saving the gif
    imageio.mimsave(gif_name, [np.array(img) for i, img in enumerate(images) if i % 1 == 0], duration=100)
