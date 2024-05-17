import gymnasium as gym
import dill

from env import FourRoomsEnv
from wrappers import gym_wrapper
from utils import obs_to_state
from shortest_path import find_all_action_values
import numpy as np
import pickle
from d3rlpy.dataset import MDPDataset

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                           agent_pos=train_config['agent positions'],
                           goal_pos=train_config['goal positions'],
                           doors_pos=train_config['topologies'],
                           agent_dir=train_config['agent directions']))

# number of episodes and steps per episode
chance_to_choose_optimal = 0

# initialize lists to store states, actions, rewards, and terminal flags
states = []
actions = []
rewards = []
terminal_flags = []


# define your custom policy here
def optimal_policy(state):
    state = obs_to_state(state)
    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
    optimal_action = np.argmax(q_values)
    return optimal_action

    # generate episodes
state, _ = env.reset()
action = optimal_policy(state)
while True:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated
    states.append(state.flatten())
    actions.append(action)
    rewards.append(reward)
    terminal_flags.append(done)

    state = next_state
    if done:
        break


# convert lists to numpy arrays
states = np.array(states)
actions = np.array(actions)
rewards = np.array(rewards)
terminal_flags = np.array(terminal_flags)

# create MDPDataset
dataset = MDPDataset(states, actions, rewards, terminal_flags, action_size=4)

# save the dataset
with open('random_dataset_flattened.pkl', 'wb') as writeFile:
    # Serialize and save the data to the file
    pickle.dump(dataset, writeFile)

# Now you can use the model to predict actions, evaluate its performance, etc.
