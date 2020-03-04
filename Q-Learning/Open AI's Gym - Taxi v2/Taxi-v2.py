import numpy as np
import gym
import random


def create_qtable(action_space_size, observation_space_size):
    return np.zeros((observation_space_size, action_space_size))


env = gym.make('Taxi-v2')
