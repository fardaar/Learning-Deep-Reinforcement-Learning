import numpy as np
import gym
import random


def create_q_table(action_size, state_size):
    # Arguments:
    #     action_size: the number of actions our agent can take
    #     state_size: the number of environment's states
    # Returns:
    #     q-table: an empty q-table initialized with zeros
    # Implements:
    #     creates the agent's q-table

    return np.zeros((state_size, action_size))


env = gym.make('FrozenLake-v0')  # Gym's Frozen Lake environment
