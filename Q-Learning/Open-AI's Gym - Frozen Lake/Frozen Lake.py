import numpy as np
import gym
import random
import model


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
q_table = create_q_table(env.action_space.n, env.observation_space.n)  # creating the q-table
hyper_parameters = {'total_episodes': 15000,  # total steps of training
                    'learning_rate': 0.8,  # learning rate
                    'max_steps': 99,  # maximum number of steps before restarting the environment
                    'gamma': 0.95}  # discount factor
exploration_parameters = {'epsilon': 1.0,  # exploration rate
                          'max_epsilon': 1.0,  # maximum exploration rate
                          'min_epsilon': 0.01,  # minimum exploration rate
                          'decay_rate': 0.005}  # the rate at which epsilon is to decay at each iteration

q_table = model.train(env, hyper_parameters, exploration_parameters, q_table)
model.test(q_table, env, hyper_parameters, 5)
