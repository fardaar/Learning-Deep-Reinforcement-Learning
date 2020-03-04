import numpy as np
import gym
import random
import model


def create_qtable(action_space_size, observation_space_size):
    return np.zeros((observation_space_size, action_space_size))


env = gym.make("Taxi-v3")
q_table = create_qtable(env.action_space.n, env.observation_space.n)

hyper_parameters = {'total_episodes': 50000,
                    'max_steps': 99,
                    'learning_rate': 0.7,
                    'gamma': 0.618}

exploration_parameters = {'epsilon': 1.0,
                          'max_epsilon': 1.0,
                          'min_epsilon': 0.01,
                          'decay_rate': 0.01}

q_table = model.train(env, hyper_parameters, exploration_parameters, q_table)

model.test(q_table, env, hyper_parameters, 5)
