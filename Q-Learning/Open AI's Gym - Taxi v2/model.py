import numpy as np
import gym
import random


def train(env, hyper_parameters, exploration_parameters, qtable):
    # Arguments:
    #     env: Gym's environment
    #     hyper_parameters: model's hyperparameters
    #     exploration_parameters: parameters for defining model's exploration parameters
    #     qtable: Agent's Q-Table
    # Returns:
    #     qtable: Agent's Q-Table after training
    # Implements:
    #     simple Q-Learning algorithm for training an agent in a gym environment

    total_episodes = hyper_parameters['total_episodes']
    learning_rate = hyper_parameters['learning_rate']
    max_steps = hyper_parameters['max_steps']
    gamma = hyper_parameters['gamma']

    epsilon = exploration_parameters['epsilon']
    max_epsilon = exploration_parameters['max_epsilon']
    min_epsilon = exploration_parameters['min_epsilon']
    decay_rate = exploration_parameters['decay_rate']

    rewards = []

    for episode in range(total_episodes):  # loop over the number of total episodes
        state = env.reset()  # resetting the environment for each episode
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):  # loop over the number of steps on each iteration
            exp_exp_tradeoff = random.uniform(0,
                                              1)  # a random number between 0 and 1 which determines either to choose an action randomly or choose one according to the q-table

            if exp_exp_tradeoff > epsilon:  # if the random number is higher than the exploration rate, use q-table
                action = np.argmax(qtable[state, :])
            else:  # if the random number is less than the exploration rate, choose a random action
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)  # feeding the action to the environment
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[
                state, action])  # updating the q-table according to the reward received from the environment

            total_rewards += reward
            state = new_state  # updating agent's current state

            if done:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode)  # updating the exploration rate according to Q-Learning's formula
        rewards.append(total_rewards)  # to track rewards

    print('Score over time: {0}'.format(sum(rewards) / total_episodes))

    return qtable


def test(qtable, env, hyper_parameters, episodes):
    # Arguments:
    #     qtable: trained q-table
    #     env: environment
    #     hyper_parameters: hyperparameters
    #     episodes: number of episodes to test the agent on
    # Implements:
    #     testing the agent with the trained q-table on a given number of episodes
    env.reset()

    for episode in range(episodes):

        state = env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(hyper_parameters['max_steps']):

            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(qtable[state, :])

            new_state, reward, done, info = env.step(action)

            if done:
                # show the result at the end of the testing in this episode
                env.render()

                # print the number of step it took.
                print("Number of steps", step)
                break

            state = new_state
    env.close()
