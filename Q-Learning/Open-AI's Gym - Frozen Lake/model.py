import numpy as np
import gym
import random


def train(env, hyper_parameters, exploration_parameters, qtable):
    total_episodes = hyper_parameters['total_episodes']
    learning_rate = hyper_parameters['learning_rate']
    max_steps = hyper_parameters['max_steps']
    gamma = hyper_parameters['gamma']

    epsilon = exploration_parameters['epsilon']
    max_epsilon = exploration_parameters['max_epsilon']
    min_epsilon = exploration_parameters['min_epsilon']
    decay_rate = exploration_parameters['decay_rate']

    rewards = []

    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0, 1)

            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            total_rewards += reward
            state = new_state

            if done:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_rewards)
    print('Score over time: {0}'.format(sum(rewards) / total_episodes))

    return qtable


def test(qtable, env, hyper_parameters, episodes):
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
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                env.render()

                # We print the number of step it took.
                print("Number of steps", step)
                break
            state = new_state
    env.close()
