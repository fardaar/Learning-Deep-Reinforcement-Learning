import numpy as np
import AgentModel
import tensorflow as tf
import experience_replay
import environment
from collections import deque
import stack_frames


class DQN:
    def __init__(self, state_shape, num_actions, gamma, batch_size, max_buffer_size, learning_rate, pretrain_length,
                 stack_size):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)
        self.gamma = gamma
        self.model = AgentModel.AgentModel(self.state_shape, self.num_actions)
        self.memory = experience_replay.create_and_fill_memory(stack_size, pretrain_length)
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

    def predict(self, inputs):
        return self.model(inputs)

    @tf.function
    def train(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        next_values = self.predict(next_states)
        indexes = tf.math.argmax(next_values, axis=1)
        targetQs = []

        for i, done in enumerate(dones):
            if done:
                targetQs.append(rewards[i])
            else:
                targetQs.append(rewards[i] + self.gamma * next_values[i][indexes[i]])

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.predict(states) * actions, axis=1)
            loss = tf.math.reduce_sum(tf.square(targetQs - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        # if np.random.random() < epsilon:
        #     return np.random.choice(self.num_actions)
        # else:
        return self.predict(states)

    def add_experience(self, experience):
        if self.max_buffer_size > self.memory.buffer_size:
            for key in self.memory.buffer.keys():
                self.memory.buffer[key].pop(0)

            self.memory.add(experience[0], experience[1], experience[2], experience[3],
                            experience[4])


def play_game(agent, epsilon):
    game, possible_actions = environment.create_environment()
    game.new_episode()
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(4)], maxlen=4)
    state = game.get_state()
    stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, True)
    action = agent.get_action(np.asarray(stacked_frames).reshape(1, 84, 84, 4), epsilon)
    action_index = tf.math.argmax(action, axis=1).numpy()[0]
    action = possible_actions[action_index]
    reward = game.make_action(action)
    done = game.is_episode_finished()
    next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
    agent.add_experience((np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, done))
    while not game.is_episode_finished:
        state = game.get_state()
        stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, True)
        action = agent.get_action(np.asarray(stacked_frames).reshape(1, 84, 84, 4), epsilon)
        action_index = tf.math.argmax(action, axis=1).numpy()[0]
        action = possible_actions[action_index]
        reward = game.make_action(action)
        done = game.is_episode_finished()
        next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
        agent.add_experience((np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, done))


# memory = experience_replay.create_and_fill_memory(pretrain_length=5)
agent = DQN(state_shape=[84, 84, 4], num_actions=3, gamma=0.01, batch_size=10, learning_rate=0.001, pretrain_length=10,
            stack_size=4, max_buffer_size=10000)
for i in range(100):
    play_game(agent, 0.001)
    agent.train()
