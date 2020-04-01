import tensorflow as tf
import numpy as np
import NeuralModel
import environment
import stack_frames
from collections import deque


class PGAgent:
    def __init__(self, state_shape, actions, hyperparams):
        self.input_shape = state_shape
        self.possible_actions = actions
        self.num_actions = len(self.possible_actions[0])
        self.learning_rate = hyperparams['learning_rate']
        self.discount_rate = hyperparams['discount_rate']
        self.model = NeuralModel.NeuralModel(self.input_shape, self.num_actions)
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def predict_action_probs(self, state):
        return self.model(state)

    @tf.function
    def train(self, episode):
        actions = episode['actions']
        rewards = episode['rewards']

        with tf.GradientTape() as tape:
            loss = -tf.math.log(actions) * rewards

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def play(self):
        episode = {'actions': [], 'rewards': []}
        game, possible_actions = environment.create_environment()
        game.new_episode()

        first_state = game.get_state()
        stacked_states = deque([np.zeros((84, 84), dtype=np.int) for _ in range(4)], maxlen=4)
        stacked_states = stack_frames.stack_frames(stacked_states, first_state.screen_buffer, True)
        action_tensor = self.predict_action_probs(np.asarray(stacked_states).reshape(1, 84, 84, 4))
        first_action = possible_actions[tf.math.argmax(action_tensor, axis=1).numpy()[0]]
        first_reward = game.make_action(first_action)
        episode['actions'].append(action_tensor)
        episode['rewards'].append(first_reward)
        self.train(episode)
        while not game.is_episode_finished():
            state = game.get_state()
            stacked_states = stack_frames.stack_frames(stacked_states, state.screen_buffer, False)
            action_tensor = self.predict_action_probs(np.asarray(stacked_states).reshape(1, 84, 84, 4))
            action = tf.math.argmax(action_tensor, axis=1).numpy()[0]
            reward = game.make_action(action)
            episode['actions'].append(action_tensor)
            episode['rewards'].append(reward)


agent = PGAgent([84, 84, 4], [[0, 0, 1], [0, 1, 0], [1, 0, 0]], {'learning_rate': 0.01, 'discount_rate': 0.01})
agent.play()
