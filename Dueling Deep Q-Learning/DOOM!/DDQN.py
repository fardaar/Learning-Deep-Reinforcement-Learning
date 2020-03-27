import tensorflow as tf
import numpy as np
from collections import deque
import AgentModel
import Memory
import environment
import stack_frames


class DDQN:
    def __init__(self, input_shape, learning_rate, actions, batch_size, gamma):
        self.TargetNetwork = AgentModel.AgentModel(input_shape)
        self.DDQNetwork = AgentModel.AgentModel(input_shape)
        self.experience_buffer = Memory.Memory(0.01)
        self.learning_rate = learning_rate
        self.actions = np.array(actions)
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.num_actions = 3

    def predict_q(self, input: np.array):
        return self.DDQNetwork(input)

    def predict_target(self, input: np.array):
        return self.TargetNetwork(input)

    @tf.function
    def train(self):
        experience_batch = self.experience_buffer.sample(self.batch_size)
        states = [experience.state for experience in experience_batch]
        actions = [experience.action for experience in experience_batch]
        rewards = [experience.reward for experience in experience_batch]
        dones = [experience.done for experience in experience_batch]
        next_states = [experience.next_state for experience in experience_batch]
        td_errors = [experience.td_error for experience in experience_batch]

        next_values = self.predict_target(next_states)
        indexes = tf.math.argmax(next_values, axis=1)
        targetQs = []

        # If we're in a terminal state, return the reward, if not, return rewards+gamma*(next action's Q-value)
        for i, done in enumerate(dones):
            if done:
                targetQs.append(rewards[i])
            else:
                targetQs.append(rewards[i] + self.gamma * next_values[i][indexes[i]])

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.predict_q(states) * actions, axis=1)
            td_error = tf.math.subtract(tf.math.square(targetQs), tf.math.square(selected_action_values))
            loss = tf.math.reduce_sum(td_error)

        variables = self.DDQNetwork.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def update_target_weights(self):
        self.TargetNetwork.set_weights(self.DDQNetwork.get_weights())

    def get_action(self, states, epsilon):
        # Arguments:
        #   states: states which we want to predict corresponding actions for
        #   epsilon: exploration rate
        # Returns:
        #   action: the action to be taken
        # Implements:
        #   Passes the states through the agent's network and calculates the action that must be taken

        # if np.random.random() < epsilon:
        #     # Explore
        #     return self.actions[np.random.choice(self.num_actions)]
        # else:
            # Exploit
        action = self.predict_q(states)
        action_index = tf.math.argmax(action, axis=1).numpy()[0]
        return self.actions[action_index]


# game, possible_actions = environment.create_environment()
# game.new_episode()
# stacked_frames = deque([np.zeros((100, 120), dtype=np.int) for _ in range(4)], maxlen=4)
# state = game.get_state()
# first_frame = state.screen_buffer
# stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, True)
# ddqn_agen = DDQN([100, 120, 4], 0.01, possible_actions)
# action = ddqn_agen.predict_q(np.array(stacked_frames).reshape(1, 100, 120, 4))
# print('hi')

# def play_game(agent, epsilon):
#     # Arguments:
#     #   agent: an agent instance
#     #   epsilon: exploration rate
#     # Returns:
#     #   -
#     # Implements:
#     #   playing DOOM!
#
#     # create an environment for agent to play in
#     game, possible_actions = environment.create_environment()
#     game.new_episode()
#
#     stacked_frames = deque([np.zeros((100, 120), dtype=np.int) for _ in range(4)], maxlen=4)
#     # get current state
#     state = game.get_state()
#     # Since after creating an enviroment we have only one frame to go with, copy it 4 times
#     stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, True)
#     # take action
#     action = agent.get_action(np.asarray(stacked_frames).reshape(1, 100, 120, 4), epsilon)
#     # get reward
#     reward = game.make_action(action)
#     # determine if we're in the terminal state or not
#     done = game.is_episode_finished()
#     # get the next state
#     next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
#     # add it to agent's replay buffer
#     agent.add_experience((np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, done))
#
#     # until we reach the terminal state:
#     while not game.is_episode_finished():
#         state = game.get_state()
#         stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
#         action = agent.get_action(np.asarray(stacked_frames).reshape(1, 84, 84, 4), epsilon)
#         reward = game.make_action(action)
#         done = game.is_episode_finished()
#         next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
#         agent.add_experience((np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, done))
game, possible_actions = environment.create_environment()
game.new_episode()
agent = DDQN(input_shape=[100, 120, 4],learning_rate=0.01,actions=possible_actions,batch_size=4,gamma=0.0001 )
epsilon=0.01
stacked_frames = deque([np.zeros((100, 120), dtype=np.int) for _ in range(4)], maxlen=4)
# get current state
state = game.get_state()
# Since after creating an enviroment we have only one frame to go with, copy it 4 times
stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, True)
# take action
action = agent.get_action(np.asarray(stacked_frames).reshape(1, 100, 120, 4), epsilon)
# get reward
reward = game.make_action(action)
# determine if we're in the terminal state or not
done = game.is_episode_finished()
# get the next state
next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
# add it to agent's replay buffer
agent.add_experience((np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, done))

# until we reach the terminal state:
while not game.is_episode_finished():
    state = game.get_state()
    stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
    action = agent.get_action(np.asarray(stacked_frames).reshape(1, 84, 84, 4), epsilon)
    reward = game.make_action(action)
    done = game.is_episode_finished()
    next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
    agent.add_experience((np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, done))
