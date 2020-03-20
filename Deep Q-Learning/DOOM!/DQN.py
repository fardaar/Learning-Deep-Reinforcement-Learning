import numpy as np
import AgentModel
import tensorflow as tf
import experience_replay
import environment
from collections import deque
import stack_frames


class DQN:
    # Here we create our agent's DQN
    def __init__(self, state_shape, num_actions, gamma, batch_size, max_buffer_size, learning_rate, pretrain_length,
                 stack_size, possible_actions):
        # Arguments:
        #   state_shape: the shape of each input which we're going to feed our agent
        #   num_actions: number of actions that our agent is capable of doing in the environment
        #   gamma: our Q-Learning discount factor
        #   batch_size: batch size
        #   max_buffer_size: the maximum limit of experiences which we can store in our replay buffer
        #   learning_rate: learning rate
        #   pretrain_length: number of instances to put inside our replay buffer before beginning training the agent
        #   stack_size: how many frames to stack together to create a single state for feeding the agent
        # Implements:
        #   Creates an instance of our agent
        self.possible_actions = possible_actions
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)  # We're gonna use RMSProp as out optimizer
        self.gamma = gamma
        self.model = AgentModel.AgentModel(self.state_shape, self.num_actions)  # Instantiate an agent
        self.memory = experience_replay.create_and_fill_memory(stack_size,
                                                               pretrain_length)  # Fill agent's replay buffer before training
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

    def predict(self, inputs):
        # Arguments:
        #   inputs: a single or a batch of states to determine the next action from
        # Returns:
        #   agent network's output. Q values for each possible action
        # Implements:
        #   A single forward pass through agent's network with the given inputs
        return self.model(inputs)

    @tf.function
    def train(self):
        # Arguments:
        #   -
        # Returns:
        #   -
        # Implements:
        #   Training the agent on the experiences in its replay buffer

        # Sampling random experiences from agent's replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # predict the next state's Q-values
        next_values = self.predict(next_states)
        # Get the indexes of the action with hte highest Q-value
        indexes = tf.math.argmax(next_values, axis=1)
        # Create an empty list. This is going to contain our target-Qs values
        targetQs = []

        # If we're in a terminal state, return the reward, if not, return rewards+gamma*(next action's Q-value)
        for i, done in enumerate(dones):
            if done:
                targetQs.append(rewards[i])
            else:
                targetQs.append(rewards[i] + self.gamma * next_values[i][indexes[i]])

        # start keeping tab on agent network's variables for backpropagation
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.predict(states) * actions, axis=1)
            loss = tf.math.reduce_sum(tf.square(tf.math.subtract(targetQs, selected_action_values)))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        # Arguments:
        #   states: states which we want to predict corresponding actions for
        #   epsilon: exploration rate
        # Returns:
        #   action: the action to be taken
        # Implements:
        #   Passes the states through the agent's network and calculates the action that must be taken

        if np.random.random() < epsilon:
            # Explore
            return self.possible_actions[np.random.choice(self.num_actions)]
        else:
            # Exploit
            action = self.predict(states)
            action_index = tf.math.argmax(action, axis=1).numpy()[0]
            return self.possible_actions[action_index]

    def add_experience(self, experience):
        # Arguments:
        #   experience: an experience instance. it contain the current state, action taken, reward received, next state and whether we have reached the terminal state or not
        # Returns:
        # -
        # Implements:
        #   filling the agent's replay buffer with new experiences

        # if the replay buffer is at it's limit delete the oldest instance and append the latest experience
        if self.max_buffer_size > self.memory.buffer_size:
            for key in self.memory.buffer.keys():
                self.memory.buffer[key].pop(0)
            self.memory.add(experience[0], experience[1], experience[2], experience[3],
                            experience[4])


def play_game(agent, epsilon):
    # Arguments:
    #   agent: an agent instance
    #   epsilon: exploration rate
    # Returns:
    #   -
    # Implements:
    #   playing DOOM!

    # create an environment for agent to play in
    game, possible_actions = environment.create_environment()
    game.new_episode()

    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for _ in range(4)], maxlen=4)
    # get current state
    state = game.get_state()
    # Since after creating an enviroment we have only one frame to go with, copy it 4 times
    stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, True)
    # take action
    action = agent.get_action(np.asarray(stacked_frames).reshape(1, 84, 84, 4), epsilon)
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


left = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]
possible_actions = [left, right, shoot]
agent = DQN(state_shape=[84, 84, 4], num_actions=3, gamma=0.01, batch_size=10, learning_rate=0.001, pretrain_length=64,
            stack_size=4, max_buffer_size=10000, possible_actions=possible_actions)

for i in range(10000):
    play_game(agent, 0.001)
    agent.train()
