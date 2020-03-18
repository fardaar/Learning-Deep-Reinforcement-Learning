from collections import deque
import numpy as np
import environment
import random
import stack_frames
import copy


class Memory:
    # This class creates our experience replay buffer. each experience is defined as a combination of state, action and reward

    def __init__(self):
        # Define buffer
        self.buffer = {'state': [],
                       'action': [],
                       'reward': []}

    def add(self, state, action, reward):
        # To add new experiences to buffer
        self.buffer['state'].append(copy.deepcopy(state))
        self.buffer['action'].append(action)
        self.buffer['reward'].append(reward)

    def sample(self):
        # To return a random sample from the buffer
        rand_index = random.randint(len(self.buffer['state']))
        sample = (
            self.buffer['state'][rand_index], self.buffer['action'][rand_index], self.buffer['reward'][rand_index])
        return sample


def create_and_fill_memory(stack_size=4, pretrain_length=64):
    # Arguments:
    #   stack_size: The size of stacks (how many frames are we going to stack together as a state)
    #   pretrain_length: How many instances we're going to fill the memory with before starting training
    # Returns:
    #   memory: The memory object that's going to be used for experience replay
    # Implements:
    #   It creates and fills a memory object which we're going to use to train our agent.

    # Create an empty deque, instantiate a memory object and create the environment
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
    memory = Memory()
    game, possible_actions = environment.create_environment()

    for i in range(pretrain_length):
        # Start a new episode of the game after it ends
        game.new_episode()
        # Fill the deque with the first frame. this is going to be filled iteratively as the game is played with different consecutive frames
        stacked_frames = stack_frames.stack_frames(stacked_frames, game.get_state().screen_buffer, True, stack_size)
        while not game.is_episode_finished():
            # Until we finish the game (Kill the monster) stack current frame in the state, choose a random action and calculate its reward
            # Add all of them to the memory
            state = game.get_state()
            stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
            action = random.choice(possible_actions)
            reward = game.make_action(action)
            memory.add(np.asarray(stacked_frames).T, action, reward)

    return memory
