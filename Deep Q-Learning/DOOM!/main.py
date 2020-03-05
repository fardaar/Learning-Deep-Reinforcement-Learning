import tensorflow as tf
import numpy as np
from vizdoom import *

import random
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt
import warnings


def create_environment():
    game = DoomGame()
    game.load_config('basic.cfg')
    game.set_doom_scenario_path("basic.wad")
    game.init()
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions


def test_envirnment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]
    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()


def preprocess_frame(frame):
    cropped_frame = frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


# test_envirnment()
game, possible_actions = create_environment()
stack_size = 4
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

model_hyperparameters = {'state_size': [84, 84, 4],
                         'action_size': game.get_available_buttons_size(),
                         'learning_rate': 0.0002}

training_hyperparameters = {'total_episodes': 500,
                            'max_steps': 100,
                            'batch_size': 64}

exploration_hyperparameters = {'explore_start': 1.0,
                               'explore_stop': 0.01,
                               'decay_rate': 0.0001}
q_learning_hyperparameters = {'gamma': 0.95}

memory_hyperparameters = {'pretrain_length': training_hyperparameters['batch_size'],
                          'memory_size': 1000000}

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False
