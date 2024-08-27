import gymnasium as gym
import torch
from gymnasium import Env
import matplotlib.pyplot as plt
import numpy as np
import cv2
from gymnasium.core import ObsType
from mpmath import hyper
from torch import optim

import hyperparameters
from memory import ReplayMemory
from model import DQN


def skip_steps_with_action(env: Env, action: int) -> ObsType:
    """

    :param env: current environment - pracitacllay a global var
    :param action: - action to be repeated for the steps
    :return: screen after last step (to be merged with the next screen)
    """
    for _ in range(hyperparameters.action_repeat - 1):
        observation, reward, terminated, truncated, info = env.step(action)
    return observation


def merge_screens(screen1: ObsType, screen2: ObsType) -> ObsType:
    """
    First, to encode a single frame we take the maximum value for each pixel colour
    value over the frame being encoded and the previous frame. This was necessary to
    remove flickering that is present in games where some objects appear only in even
    frames while other objects appear only in odd frames, an artefact caused by the
    limited number of sprites Atari 2600 can display at once.


    :param screen1: first screen to be merged
    :param screen2: second screen to be merged
    :return: merged screen
    """
    return np.maximum(screen1, screen2)


def extract_luminance(screen: ObsType) -> ObsType:
    """
    Second, we then extract
    the Y channel, also known as luminance, from the RGB frame...

    :param screen: screen to be processed
    :return: processed screen
    """
    return 0.299 * screen[:, :, 0] + 0.587 * screen[:, :, 1] + 0.114 * screen[:, :, 2]


def resize_screen(screen: ObsType) -> ObsType:
    """
    ...and rescale it to 84 x 84.

    :param screen: screen to be resized
    :return: resized screen
    """
    return cv2.resize(screen, (84, 84),
                      interpolation=cv2.INTER_LINEAR)  # TODO: check if this is the correct interpolation


def preprocess_screen(screen: ObsType, previous_screen) -> ObsType:
    """
    Combining all the above steps

    :param screen: screen to be preprocessed
    :param previous_screen: previous screen to be merged with the current screen
    :return: preprocessed screen
    """
    merged_screen = merge_screens(screen, previous_screen)
    luminance = extract_luminance(merged_screen)
    resized_screen = resize_screen(luminance)
    return resized_screen.astype(np.float32) / 255.0  # Normalize the pixel values


def network_input_to_tensor(network_input):
    stacked_observations = np.stack(network_input, axis=0)

    stacked_observations_tensor = torch.from_numpy(stacked_observations)


    return stacked_observations_tensor


def clip(x):
    """
    As the scale of scores varies greatly from game to game, we clipped all positive
    rewards at 1 and all negative rewards at 21, leaving 0 rewards unchanged.
    Clipping the rewards in this manner limits the scale of the error derivatives and
    makes it easier to use the same learning rate across multiple games. At the same time,
    it could affect the performance of our agent since it cannot differentiate between
    rewards of different magnitude
    :param x:
    :return:
    """
    return np.maximum(-1.0, np.minimum(x, 1.0))


env = gym.make("VideoPinballNoFrameskip-v4", render_mode="rgb_array")

penultimate_observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)
initial_observations = [(penultimate_observation, observation)]

penultimate_observation = skip_steps_with_action(env, 0)

for _ in range(hyperparameters.agent_history_length - 1):
    observation, reward, terminated, truncated, info = env.step(0)
    initial_observations.append((penultimate_observation, observation))
    penultimate_observation = skip_steps_with_action(env, 0)

network_input = []
for (penultimate_observation, observation) in initial_observations:
    network_input.append(preprocess_screen(observation, penultimate_observation))

for i, obs in enumerate(initial_observations):
    plt.imsave(f"observation_{i}.png", obs)
for i, obs in enumerate(network_input):
    plt.imsave(f"observation_merged_{i}_reshaped.png", obs)

last_frame_unmerged = observation

exploration_rate = hyperparameters.initial_exploration

network = DQN(9)

memory = ReplayMemory(hyperparameters.replay_memory_size)
input_tensor = network_input_to_tensor(network_input)

has_only_chosen_no_op = True
no_op_chosen_for_frames_count = 0

optimizer = optim.RMSprop(network.parameters(), lr=hyperparameters.learning_rate,
                          alpha=hyperparameters.squared_gradient_momentum, eps=hyperparameters.min_squared_gradient)

for frame in range(50_000_000):
    print(frame)
    if frame < hyperparameters.replay_start_size or np.random.rand() < exploration_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(network.forward(input_tensor).detach().numpy())  # TODO check if forward or __call__

    if has_only_chosen_no_op:
        if action == 0:
            no_op_chosen_for_frames_count += 1
        else:
            has_only_chosen_no_op = False

        if no_op_chosen_for_frames_count >= hyperparameters.no_op_max:
            while action == 0:
                action = env.action_space.sample()
                has_only_chosen_no_op = False

    # STEP
    observation, reward, terminated, truncated, info = env.step(action)

        # MEMORY AND VARIABLE UPDATE
    next_preprocessed_observation = preprocess_screen(observation, last_frame_unmerged)
    new_network_input = network_input[1:] + [next_preprocessed_observation]
    new_input_tensor = network_input_to_tensor(new_network_input)
    reward = clip(reward)
    memory.push(input_tensor, action, new_input_tensor, reward)
    input_tensor = new_input_tensor
    network_input = new_network_input

    # LEARNING
    # if frame >= hyperparameters.replay_start_size and frame % hyperparameters.update_frequency == 0: # TODO: uncomment this and remove lower one
    if frame>33 and frame % hyperparameters.update_frequency == 0:
        minibatch = memory.sample(hyperparameters.minibatch_size)

        # Extract the states, actions, next states, and rewards from the minibatch
        states, actions, next_states, rewards = zip(*minibatch)

        # Convert the data to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Compute the Q-values for the current states and actions
        q_values = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the target Q-values using the target network
        with torch.no_grad():
            next_q_values = network(next_states).max(1)[0]
            target_q_values = rewards + hyperparameters.discount_factor * next_q_values

        # Compute the loss
        loss = torch.nn.functional.smooth_l1_loss(q_values, target_q_values)

        # Perform the optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # SKIP AND REMEMBER LAST FRAME
    last_frame_unmerged = skip_steps_with_action(env, action)

    # ANNEALING
    if frame < hyperparameters.final_exploration_frame:
        exploration_rate = hyperparameters.final_exploration + (
                    hyperparameters.initial_exploration - hyperparameters.final_exploration) * (
                                       hyperparameters.final_exploration_frame - frame) / hyperparameters.final_exploration_frame
    else:
        exploration_rate = hyperparameters.final_exploration

env.close()
