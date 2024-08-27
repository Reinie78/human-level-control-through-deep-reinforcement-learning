import gymnasium as gym
import torch
from gymnasium import Env
import matplotlib.pyplot as plt
import numpy as np
import cv2
from gymnasium.core import ObsType

import hyperparameters
from model import DQN


def skip_steps_with_action(env: Env, action: int) -> ObsType:
   """

   :param env: current environment - pracitacllay a global var
   :param action: - action to be repeated for the steps
   :return: screen after last step (to be merged with the next screen)
   """
   for _ in range(hyperparameters.action_repeat-1):
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
   return 0.299 * screen[:,:,0] + 0.587 * screen[:,:,1] + 0.114 * screen[:,:,2]

def resize_screen(screen: ObsType) -> ObsType:
   """
   ...and rescale it to 84 x 84.

   :param screen: screen to be resized
   :return: resized screen
   """
   return cv2.resize(screen, (84, 84), interpolation=cv2.INTER_LINEAR) # TODO: check if this is the correct interpolation

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


    input_tensor = stacked_observations_tensor.unsqueeze(0)



env = gym.make("VideoPinballDeterministic-v4", render_mode="rgb_array")

penultimate_observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)
initial_observations = [(penultimate_observation, observation)]

penultimate_observation = skip_steps_with_action(env, 0)

for _ in range(hyperparameters.agent_history_length-1):
   observation, reward, terminated, truncated, info = env.step(0)
   initial_observations.append((penultimate_observation, observation))
   penultimate_observation = skip_steps_with_action(env, 0)


network_input = []
for (penultimate_observation, observation) in initial_observations:
   network_input.append(preprocess_screen(observation, penultimate_observation))


for i,obs in enumerate(initial_observations):
   plt.imsave(f"observation_{i}.png", obs)
for i,obs in enumerate(network_input):
   plt.imsave(f"observation_merged_{i}_reshaped.png", obs)

last_frame_unmerged = initial_observations[-1]

exploration_rate = hyperparameters.initial_exploration

network = DQN(9)



network.forward(input_tensor)

# for _ in range(4):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#    plt.imsave(f"observation_{_}.png", observation)
#    if terminated or truncated:
#       observation, info = env.reset()

env.close()
