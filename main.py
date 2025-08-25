import gymnasium as gym
import gc
import ale_py
import torch
from gymnasium import Env
import matplotlib.pyplot as plt
import numpy as np
import cv2
from gymnasium.core import ObsType
from torch import optim
import argparse
from game2netInput import vgname_2_action

import hyperparameters
from memory import ReplayMemory
from DQNmodel import DQN
from PCNNmodel import PCNN
from trainingOut import EpisodeTracker

#import os
#os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:\\temp\\torch_cache"
#os.environ["TORCHINDUCTOR_DISABLE_CACHE_LOCKING"] = "1"


def initialize_networks(num_actions,device, use_pcnn=False):
    if use_pcnn:
        #from PCNNmodel import PCNN
        main_network = PCNN(input_shape=(4, 84, 84), num_actions=num_actions)
        target_network = PCNN(input_shape=(4, 84, 84), num_actions=num_actions)
    else:
        #from model import DQN
        main_network = DQN(num_actions)
        target_network = DQN(num_actions)


    # Initialize target network with same weights as main network
    target_network.load_state_dict(main_network.state_dict())
    target_network.eval()  # Target network is always in eval mode

    #Make the networks use the GPU
    main_network.to(device)
    target_network.to(device)

#    if hasattr(torch, 'compile'):
#        try:
#            main_network = torch.compile(main_network, mode="reduce-overhead", fullgraph=True)
#            target_network = torch.compile(target_network, mode="reduce-overhead", fullgraph=True)
#            print("Models compiled successfully for better GPU performance")
#        except Exception as e:
#            print(f"Model compilation failed, continuing without: {e}")

    return main_network, target_network


def get_optimizer(model, use_pcnn=False):
    """Get optimal optimizer based on model type"""
    if use_pcnn:
        # AdamW often works better for complex models like PCNN
        # Higher learning rate for PCNN since it has more parameters to train
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparameters.learning_rate * 2,  # Slightly higher for PCNN
            weight_decay=1e-4,
            eps=1e-8
        )
        print("Using AdamW optimizer for PCNN")
    else:
        # Original RMSprop for DQN
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=hyperparameters.learning_rate,
            alpha=hyperparameters.squared_gradient_momentum,
            eps=hyperparameters.min_squared_gradient
        )
        print("Using RMSprop optimizer for DQN")

    return optimizer


def update_target_network(main_network, target_network):
    """Hard update: copy weights from main to target network"""
    target_network.load_state_dict(main_network.state_dict())


def soft_update_target_network(main_network, target_network, tau=0.005):
    """Soft update: slowly blend main network weights into target network"""
    for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


def skip_steps_with_action(env: Env, action: int) -> ObsType:
    """

    :param env: current environment - pracitacllay a global var
    :param action: - action to be repeated for the steps
    :return: screen after last step (to be merged with the next screen)
    """
    for _ in range(hyperparameters.action_repeat - 1):
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    return observation, terminated


###############################################
################# PREPROCESSING ###############
###############################################

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
                      interpolation=cv2.INTER_LINEAR)


# W tym robimy resiza i merge dwóch ekranów żeby zrobić ekran przejścia na podstawie luminance
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
    return torch.from_numpy(stacked_observations)


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

def save_model(agent, path):
    torch.save(agent.state_dict(), path)

def load_model(path):
    model = main_network()
    model.load_state_dict(torch.load(path))
    return model

# TODO: dodaj argument definiujący sieć z której korzystamy
def args_parse():
    #print(gym.envs.registry.keys())
    parser = argparse.ArgumentParser(description="Atari: DQN")
    parser.add_argument('--env', default="ALE/VideoPinball-v5", help='Should be NoFrameskip environment')
    parser.add_argument('--train', action="store_true", help='Train agent with given environment')
    #parser.add_argument('--PCNN', action="store_true")
    #parser.add_argument('--play', help="Play with a given weight directory")
    #parser.add_argument('--log_interval', default=100, help="Interval of logging stdout", type=int)
    #parser.add_argument('--save_weight_interval', default=1000, help="Interval of saving weights", type=int)
    args = parser.parse_args()
    return args

device_for_network = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

args = args_parse()

env = gym.make(args.env, render_mode="rgb_array")

#metrics = TrainingMetrics() ##TODO metrics tracking here
#current_episode_score = 0

episode_tracker = EpisodeTracker()

exploration_rate = hyperparameters.initial_exploration
#network = DQN(vgname_2_action[args.env])
#network = PCNN(input_shape=(4, 84, 84), num_actions=vgname_2_action[args.env])

main_network, target_network = initialize_networks(vgname_2_action[args.env], use_pcnn=False, device=device_for_network)

memory = ReplayMemory(hyperparameters.replay_memory_size)
#optimizer = optim.RMSprop(main_network.parameters(), lr=hyperparameters.learning_rate,
#                         alpha=hyperparameters.squared_gradient_momentum, eps=hyperparameters.min_squared_gradient)
optimizer = get_optimizer(main_network, use_pcnn=False)



###################################################
#############   MAIN LOOP #########################
###################################################
should_reset = True
frame_times = []
best_score = -float('inf')

for frame in range(hyperparameters.TOTAL_FRAMES):
#    if frame%100 == 0:
#        print(frame)

    if frame % 10000 == 0:
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Clear GPU cache too
        print(f"Frame {frame}: Episodes={episode_tracker.episode_count}")
        if len(episode_tracker.losses) > 0:
            print(f"Recent loss: {episode_tracker.losses[-1]:.6f}")
        if len(episode_tracker.episode_scores) > 0:
            print(f"Recent score: {episode_tracker.episode_scores[-1]:.1f}")

    if should_reset:
        penultimate_observation, info = env.reset()

        observation, reward, terminated, truncated, info = env.step(1)
        episode_ended, final_score = episode_tracker.step(reward, terminated, truncated, info)

        initial_observations = [(penultimate_observation, observation)]
        last_frame_unmerged = observation
        if not terminated:
            penultimate_observation, terminated = skip_steps_with_action(env, 0)

#        if not terminated:
            for _ in range(hyperparameters.agent_history_length - 1):
                observation, reward, terminated, truncated, info = env.step(0)
 #               episode_ended, final_score = episode_tracker.step(reward, terminated, truncated, info)

                initial_observations.append((penultimate_observation, observation))
                penultimate_observation, terminated = skip_steps_with_action(env, 0)

#        if not terminated:
            network_input = []
            for (penultimate_observation, observation) in initial_observations:
                network_input.append(preprocess_screen(observation, penultimate_observation))

            #for i, obs in enumerate(initial_observations): #TODO odpal pare razy z tym coby były zdjęcia do pracy ALBO ściągnij framy z neta
            #    plt.imsave(f"observation_{i}.png", obs)
            # i, obs in enumerate(network_input):
            #    plt.imsave(f"observation_merged_{i}_reshaped.png", obs)

            last_frame_unmerged = observation
            input_tensor = network_input_to_tensor(network_input).to(device_for_network)
            has_only_chosen_no_op = True
            no_op_chosen_for_frames_count = 0

        should_reset = terminated

    if frame < hyperparameters.replay_start_size or np.random.rand() < exploration_rate:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            #action = np.argmax(main_network.forward(input_tensor).detach().numpy())  # TODO check if forward or __call__
            action = main_network(input_tensor.unsqueeze(0)).argmax().item()
            #q_values = main_network(input_tensor.unsqueeze(0))  # Add batch dimension TODO figure this out
            #action = q_values.argmax().item()

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
    episode_ended, final_score = episode_tracker.step(reward, terminated, truncated, info)



#    if 'episode' in info and 'r' in info['episode']:
#        episode_score = info['episode']['r']
#        print(f"Episode finished with score: {episode_score}")

        # MEMORY AND VARIABLE UPDATE

    next_preprocessed_observation = preprocess_screen(observation, last_frame_unmerged)
    new_network_input = network_input[1:] + [next_preprocessed_observation]
    new_input_tensor = network_input_to_tensor(new_network_input)
    reward = clip(reward)

    input_tensor = new_input_tensor
    input_tensor = input_tensor.to(device_for_network)
    memory.push(input_tensor.cpu(), action, new_input_tensor, reward, terminated)

    network_input = new_network_input

    ##############################################
    ########### REINFORCEMENT LEARNING ###########
    ##############################################
    if frame >= hyperparameters.replay_start_size:

        minibatch = memory.sample(hyperparameters.minibatch_size)

        # Extract the states, actions, next states, and rewards from the minibatch
        states, actions, next_states, rewards, dones = zip(*minibatch)

        # Convert the data to tensors
        states = torch.stack(states).to(device_for_network, non_blocking=True)
        actions = torch.tensor(actions, dtype=torch.long, device=device_for_network)
        next_states = torch.stack(next_states).to(device_for_network, non_blocking=True)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device_for_network)
        dones = torch.tensor(dones, dtype=torch.bool, device=device_for_network)

        # Compute the Q-values for the current states and actions
        q_values = main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)


        # Compute the target Q-values using the target network
        with torch.no_grad():
            next_q_values = target_network(next_states).max(1)[0]
            target_q_values = rewards + (hyperparameters.discount_factor * next_q_values * ~dones)

        # Compute the loss
        loss = torch.nn.functional.smooth_l1_loss(q_values, target_q_values)

        # Perform the optimization step
        optimizer.zero_grad()
        loss.backward()

#        for param in main_network.parameters():
#            if param.grad is not None:
#                total_grad_norm += param.grad.data.norm(2).item() ** 2
#        total_grad_norm = total_grad_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(main_network.parameters(), max_norm=10.0)

        optimizer.step()
        episode_tracker.losses.append(loss.item())

        if hasattr(hyperparameters, 'USE_SOFT_UPDATE') and hyperparameters.USE_SOFT_UPDATE:
            # Soft update every step
            soft_update_target_network(main_network, target_network, hyperparameters.SOFT_UPDATE_TAU)
        elif frame % hyperparameters.TARGET_UPDATE_FREQUENCY == 0:
            # Hard update every N steps
            print(f"Updating target network at frame {frame}")
            update_target_network(main_network, target_network)

 #   episode_tracker.log_training_step(
 #       loss=loss.item(),
 #       q_values=q_values,  # or q_values from action selection
 #       td_error=torch.abs(q_values - target_q_values),
 #       gradient_norm=total_grad_norm,
 #       exploration_rate=exploration_rate
  #  )

    # SKIP AND REMEMBER LAST FRAME
    if not terminated:
        last_frame_unmerged, terminated = skip_steps_with_action(env, action)

    # ANNEALING
    if frame < hyperparameters.final_exploration_frame:
        exploration_rate = hyperparameters.final_exploration + (
                    hyperparameters.initial_exploration - hyperparameters.final_exploration) * (
                                       hyperparameters.final_exploration_frame - frame) / hyperparameters.final_exploration_frame
    else:
        exploration_rate = hyperparameters.final_exploration
    if terminated:
        should_reset = True

#print(episode_tracker._extract_official_score())
episode_tracker.print_stats()
env.close()

