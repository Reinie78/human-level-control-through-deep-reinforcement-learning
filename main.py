import pickle
import time

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
from eval import evaluate_agent, save_eval_results

#import os
#os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:\\temp\\torch_cache"
#os.environ["TORCHINDUCTOR_DISABLE_CACHE_LOCKING"] = "1"

###TODO:
### - revamp training loop and clean it up (all in one file ew)
### - add version with memory being transferred to device


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
    for _ in range(hyperparameters.action_repeat - 1):
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    return observation, terminated


###############################################
################# PREPROCESSING ###############
###############################################

def merge_screens(screen1: ObsType, screen2: ObsType) -> ObsType:
    return np.maximum(screen1, screen2)


def extract_luminance(screen: ObsType) -> ObsType:
    return 0.299 * screen[:, :, 0] + 0.587 * screen[:, :, 1] + 0.114 * screen[:, :, 2]


def resize_screen(screen: ObsType) -> ObsType:
    return cv2.resize(screen, (84, 84),
                      interpolation=cv2.INTER_LINEAR)


def preprocess_screen(screen: ObsType, previous_screen) -> ObsType:
    merged_screen = merge_screens(screen, previous_screen)
    luminance = extract_luminance(merged_screen)
    resized_screen = resize_screen(luminance)
    return resized_screen.astype(np.float32) / 255.0  # Normalize the pixel values


def network_input_to_tensor(network_input):
    stacked_observations = np.stack(network_input, axis=0)
    return torch.from_numpy(stacked_observations)


def clip(x):
    return np.maximum(-1.0, np.minimum(x, 1.0))

def save_model(agent, path):
    torch.save(agent.state_dict(), path)

def load_model(network, path):
    network.load_state_dict(torch.load(path))
    return network

def args_parse():
    parser = argparse.ArgumentParser(description="Atari: DQN")
    parser.add_argument('--env', default="ALE/Pong-v5", help='Should be NoFrameskip environment')
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

episode_tracker = EpisodeTracker()

exploration_rate = hyperparameters.initial_exploration
#network = DQN(vgname_2_action[args.env])
#network = PCNN(input_shape=(4, 84, 84), num_actions=vgname_2_action[args.env])

main_network, target_network = initialize_networks(vgname_2_action[args.env], use_pcnn=False, device=device_for_network)


#optimizer = optim.RMSprop(main_network.parameters(), lr=hyperparameters.learning_rate,
#                         alpha=hyperparameters.squared_gradient_momentum, eps=hyperparameters.min_squared_gradient)
optimizer = get_optimizer(main_network, use_pcnn=False)



###################################################
#############   MAIN LOOP #########################
###################################################
cpcounter = 0
start_time = 0
should_reset = True
should_use_pickle = False
load_network = False
#start_frame = 0 if not should_use_pickle else hyperparameters.replay_start_size
memory = ReplayMemory(hyperparameters.replay_memory_size)
if should_use_pickle:
    print("loading memory")
    with open(f"memory{hyperparameters.replay_start_size}.pkl", "rb") as pkl:
        memory.memory = pickle.load(pkl)
#input_tensor = network_input_to_tensor(network_input).to(device_for_network)
frame_times = []
best_score = -float('inf')

if load_network:
     main_network = load_model(main_network, "PCNNcheck2v1.pth")
     print("loaded network")
     start_frame = 2000000
else:
    start_frame = 0

for frame in range(start_frame, hyperparameters.TOTAL_FRAMES):

#    if frame == hyperparameters.replay_start_size+1 and not should_use_pickle:
#        with open(f"memory{hyperparameters.replay_start_size}.pkl", "wb") as pkl:
#            pickle.dump(memory.memory, pkl)
#        print("saved")

    if frame % 10000 == 0:
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Clear GPU cache too
        print(f"Frame {frame}: Episodes={episode_tracker.episode_count}")
        if len(episode_tracker.losses) > 0:
            print(f"Recent loss: {episode_tracker.losses[-1]:.6f}")
        if len(episode_tracker.episode_scores) > 0:
            print(f"Recent score: {episode_tracker.episode_scores[-1]:.1f}")

    if should_reset:

        penultimate_observation, _info = env.reset()

        observation, reward, terminated, truncated, info = env.step(0)
        episode_ended, final_score = episode_tracker.step(reward, terminated, truncated, info)

        initial_observations = [(penultimate_observation, observation)]
        last_frame_unmerged = observation
        if not terminated:
            penultimate_observation, terminated = skip_steps_with_action(env, 0)

            for _ in range(hyperparameters.agent_history_length - 1):
                observation, reward, terminated, truncated, info = env.step(0)
                episode_ended, final_score = episode_tracker.step(reward, terminated, truncated, info)
                initial_observations.append((penultimate_observation, observation))
                penultimate_observation, terminated = skip_steps_with_action(env, 0)

            network_input = []
            for (penultimate_observation, observation) in initial_observations:
                network_input.append(preprocess_screen(observation, penultimate_observation))

            last_frame_unmerged = observation
            input_tensor = network_input_to_tensor(network_input).to(device_for_network)
            has_only_chosen_no_op = True
            no_op_chosen_for_frames_count = 0

        should_reset = False

    if frame < hyperparameters.replay_start_size or np.random.rand() < exploration_rate:
        action = env.action_space.sample()
    else:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            action = main_network(input_tensor.unsqueeze(0)).argmax().item()

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
    memory.push(input_tensor, action, new_input_tensor, reward, terminated)
    input_tensor = input_tensor.to(device_for_network)


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

        #with autocast(device_type='cuda', dtype=torch.float16):

        # Compute the target Q-values using the target network
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16): #previously used torch.no_grad
                next_q_values = target_network(next_states).max(1)[0]
                target_q_values = rewards + (hyperparameters.discount_factor * next_q_values * ~dones)
#            episode_tracker.add_q_values(next_q_values)

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
    if terminated or truncated:
        should_reset = True

    if frame > 0 and frame % 1000000 == 0:
        print(f"Running eval at frame {frame}...")
        eval_results = evaluate_agent(
            main_network,
            env,
            device=device_for_network,
            seed=1_000_000 + frame,  # held-out from training
            verbose=True,
        )
        save_eval_results(eval_results, f"DQNevals/DQNeval_frame_{frame}.json")
        save_model(main_network, f"DQNcheckpoints/DQNcheck{cpcounter}v2.pth")
        cpcounter += 1
        print("Eval ended")
        should_reset = True

#print(episode_tracker._extract_official_score())

episode_tracker.print_stats()
env.close()

