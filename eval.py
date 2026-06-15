"""
Evaluation function for the DQN/PCNN agent trained in main.py.

Replicates the same preprocessing pipeline as the training loop (action
repeat, max of two consecutive frames, luminance + 84x84 resize, 4-frame
stack) so the network sees inputs in the exact format it was trained on,
and reuses the env that main.py already built — no separate factory.

DeepMind protocol (Mnih et al., 2015): 30 episodes per evaluation, each
capped at 5 minutes of emulator time (~18,000 raw frames at 60 fps),
epsilon-greedy with eps = 0.05.
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env

import hyperparameters


# ---------------------------------------------------------------------------
# Preprocessing — kept in sync with main.py.
#
# These are intentionally duplicated rather than imported from main.py because
# main.py runs the training loop at module level; importing from it would
# kick off training. Once main.py is refactored into functions, these can be
# replaced with imports.
# ---------------------------------------------------------------------------

def _merge_screens(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return np.maximum(s1, s2)


def _extract_luminance(s: np.ndarray) -> np.ndarray:
    return 0.299 * s[:, :, 0] + 0.587 * s[:, :, 1] + 0.114 * s[:, :, 2]


def _resize_screen(s: np.ndarray) -> np.ndarray:
    return cv2.resize(s, (84, 84), interpolation=cv2.INTER_LINEAR)


def _preprocess_screen(screen: np.ndarray, previous_screen: np.ndarray) -> np.ndarray:
    merged = _merge_screens(screen, previous_screen)
    lum = _extract_luminance(merged)
    return _resize_screen(lum).astype(np.float32) / 255.0


def _stack_to_tensor(stack: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(stack, axis=0))


def _skip_with_action(env: Env, action: int, n: int):
    """Repeat `action` n times, accumulating reward.

    Returns (last_observation, terminated, truncated, summed_reward). Stops
    early on terminated/truncated, matching the training loop's behaviour but
    also accumulating reward (which the training loop's helper discards).
    """
    last_obs = None
    summed_reward = 0.0
    terminated = False
    truncated = False
    for _ in range(n):
        last_obs, reward, terminated, truncated, _info = env.step(action)
        summed_reward += float(reward)
        if terminated or truncated:
            break
    return last_obs, terminated, truncated, summed_reward


@torch.inference_mode()
def _select_action(
    network: nn.Module,
    input_tensor: torch.Tensor,
    n_actions: int,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    # Plain fp32, contiguous input. No autocast in eval — cuDNN's "FIND was
    # unable to find an engine" error tends to fire when benchmark mode +
    # autocast + a freshly switched eval-mode network all hit the same call.
    x = input_tensor.unsqueeze(0).contiguous()
    q_values = network(x)
    return int(q_values.argmax(dim=1).item())


# ---------------------------------------------------------------------------
# Public eval entry point
# ---------------------------------------------------------------------------

def evaluate_agent(
    network: nn.Module,
    env: Env,
    n_episodes: int = 30,
    epsilon: float = 0.05,
    max_episode_seconds: float = 300.0,
    device: torch.device | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Run a held-out evaluation on the provided env.

    The env is the same one main.py uses for training — it is reset between
    episodes, so this is safe to call mid-training or after it finishes.

    Args:
        network: Online network (DQN or PCNN); forward returns Q-values.
        env: A gymnasium env configured the same way as in training.
        n_episodes: 30 matches the DeepMind protocol.
        epsilon: 0.05 matches the DeepMind protocol.
        max_episode_seconds: Per-episode cap; 5 min * 60 fps = 18,000 raw frames.
        device: Defaults to whatever device the network parameters live on.
        seed: Base seed; episode k uses seed + k.
        verbose: Print per-episode results.

    Returns:
        A dict with aggregate stats, per-episode returns/lengths, and wall time.
    """
    if device is None:
        device = next(network.parameters()).device

    was_training = network.training
    network.eval()

    # cuDNN's "FIND was unable to find an engine ... 0 plans" error fires
    # when benchmark mode is on and a new (shape, dtype, mode) combo hits the
    # planner mid-run. Eval is cheap enough that we can just turn benchmark
    # off for the duration and restore the user's setting afterwards.
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False

    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    action_repeat = hyperparameters.action_repeat
    history_length = hyperparameters.agent_history_length
    no_op_max = hyperparameters.no_op_max
    max_episode_frames = int(max_episode_seconds * 60.0)  # 60 fps emulator

    returns: list[float] = []
    lengths: list[int] = []
    t_start = time.perf_counter()

    try:
        for ep in range(n_episodes):
            episode_seed = seed + ep
            penultimate_obs, _info = env.reset(seed=episode_seed)

            episode_return = 0.0
            frames_consumed = 0

            # First raw step — mirrors the reset block in main.py's training loop.
            obs, reward, terminated, truncated, _info = env.step(0)
            episode_return += float(reward)
            frames_consumed += 1

            initial_observations = [(penultimate_obs, obs)]
            last_frame_unmerged = obs

            if not (terminated or truncated):
                penultimate_obs, terminated, truncated, r = _skip_with_action(
                    env, 0, action_repeat - 1
                )
                episode_return += r
                frames_consumed += action_repeat - 1

                # Fill the rest of the frame stack with no-ops.
                for _ in range(history_length - 1):
                    if terminated or truncated:
                        break
                    obs, reward, terminated, truncated, _info = env.step(0)
                    episode_return += float(reward)
                    frames_consumed += 1
                    initial_observations.append((penultimate_obs, obs))
                    if terminated or truncated:
                        break
                    penultimate_obs, terminated, truncated, r = _skip_with_action(
                        env, 0, action_repeat - 1
                    )
                    episode_return += r
                    frames_consumed += action_repeat - 1

                network_input = [
                    _preprocess_screen(o, p) for (p, o) in initial_observations
                ]
                # If the episode ended before the stack was full, pad with zeros
                # so the network input still has the expected shape.
                while len(network_input) < history_length:
                    network_input.append(np.zeros_like(network_input[-1]))

                last_frame_unmerged = obs
                input_tensor = _stack_to_tensor(network_input).to(device)

                # no-op-max guard (same as training): force a non-no-op action
                # once the agent has selected no-op too many times in a row.
                has_only_chosen_no_op = True
                no_op_chosen_for_frames_count = 0

                while not (terminated or truncated):
                    action = _select_action(
                        network, input_tensor, n_actions, epsilon, device, rng
                    )

                    if has_only_chosen_no_op:
                        if action == 0:
                            no_op_chosen_for_frames_count += 1
                        else:
                            has_only_chosen_no_op = False
                        if no_op_chosen_for_frames_count >= no_op_max:
                            while action == 0:
                                action = int(rng.integers(0, n_actions))
                            has_only_chosen_no_op = False

                    obs, reward, terminated, truncated, _info = env.step(action)
                    episode_return += float(reward)
                    frames_consumed += 1

                    next_preprocessed = _preprocess_screen(obs, last_frame_unmerged)
                    network_input = network_input[1:] + [next_preprocessed]
                    input_tensor = _stack_to_tensor(network_input).to(device)

                    if not (terminated or truncated):
                        last_frame_unmerged, terminated, truncated, r = _skip_with_action(
                            env, action, action_repeat - 1
                        )
                        episode_return += r
                        frames_consumed += action_repeat - 1

                    if frames_consumed >= max_episode_frames:
                        truncated = True

            returns.append(episode_return)
            lengths.append(frames_consumed)
            if verbose:
                print(
                    f"  [eval ep {ep + 1}/{n_episodes}] "
                    f"return={episode_return:.1f}, frames={frames_consumed}"
                )
    finally:
        torch.backends.cudnn.benchmark = prev_benchmark
        if was_training:
            network.train()

    wall_time = time.perf_counter() - t_start
    returns_arr = np.asarray(returns, dtype=np.float64)

    n = len(returns_arr)
    std_return = float(returns_arr.std(ddof=1)) if n > 1 else 0.0
    std_error = std_return / np.sqrt(n) if n > 0 else 0.0
    # 95% normal-approximation confidence interval for the mean return.
    ci95_half = 1.96 * std_error
    mean_return = float(returns_arr.mean()) if n > 0 else 0.0

    results = {
        "n_episodes": n_episodes,
        "epsilon": epsilon,
        "seed": seed,
        "mean_return": mean_return,
        "std_return": std_return,
        "std_error": float(std_error),
        "ci95_half_width": float(ci95_half),
        "ci95_low": float(mean_return - ci95_half),
        "ci95_high": float(mean_return + ci95_half),
        "median_return": float(np.median(returns_arr)),
        "min_return": float(returns_arr.min()),
        "max_return": float(returns_arr.max()),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "lengths": lengths,
        "wall_time_seconds": wall_time,
    }

    if verbose:
        print(
            f"Eval over {n_episodes} episodes (eps={epsilon}): "
            f"return = {results['mean_return']:.2f} +/- {results['std_return']:.2f} "
            f"(95% CI [{results['ci95_low']:.2f}, {results['ci95_high']:.2f}], "
            f"median {results['median_return']:.2f}, "
            f"min {results['min_return']:.2f}, max {results['max_return']:.2f}), "
            f"mean length = {results['mean_length']:.1f} frames, "
            f"wall_time = {wall_time:.1f}s"
        )

    return results


# ---------------------------------------------------------------------------
# Run an evaluation from a saved weights file (checkpoint).
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    weights_path: str,
    model: str = "pcnn",
    env_id: str = "ALE/Pong-v5",
    n_episodes: int = 50,
    epsilon: float = 0.05,
    seed: int = 0,
    device: torch.device | None = None,
    out_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Load a trained weight set from disk and evaluate it.

    Builds the env and network exactly as training does, loads the state_dict
    from `weights_path`, runs evaluate_agent(), and (optionally) saves the
    results JSON to `out_path`.

    Args:
        weights_path: Path to a .pth state_dict saved by main.py's save_model().
        model: "dqn" or "pcnn" -- which architecture the weights belong to.
        env_id: Gymnasium env id (must match the training env).
        n_episodes: Number of eval episodes (50 gives a tighter CI than 30).
        epsilon: Evaluation epsilon (0.05 = DeepMind protocol).
        seed: Base seed; episode k uses seed + k. Use the SAME seed across the
              models/runs you compare so they face identical start conditions.
        device: Torch device; defaults to cuda if available.
        out_path: If given, the results dict is written there as JSON.
        verbose: Print per-episode and summary lines.

    Returns:
        The results dict from evaluate_agent (with CI fields).
    """
    import gymnasium as gym
    import ale_py  # noqa: F401  (registers ALE envs as a side effect)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_id, render_mode="rgb_array", repeat_action_probability=0.0)
    n_actions = env.action_space.n

    model = model.lower()
    if model == "pcnn":
        from PCNNmodel import PCNN
        network = PCNN(input_shape=(4, 84, 84), num_actions=n_actions)
    elif model == "dqn":
        from DQNmodel import DQN
        network = DQN(n_actions)
    else:
        raise ValueError(f"Unknown model '{model}', expected 'dqn' or 'pcnn'.")

    state_dict = torch.load(weights_path, map_location=device)
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()

    if verbose:
        print(f"Loaded {model.upper()} weights from {weights_path}")
        print(f"Evaluating on {env_id}: {n_episodes} episodes, eps={epsilon}, seed={seed}")

    results = evaluate_agent(
        network,
        env,
        n_episodes=n_episodes,
        epsilon=epsilon,
        device=device,
        seed=seed,
        verbose=verbose,
    )
    results["model"] = model
    results["env"] = env_id
    results["weights_path"] = weights_path

    if out_path is not None:
        save_eval_results(results, out_path)
        if verbose:
            print(f"Saved eval results to {out_path}")

    env.close()
    return results


def _build_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Evaluate a trained DQN/PCNN checkpoint from a weights file."
    )
    p.add_argument("weights", help="Path to the .pth state_dict to evaluate.")
    p.add_argument("--model", default="pcnn", choices=["dqn", "pcnn"],
                   help="Architecture the weights belong to (default: pcnn).")
    p.add_argument("--env", default="ALE/Pong-v5", help="Gymnasium env id.")
    p.add_argument("--n-episodes", type=int, default=50,
                   help="Number of evaluation episodes (default: 50).")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="Evaluation epsilon (default: 0.05).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; keep identical across compared runs.")
    p.add_argument("--out", default=None,
                   help="Optional path to write the results JSON.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-episode prints.")
    return p


if __name__ == "__main__":
    _args = _build_cli().parse_args()
    evaluate_checkpoint(
        weights_path=_args.weights,
        model=_args.model,
        env_id=_args.env,
        n_episodes=_args.n_episodes,
        epsilon=_args.epsilon,
        seed=_args.seed,
        out_path=_args.out,
        verbose=not _args.quiet,
    )


def save_eval_results(results: dict, path: str) -> None:
    """Persist eval results to a JSON file.

    Creates parent directories if missing. If `path` ends with ".jsonl"
    the entry is appended as a single JSON line (useful for keeping a
    rolling log across many evals); otherwise a standalone JSON file
    is written, overwriting any existing one at that path.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if path.endswith(".jsonl"):
        with open(path, "a") as f:
            f.write(json.dumps(results) + "\n")
    else:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
