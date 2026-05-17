"""
Evaluation utilities for DQN-style agents on ALE/Gymnasium environments.

Designed to be architecture-agnostic: works for any nn.Module whose forward
pass returns Q-values of shape [batch, n_actions].
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
import json
import time

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


@dataclass
class EvalResults:
    """Container for evaluation results."""
    mean_return: float
    std_return: float
    median_return: float
    min_return: float
    max_return: float
    mean_length: float
    n_episodes: int
    epsilon: float
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    wall_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Eval over {self.n_episodes} episodes (eps={self.epsilon}): "
            f"return = {self.mean_return:.2f} +/- {self.std_return:.2f} "
            f"(median {self.median_return:.2f}, "
            f"min {self.min_return:.2f}, max {self.max_return:.2f}), "
            f"length = {self.mean_length:.1f}, "
            f"wall_time = {self.wall_time_seconds:.1f}s"
        )


@torch.no_grad()
def select_action(
    network: nn.Module,
    state: np.ndarray,
    n_actions: int,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    """Epsilon-greedy action selection using the online network."""
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))

    # State comes from the env as uint8 frames; convert to float and normalize.
    # Adjust normalization here if your training pipeline differs.
    state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device)
    state_t = state_t.unsqueeze(0) / 255.0  # add batch dim, normalize to [0, 1]

    q_values = network(state_t)
    return int(q_values.argmax(dim=1).item())


def evaluate_agent(
    network: nn.Module,
    env_factory: Callable[[int], gym.Env],
    n_episodes: int = 30,
    epsilon: float = 0.05,
    device: Optional[torch.device] = None,
    seed: int = 0,
    max_steps_per_episode: int = 108_000,  # 30 min at 60 fps, DeepMind default
    verbose: bool = False,
) -> EvalResults:
    """
    Run evaluation episodes and return aggregated statistics.

    Args:
        network: The online (agent) network. Will be set to eval mode.
        env_factory: Callable that takes a seed and returns a fresh env.
                     Letting the caller build the env means this function
                     doesn't need to know about wrappers, render modes, etc.
        n_episodes: Number of evaluation episodes to run.
        epsilon: Exploration rate for eval. 0.05 is the DeepMind default.
        device: Torch device. Inferred from the network if None.
        seed: Base seed; each episode uses seed + episode_index.
        max_steps_per_episode: Hard cap to prevent runaway episodes.
        verbose: Print per-episode results.

    Returns:
        EvalResults with returns, lengths, and summary stats.
    """
    if device is None:
        device = next(network.parameters()).device

    # eval() disables dropout/batchnorm running-stat updates, etc.
    # Important even if your nets don't use them — good habit and safe.
    was_training = network.training
    network.eval()

    rng = np.random.default_rng(seed)
    returns: list[float] = []
    lengths: list[int] = []
    t_start = time.perf_counter()

    try:
        for ep in range(n_episodes):
            # Fresh env per episode with a unique seed so episodes are
            # independent and reproducible.
            env = env_factory(seed + ep)
            n_actions = env.action_space.n

            obs, _info = env.reset(seed=seed + ep)
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = select_action(
                    network, obs, n_actions, epsilon, device, rng
                )
                obs, reward, terminated, truncated, _info = env.step(action)
                episode_return += float(reward)
                episode_length += 1

                if episode_length >= max_steps_per_episode:
                    truncated = True

            env.close()
            returns.append(episode_return)
            lengths.append(episode_length)

            if verbose:
                print(
                    f"  [eval ep {ep + 1}/{n_episodes}] "
                    f"return={episode_return:.1f}, length={episode_length}"
                )
    finally:
        # Always restore training mode, even if eval was interrupted.
        if was_training:
            network.train()

    wall_time = time.perf_counter() - t_start
    returns_arr = np.asarray(returns, dtype=np.float64)

    return EvalResults(
        mean_return=float(returns_arr.mean()),
        std_return=float(returns_arr.std(ddof=1)) if len(returns) > 1 else 0.0,
        median_return=float(np.median(returns_arr)),
        min_return=float(returns_arr.min()),
        max_return=float(returns_arr.max()),
        mean_length=float(np.mean(lengths)),
        n_episodes=n_episodes,
        epsilon=epsilon,
        returns=returns,
        lengths=lengths,
        wall_time_seconds=wall_time,
    )


# ---------------------------------------------------------------------------
# Example env factory and usage
# ---------------------------------------------------------------------------

def make_atari_env_factory(
    env_id: str = "ALE/Breakout-v5",
    frame_stack: int = 4,
    frame_skip: int = 4,
    noop_max: int = 30,
    terminal_on_life_loss: bool = False,  # False for eval, even if True during training
) -> Callable[[int], gym.Env]:
    """
    Build an env factory with the standard DeepMind Atari preprocessing.

    Note terminal_on_life_loss defaults to False here: the convention is to
    *train* with life loss as terminal but *evaluate* on full episodes, since
    that matches how scores are reported in the literature.
    """
    def _factory(seed: int) -> gym.Env:
        env = gym.make(env_id, frameskip=1)  # AtariPreprocessing handles frame skip
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=84,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=True,
            scale_obs=False,  # we scale to [0,1] in select_action
        )
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
        env.action_space.seed(seed)
        return env

    return _factory


def save_eval_results(results: EvalResults, path: str) -> None:
    """Persist eval results to JSON for later analysis / plotting."""
    with open(path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Example of how you'd wire this into a training loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Pseudocode — adapt to your training class
    #
    # env_factory = make_atari_env_factory("ALE/Breakout-v5")
    #
    # for step in range(total_steps):
    #     ...training step...
    #
    #     if step % eval_interval == 0:
    #         results = evaluate_agent(
    #             network=agent.online_net,
    #             env_factory=env_factory,
    #             n_episodes=10,         # cheap periodic eval
    #             epsilon=0.05,
    #             seed=1000 + step,      # held-out seeds (above training range)
    #             verbose=False,
    #         )
    #         metrics_tracker.log("eval/mean_return", results.mean_return, step)
    #         metrics_tracker.log("eval/std_return", results.std_return, step)
    #         save_eval_results(results, f"evals/step_{step}.json")
    #
    #         # Save lightweight checkpoint
    #         torch.save({
    #             "step": step,
    #             "online_state_dict": agent.online_net.state_dict(),
    #             "eval_mean_return": results.mean_return,
    #             "config": agent.config,
    #         }, f"checkpoints/eval_step_{step}.pt")
    #
    # # Final official evaluation on best checkpoint:
    # best_ckpt = load_best_checkpoint("checkpoints/")
    # agent.online_net.load_state_dict(best_ckpt["online_state_dict"])
    # final = evaluate_agent(
    #     network=agent.online_net,
    #     env_factory=env_factory,
    #     n_episodes=100,                # rigorous
    #     epsilon=0.05,
    #     seed=999_000,                  # held-out from training AND periodic eval
    #     verbose=True,
    # )
    # print(final.summary())
    # save_eval_results(final, "evals/final_official.json")
    pass