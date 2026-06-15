import os
import time
from collections import deque

import numpy as np
from torch import mean as tensor_mean


class EpisodeTracker:
    """Tracks per-episode training metrics.

    Correct usage (see main.py):
        tracker.reset_current()          # at the start of every fresh episode
        tracker.record_step(r, frames)   # once per environment transition
        tracker.record_loss(loss)        # once per optimisation step
        tracker.end_episode()            # exactly once, when terminated/truncated

    The previous implementation logged an episode whenever step() happened to be
    called with terminated=True. Because Pong terminations frequently land on a
    frame-skip step (not the main step), the terminal was often missed and the
    score/length kept accumulating across several games -- producing the
    inflated values (scores in the hundreds, lengths far exceeding one game).
    Splitting reward accumulation (record_step) from finalisation (end_episode)
    removes that bug: each real episode is closed exactly once.
    """

    def __init__(self, metrics_dir="DQNmetrics", save_every=100):
        self.metrics_dir = metrics_dir
        self.save_every = save_every

        self.current_episode_score = 0.0
        self.current_episode_length = 0
        self.episode_count = 0
        self.episodes = []
        self.best_score = float("-inf")
        self.csv_saved = 0

        # Per-episode optimisation losses (cleared at each end_episode()).
        self.losses = []

        # Optional extra diagnostics kept from the original tracker.
        self.q_values = []
        self.exploration_rates = deque(maxlen=1000)

        # Timing: wall-clock between consecutive episode ends.
        self.start_time = time.time()
        self.total_frames = 0
        self.last_log_time = time.time()
        self.last_log_frame = 0

        os.makedirs(self.metrics_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def reset_current(self):
        """Discard any in-progress (e.g. eval-interrupted) partial episode.

        A normally finished episode already has these counters at zero after
        end_episode(), so calling this at the top of a fresh episode is a no-op
        in the common case and only matters after a forced mid-game reset.
        """
        self.current_episode_score = 0.0
        self.current_episode_length = 0
        self.losses.clear()

    def record_step(self, reward, frames=1):
        """Accumulate reward and frame count for the current episode."""
        self.current_episode_score += float(reward)
        self.current_episode_length += int(frames)
        self.total_frames += int(frames)

    def record_loss(self, loss):
        self.losses.append(float(loss))

    def add_q_values(self, q_values):
        self.q_values.append(tensor_mean(q_values.cpu()))

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------
    def end_episode(self):
        """Finalise the current episode exactly once, log it, and reset."""
        time_taken = time.time() - self.start_time
        self.start_time = time.time()

        final_score = self.current_episode_score

        if len(self.losses) > 0:
            loss_mean = float(np.mean(self.losses))
            loss_std = float(np.std(self.losses))
        else:
            loss_mean = 0.0
            loss_std = 0.0

        def sanitize(v):
            return -1.0 if (np.isnan(v) or np.isinf(v)) else float(v)

        self.episodes.append(
            (
                final_score,
                self.current_episode_length,
                time_taken,
                sanitize(loss_mean),
                sanitize(loss_std),
            )
        )
        self.episode_count += 1

        if final_score > self.best_score:
            self.best_score = final_score

        if self.episode_count % self.save_every == 0:
            self.save_everything_to_csv()

        # Reset for the next episode.
        self.current_episode_score = 0.0
        self.current_episode_length = 0
        self.losses.clear()
        self.q_values.clear()

        return final_score

    def save_everything_to_csv(self):
        path = os.path.join(self.metrics_dir, f"save_{self.csv_saved}.csv")
        np.savetxt(path, self.episodes, delimiter=",")
        self.csv_saved += 1
        self.episodes = []

    # ------------------------------------------------------------------
    # Backward-compatible helper (kept so old callers don't crash).
    # ------------------------------------------------------------------
    def step(self, reward, terminated, truncated, info=None, frames=1):
        """Deprecated combined call. Prefer record_step()/end_episode().

        Records the reward and, if the transition is terminal, finalises the
        episode. Kept only for compatibility with code paths that still pass a
        full (reward, terminated, truncated) tuple.
        """
        self.record_step(reward, frames=frames)
        final_score = None
        episode_ended = bool(terminated or truncated)
        if episode_ended:
            final_score = self.end_episode()
        return episode_ended, final_score

    # ------------------------------------------------------------------
    # Stats / reporting
    # ------------------------------------------------------------------
    @property
    def episode_scores(self):
        return [x[0] for x in self.episodes]

    def get_recent_training_stats(self, window=100):
        stats = {
            "has_training_data": len(self.losses) > 0,
            "avg_loss": None,
            "avg_q_value": None,
            "exploration_rate": None,
        }
        if len(self.losses) > 0:
            stats["avg_loss"] = float(np.mean(self.losses[-window:]))
        if len(self.q_values) > 0:
            stats["avg_q_value"] = float(np.mean([float(q) for q in self.q_values[-window:]]))
        if len(self.exploration_rates) > 0:
            stats["exploration_rate"] = self.exploration_rates[-1]
        return stats

    def print_comprehensive_stats(self, frame=None):
        print(f"\n{'=' * 60}")
        print("COMPREHENSIVE TRAINING STATISTICS")
        if frame:
            print(f"Frame: {frame:,}")
        print(f"{'=' * 60}")

        scores = [x[0] for x in self.episodes]
        lengths = [x[1] for x in self.episodes]
        if len(scores) > 0:
            print("\nEpisode Statistics:")
            print(f"   Episodes completed: {self.episode_count}")
            print(f"   Mean score (last {len(scores)}): {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
            print(f"   Best score: {self.best_score:.1f}")
            print(f"   Mean episode length: {np.mean(lengths):.1f}")
            print(f"   Last 5 scores: {[f'{s:.1f}' for s in scores[-5:]]}")

        training_stats = self.get_recent_training_stats()
        if training_stats["has_training_data"]:
            print("\nTraining Metrics (recent averages):")
            if training_stats["avg_loss"] is not None:
                print(f"   Loss: {training_stats['avg_loss']:.6f}")
            if training_stats["exploration_rate"] is not None:
                print(f"   Exploration Rate: {training_stats['exploration_rate']:.3f}")
