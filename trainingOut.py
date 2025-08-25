import numpy as np
from collections import deque
import time


class EpisodeTracker:
    def __init__(self):
        self.current_episode_score = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.episode_scores = deque(maxlen=100)  # Keep last 100 episodes
        self.episode_lengths = deque(maxlen=100)
        self.best_score = float('-inf')

        # Training metrics
        self.losses = deque(maxlen=1000)
        self.q_values = deque(maxlen=1000)
        self.td_errors = deque(maxlen=1000)
        self.gradient_norms = deque(maxlen=1000)

        # Action tracking
        self.action_counts = {}
        self.total_actions = 0

        # Performance tracking
        self.start_time = time.time()
        self.total_frames = 0
        self.last_log_time = time.time()
        self.last_log_frame = 0

        # Exploration tracking
        self.exploration_rates = deque(maxlen=1000)

    def _extract_official_score(self, info):
        """Try to extract official score from info dict"""
        if not isinstance(info, dict):
            return None

        # Method 1: Direct episode info (newer gym versions)
        if 'episode' in info:
            episode_info = info['episode']
            if isinstance(episode_info, dict):
                if 'r' in episode_info:
                    return episode_info['r']
                if 'total_reward' in episode_info:
                    return episode_info['total_reward']

        # Method 2: ALE specific info
        if 'ale.lives' in info and 'ale.score' in info:
            return info['ale.score']

        # Method 3: Other common keys
        for key in ['score', 'total_reward', 'episode_reward', 'return']:
            if key in info:
                return info[key]

        return None

    def step(self, reward, terminated, truncated, info):
        """Call this after every env.step()"""
        self.current_episode_score += reward
        self.current_episode_length += 1

        episode_ended = terminated or truncated
        final_score = None

        if episode_ended:
            # Episode finished - log the score
            final_score = self.current_episode_score
            self.episode_scores.append(final_score)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1

            if final_score > self.best_score:
                self.best_score = final_score

            '''# Try multiple ways to get the "official" score from info
            official_score = self._extract_official_score(info)

            recent_stats = self.get_recent_training_stats()

            if official_score is not None:
                print(f"Episode {self.episode_count}: Official Score = {official_score}, "
                      f"Tracked Score = {final_score}, Length = {self.current_episode_length}")

            if recent_stats['has_training_data']:
                print(f"   Recent Loss: {recent_stats['avg_loss']:.6f}")
                if recent_stats['avg_q_value'] is not None:
                    print(f"   Avg Q-value: {recent_stats['avg_q_value']:.4f}")
                if recent_stats['exploration_rate'] is not None:
                    print(f"   Exploration: {recent_stats['exploration_rate']:.3f}")'''

            # Reset for next episode
            self.current_episode_score = 0
            self.current_episode_length = 0

        return episode_ended, final_score

    def log_action(self, action):
        """Log action taken - call after action selection"""
        if action not in self.action_counts:
            self.action_counts[action] = 0
        self.action_counts[action] += 1
        self.total_actions += 1

    def get_recent_training_stats(self, window=100):
        """Get recent training statistics"""
        stats = {
            'has_training_data': len(self.losses) > 0,
            'avg_loss': None,
            'avg_q_value': None,
            'avg_td_error': None,
            'avg_gradient_norm': None,
            'exploration_rate': None
        }

        if len(self.losses) > 0:
            recent_losses = list(self.losses)[-window:]
            stats['avg_loss'] = np.mean(recent_losses)

        if len(self.q_values) > 0:
            recent_q = list(self.q_values)[-window:]
            stats['avg_q_value'] = np.mean(recent_q)

        if len(self.td_errors) > 0:
            recent_td = list(self.td_errors)[-window:]
            stats['avg_td_error'] = np.mean(recent_td)

        if len(self.gradient_norms) > 0:
            recent_grad = list(self.gradient_norms)[-window:]
            stats['avg_gradient_norm'] = np.mean(recent_grad)

        if len(self.exploration_rates) > 0:
            stats['exploration_rate'] = self.exploration_rates[-1]

        return stats

    def get_action_distribution(self):
        """Get current action distribution"""
        if self.total_actions == 0:
            return {}

        distribution = {}
        for action, count in self.action_counts.items():
            distribution[action] = {
                'count': count,
                'percentage': (count / self.total_actions) * 100
            }
        return distribution

    def print_comprehensive_stats(self, frame=None):
        """Print comprehensive statistics"""
        print(f"\n{'=' * 60}")
        print(f"📊 COMPREHENSIVE TRAINING STATISTICS")
        if frame:
            print(f"Frame: {frame:,}")
        print(f"{'=' * 60}")

        # Episode statistics
        if len(self.episode_scores) > 0:
            print(f"\n🎮 Episode Statistics:")
            print(f"   Episodes completed: {self.episode_count}")
            print(
                f"   Mean score (last {len(self.episode_scores)}): {np.mean(self.episode_scores):.2f} ± {np.std(self.episode_scores):.2f}")
            print(f"   Best score: {self.best_score:.1f}")
            print(f"   Mean episode length: {np.mean(self.episode_lengths):.1f}")

            # Recent episode scores
            recent_scores = list(self.episode_scores)[-5:]
            if recent_scores:
                print(f"   Last 5 scores: {[f'{s:.1f}' for s in recent_scores]}")

        # Training metrics
        training_stats = self.get_recent_training_stats()
        if training_stats['has_training_data']:
            print(f"\n🔧 Training Metrics (recent averages):")
            if training_stats['avg_loss'] is not None:
                print(f"   Loss: {training_stats['avg_loss']:.6f}")
            if training_stats['avg_q_value'] is not None:
                print(f"   Q-values: {training_stats['avg_q_value']:.4f}")
            if training_stats['avg_td_error'] is not None:
                print(f"   TD Error: {training_stats['avg_td_error']:.4f}")
            if training_stats['avg_gradient_norm'] is not None:
                print(f"   Gradient Norm: {training_stats['avg_gradient_norm']:.4f}")
            if training_stats['exploration_rate'] is not None:
                print(f"   Exploration Rate: {training_stats['exploration_rate']:.3f}")

        # Action distribution
        action_dist = self.get_action_distribution()
        if action_dist:
            print(f"\n🎯 Action Distribution:")
            for action in sorted(action_dist.keys()):
                count = action_dist[action]['count']
                pct = action_dist[action]['percentage']
                print(f"   Action {action}: {count:,} ({pct:.1f}%)")

        # Performance metrics
        if self.total_frames > 0:
            elapsed = time.time() - self.start_time
            fps = self.total_frames / elapsed if elapsed > 0 else 0

            # Recent FPS
            current_time = time.time()
            time_since_last = current_time - self.last_log_time
            frames_since_last = self.total_frames - self.last_log_frame
            recent_fps = frames_since_last / time_since_last if time_since_last > 0 else 0

            print(f"\n⚡ Performance:")
            print(f"   Total frames: {self.total_frames:,}")
            print(f"   Overall FPS: {fps:.1f}")
            print(f"   Recent FPS: {recent_fps:.1f}")
            print(f"   Training time: {elapsed / 3600:.2f} hours")

            # Update for next calculation
            self.last_log_time = current_time
            self.last_log_frame = self.total_frames