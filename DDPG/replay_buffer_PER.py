####################################################################################################################################################################################
# Implemented based on:
#  T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” in International Conference on Learning Representations (ICLR), 2015.
####################################################################################################################################################################################

from collections import deque
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha  # Priority exponent (0=uniform, 1=full prioritization)
        self.buffer = deque()
        self.priorities = deque()
        self.cur_size = 0

    def __len__(self):
        return self.cur_size

    def add(self, experience):
        """Add experience with maximum priority so it is sampled soon."""
        max_priority = max(self.priorities) if self.buffer else 1.0
        if self.cur_size < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
            self.cur_size += 1
        else:
            self.buffer.popleft()
            self.priorities.popleft()
            self.buffer.append(experience)
            self.priorities.append(max_priority)

    def get_batch(self, batch_size):
        """Sample a batch of experiences based on priority."""
        if self.cur_size == 0:
            return [], []

        priorities = np.array(self.priorities)
        scaled_priorities = priorities ** self.alpha
        sample_probabilities = scaled_priorities / np.sum(scaled_priorities)

        indices = np.random.choice(self.cur_size, batch_size, p=sample_probabilities)
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices

    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Small constant to avoid zero priority

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.cur_size = 0