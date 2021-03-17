"""
TODO
"""

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """ Stores experiences from which a DQN agent can learn from.
    """

    def __init__(self, input_shape: Tuple[int, ...], size: int):
        self._size = size
        self._obs = np.zeros((size, *input_shape), dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.int32)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._next_obs = np.zeros((size, *input_shape), dtype=np.float32)
        self._dones = np.zeros(size, dtype=bool)
        self._ptr = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        """ Adds a new experience to the buffer. """
        idx = self._ptr % self._size
        self._ptr += 1

        self._obs[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_obs[idx] = next_obs
        self._dones[idx] = done

    def make_batch(self, batch_size):
        """ Samples a batch of experiences from the buffer. """
        idx = np.random.choice(len(self), size=batch_size, replace=False)
        return (self._obs[idx],
                self._actions[idx],
                self._rewards[idx],
                self._next_obs[idx],
                self._dones[idx])

    def __len__(self):
        return min(self._ptr, self._size)
