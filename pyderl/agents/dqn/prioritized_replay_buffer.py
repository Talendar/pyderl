"""
Based on: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""

import random
from typing import List, Tuple

import numpy as np

from pyderl.utils.data_structures import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer:
    """ Prioritized replay buffer.

    Args:
        size (int): Max number of transitions to store in the buffer. When the
            buffer overflows the old memories are dropped.
        alpha (float): How much prioritization is used (0 for no
            prioritization and 1 for full prioritization).
    """

    def __init__(self, size: int, alpha: float) -> None:
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def __len__(self):
        return len(self._storage)

    def _add_to_storage(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        return (np.array(obses_t),
                np.array(actions),
                np.array(rewards),
                np.array(obses_tp1),
                np.array(dones))

    def add(self, obs_t, action, reward, obs_tp1, done):
        idx = self._next_idx
        self._add_to_storage(obs_t, action, reward, obs_tp1, done)

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size

        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)

        return res

    def sample(self, batch_size: int, beta: float) -> Tuple[np.ndarray, ...]:
        """ Sample a batch of experiences.

        Compared to uniform sampling, this method also returns the importance
        weights and idxes of sampled experiences.

        Args:
            batch_size (int): How many transitions to sample.
            beta (float): To what degree to use importance weights (0 means no
                corrections and 1 means full correction).

        Returns:
            Tuple of numpy arrays. More specifically:

                * obs_batch: Batch of observations.
                * act_batch: Batch of actions executed given obs_batch.
                * rew_batch: Rewards received as results of executing act_batch.
                * next_obs_batch: Next set of observations seen after executing
                  act_batch.
                * done_mask: done_mask[i] = 1 if executing act_batch[i] resulted
                  in the end of an episode and 0 otherwise.
                * weights: Array of shape (batch_size,) and dtype np.float32
                  denoting importance weight of each sampled transition.
                * idxes: Array of shape (batch_size,) and dtype np.int32 indices
                  in buffer of sampled experiences.

        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)

        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self,
                          idxes: List[int],
                          priorities: List[float]) -> None:
        """ Update priorities of the sampled transitions.

        Sets the priority of a transition at index idxes[i] in the buffer to
        priorities[i].

        Args:
            idxes (List[int]): List with the indices of the sampled transitions.
            priorities (List[float]): List with the updated priorities
                corresponding to the transitions at the sampled indices, denoted
                by variable `idxes`.
        """
        assert len(idxes) == len(priorities)

        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)

            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
