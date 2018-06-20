"""
Various replay buffer implementations.
"""

from abc import ABC, abstractmethod, abstractproperty
from math import sqrt
import random

import numpy as np


class ReplayBuffer(ABC):
    """
    A generic experience replay buffer.

    Replay buffers are useful for off-policy algorithms
    like DQN.
    """
    @abstractproperty
    def size(self):
        """
        Get the current number of transitions stored in
        the buffer.
        """
        pass

    @abstractmethod
    def sample(self, num_samples):
        """
        Sample a batch of experience from the buffer.

        The samples may be drawn in any manner, with or
        without replacement.

        Args:
          num_samples: the number of steps to sample.

        Returns:
          A sequence of num_samples transition dicts.

        Each transition dict is a copy of a dict passed to
        add_sample(), but with extra keys:
          'weight': an importance-sampling weight for the
            sample, possibly relative to the rest of the
            samples in the batch.
          'id': (optional) a way to identify the sample
            for update_weights(). This is specific to the
            buffer implementation. This only remains valid
            until the buffer is modified in some way.
        """
        pass

    @abstractmethod
    def add_sample(self, sample, init_weight=None):
        """
        Add a sampled transition to the buffer.

        Args:
          sample: a dict describing a state transition.
          init_weight: an initial sampling weight for
            the transition. This is related to the weights
            passed to update_weights().
        """
        pass

    def update_weights(self, samples, new_weights):
        """
        Provide the replay buffer with weights for some
        previously-sampled transitions.

        Args:
          samples: a sequence of transitions returned by a
            previous call to sample(). The buffer must not
            have been modified since the transitions were
            sampled.
          new_weights: a sequence of weights, one per
            sample, indicating something like the loss of
            each sample. The exact meaning is specific to
            the replay buffer implementation.

        Some buffer implementations may choose to
        completely ignore this method.
        """
        pass


class UniformReplayBuffer(ReplayBuffer):
    """
    The simplest possible replay buffer.

    Samples are drawn uniformly, and the buffer is kept to
    a certain size by pruning the oldest samples.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.transitions = []

    @property
    def size(self):
        return len(self.transitions)

    def sample(self, num_samples):
        res = [random.choice(self.transitions).copy() for _ in range(num_samples)]
        for transition in res:
            transition['weight'] = 1
        return res

    def add_sample(self, sample, init_weight=None):
        self.transitions.append(sample)
        while len(self.transitions) > self.capacity:
            del self.transitions[0]


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    A prioritized replay buffer with loss-proportional
    sampling.

    Weights passed to add_sample() and update_weights()
    are assumed to be error terms (e.g. the absolute TD
    error).
    """

    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0):
        """
        Create a prioritized replay buffer.

        The beta parameter can be any object that has
        support for the float() built-in.
        This way, you can use a TFScheduleValue.

        Args:
          capacity: the maximum number of transitions to
            store in the buffer.
          alpha: an exponent controlling the temperature.
            Higher values result in more prioritization.
            A value of 0 yields uniform prioritization.
          beta: an exponent controlling the amount of
            importance sampling. A value of 1 yields
            unbiased sampling. A value of 0 yields no
            importance sampling.
          first_max: the initial weight for new samples
            when no init_weight is specified and the
            buffer is completely empty.
          epsilon: a value which is added to every error
            term before the error term is used.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.transitions = []
        self.errors = FloatBuffer(capacity)
        self._max_weight_arg = first_max

    @property
    def size(self):
        return len(self.transitions)

    def sample(self, num_samples):
        indices, probs = self.errors.sample(num_samples)
        beta = float(self.beta)
        importance_weights = np.power(probs * self.size, -beta)
        importance_weights /= np.power(self.errors.min() / self.errors.sum() * self.size, -beta)
        samples = []
        for i, weight in zip(indices, importance_weights):
            sample = self.transitions[i].copy()
            sample['weight'] = weight
            sample['id'] = i
            samples.append(sample)
        return samples

    def add_sample(self, sample, init_weight=None):
        """
        Add a sample to the buffer.

        When new samples are added without an explicit
        initial weight, the maximum weight argument ever
        seen is used. When the buffer is empty, first_max
        is used.
        """
        self.transitions.append(sample)
        if init_weight is None:
            self.errors.append(self._process_weight(self._max_weight_arg))
        else:
            self.errors.append(self._process_weight(init_weight))
        while len(self.transitions) > self.capacity:
            del self.transitions[0]

    def update_weights(self, samples, new_weights):
        for sample, weight in zip(samples, new_weights):
            self.errors.set_value(sample['id'], self._process_weight(weight))

    def _process_weight(self, weight):
        self._max_weight_arg = max(self._max_weight_arg, weight)
        return (weight + self.epsilon) ** self.alpha


class FloatBuffer:
    """A ring-buffer of floating point values."""

    def __init__(self, capacity, dtype='float64'):
        self._capacity = capacity
        self._start = 0
        self._used = 0
        self._buffer = np.zeros((capacity,), dtype=dtype)
        self._bin_size = int(sqrt(capacity))
        num_bins = capacity // self._bin_size
        if num_bins * self._bin_size < capacity:
            num_bins += 1
        self._bin_sums = np.zeros((num_bins,), dtype=dtype)
        self._min = 0

    def append(self, value):
        """
        Add a value to the end of the buffer.

        If the buffer is full, the first value is removed.
        """
        idx = (self._start + self._used) % self._capacity
        if self._used < self._capacity:
            self._used += 1
        else:
            self._start = (self._start + 1) % self._capacity
        self._set_idx(idx, value)

    def sample(self, num_values):
        """
        Sample indices in proportion to their value.

        Returns:
          A tuple (indices, probs)
        """
        assert self._used >= num_values
        res = []
        probs = []
        bin_probs = self._bin_sums / np.sum(self._bin_sums)
        while len(res) < num_values:
            bin_idx = np.random.choice(len(self._bin_sums), p=bin_probs)
            bin_values = self._bin(bin_idx)
            sub_probs = bin_values / np.sum(bin_values)
            sub_idx = np.random.choice(len(bin_values), p=sub_probs)
            idx = bin_idx * self._bin_size + sub_idx
            res.append(idx)
            probs.append(bin_probs[bin_idx] * sub_probs[sub_idx])
        return (np.array(list(res)) - self._start) % self._capacity, np.array(probs)

    def set_value(self, idx, value):
        """Set the value at the given index."""
        idx = (idx + self._start) % self._capacity
        self._set_idx(idx, value)

    def min(self):
        """Get the minimum value in the buffer."""
        return self._min

    def sum(self):
        """Get the sum of the values in the buffer."""
        return np.sum(self._bin_sums)

    def _set_idx(self, idx, value):
        assert not np.isnan(value)
        assert value > 0
        needs_recompute = False
        if self._min == self._buffer[idx]:
            needs_recompute = True
        elif value < self._min:
            self._min = value
        bin_idx = idx // self._bin_size
        self._buffer[idx] = value
        self._bin_sums[bin_idx] = np.sum(self._bin(bin_idx))
        if needs_recompute:
            self._recompute_min()

    def _bin(self, bin_idx):
        if bin_idx == len(self._bin_sums) - 1:
            return self._buffer[self._bin_size * bin_idx:]
        return self._buffer[self._bin_size * bin_idx:self._bin_size * (bin_idx + 1)]

    def _recompute_min(self):
        if self._used < self._capacity:
            self._min = np.min(self._buffer[:self._used])
        else:
            self._min = np.min(self._buffer)
