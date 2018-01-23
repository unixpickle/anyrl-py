"""
Various replay buffer implementations.
"""

from abc import ABC, abstractmethod, abstractproperty
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

        Args:
          num_samples: the number of steps to sample.
            There must be at least num_samples entries in
            the buffer.

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
        res = [x.copy() for x in random.sample(self.transitions, num_samples)]
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
    # pylint: disable=R0913
    def __init__(self, capacity, alpha, beta, default_init_weight=1e5, epsilon=0):
        """
        Create a prioritized replay buffer.

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
          default_init_weight: the initial weight for new
            samples when add_sample() is called without an
            init_weight argument.
          epsilon: a value which is added to every error
            term before the error term is used.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.default_init_weight = default_init_weight
        self.epsilon = epsilon
        self.transitions = []
        self.errors = FloatBuffer(capacity)

    @property
    def size(self):
        return len(self.transitions)

    def sample(self, num_samples):
        probs = np.power(self.errors.values, self.alpha).astype('float64')
        probs /= np.sum(probs)
        sampled_indices = np.random.choice(len(probs), size=num_samples, replace=False, p=probs)
        importance_weights = np.power(probs[sampled_indices] * self.size, -self.beta)
        importance_weights /= np.amax(importance_weights)
        samples = []
        for i, weight in zip(self.errors.indices(sampled_indices), importance_weights):
            sample = self.transitions[i].copy()
            sample['weight'] = weight
            sample['id'] = i
            samples.append(sample)
        return samples

    def add_sample(self, sample, init_weight=None):
        self.transitions.append(sample)
        if init_weight is None:
            self.errors.append(self.default_init_weight)
        else:
            self.errors.append(init_weight + self.epsilon)
        while len(self.transitions) > self.capacity:
            del self.transitions[0]

    def update_weights(self, samples, new_weights):
        for sample, weight in zip(samples, new_weights):
            self.errors.values[self.errors.unindices(sample['id'])] = weight + self.epsilon

class FloatBuffer:
    """A ring-buffer of floating point values."""
    def __init__(self, capacity, dtype='float64'):
        self._capacity = capacity
        self._start = 0
        self._used = 0
        self._buffer = np.zeros((capacity,), dtype=dtype)

    def append(self, value):
        """
        Add a value to the end of the buffer.

        If the buffer is full, the first value is removed.
        """
        self._buffer[(self._start + self._used) % self._capacity] = value
        if self._used < self._capacity:
            self._used += 1
        else:
            self._start = (self._start + 1) % self._capacity

    @property
    def values(self):
        """
        Get a slice into the values in the buffer.

        The resulting array may be out of order.
        To get the proper ordering, use indices() and
        unindices().
        """
        if self._used < self._capacity:
            return self._buffer[:self._used]
        return self._buffer

    def indices(self, values_indices):
        """
        Convert indices into values() into indices in the
        actual buffer.
        """
        if self._start == 0:
            return values_indices
        return (values_indices - self._start) % self._capacity

    def unindices(self, indices):
        """Perform the inverse of indices()."""
        if self._start == 0:
            return indices
        return (indices + self._start) % self._capacity

    def max(self):
        """Get the maximum value in the buffer."""
        return np.max(self.values)
