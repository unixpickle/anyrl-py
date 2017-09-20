"""
Abstractions for probability distributions.
"""

from abc import ABC, abstractmethod, abstractproperty

class Distribution(ABC):
    """
    A parametric probability distribution.

    All methods operate on and produce TensorFlow tensors
    except for sample(), which takes an array and produces
    a batch of space-specific objects.
    """
    @abstractproperty
    def param_size(self):
        """
        Get the size of the parameter vectors for this
        distribution.
        """
        pass

    @abstractmethod
    def sample(self, param_batch):
        """
        Create a list of samples from the distribution
        given the batch of parameter vectors.
        """
        pass

    @abstractmethod
    def log_probs(self, param_batch, samples):
        """
        Compute the log probability (or log density) of
        the samples, given the parameters.
        """
        pass

    @abstractmethod
    def entropy(self, param_batch):
        """
        Compute the entropy (or differential entropy) for
        each set of parameters.
        """
        pass

    @abstractmethod
    def kl_divergence(self, param_batch_1, param_batch_2):
        """
        Compute KL(params_1 || params_2) for each pair of
        parameters in the batch.
        """
        pass
