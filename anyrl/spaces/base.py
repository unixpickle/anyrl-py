"""
Abstractions for probability distributions.
"""

from abc import ABC, abstractmethod, abstractproperty


class Vectorizer(ABC):
    """
    A way to convert action or observation space elements
    to tensors.
    """
    @abstractproperty
    def out_shape(self):
        """
        The shape of vectorized space elements.
        """
        pass

    @abstractmethod
    def to_vecs(self, space_elements):
        """
        Convert a list-like object of space elements to a
        list-like object of tensors.
        """
        pass


class Distribution(Vectorizer):
    """
    A parametric probability distribution.

    All methods operate on and produce TensorFlow tensors
    except for sample().
    """
    @abstractproperty
    def param_shape(self):
        """
        Get the shape of the distribution parameters.
        """
        pass

    @abstractmethod
    def sample(self, param_batch):
        """
        Create a list of samples from the distribution
        given the batch of parameter vectors.

        The param_batch should be some kind of array-like
        object.
        The result is a list-like object of space
        elements.
        """
        pass

    @abstractmethod
    def mode(self, param_batch):
        """
        Compute the most likely sample for each parameter
        vector in a batch of parameter vectors.

        The param_batch should be some kind of array-like
        object.
        The result is a list-like object of space
        elements.
        """
        pass

    @abstractmethod
    def log_prob(self, param_batch, sample_vecs):
        """
        Compute the log probability (or log density) of
        the samples, given the parameters.

        You can obtain vectors for sample_vecs via the
        to_vecs() method.
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
