"""
Tests for probability distributions.
"""

# pylint: disable=E1129

import unittest

import gym
import numpy as np
import tensorflow as tf

from anyrl.spaces import (CategoricalSoftmax, BoxGaussian, MultiBernoulli, TupleDistribution,
                          gym_space_distribution)

# Number of times to run sample-based tests.
NUM_SAMPLE_TRIES = 3

class DistributionTester:
    """
    Various generic tests for Distributions.
    """
    def __init__(self, test, dist, batch_size=200000, prec=3e-2):
        self.prec = prec
        self.test = test
        self.dist = dist
        self.batch_size = batch_size
        self.session = None

    def test_all(self):
        """
        Run all generic tests.
        """
        np.random.seed(1337)
        with tf.Graph().as_default():
            self.session = tf.Session()
            with self.session:
                self.test_shapes()
                self.test_entropy()
                self.test_kl()

    def test_shapes(self):
        """
        Test the reported shapes.
        """
        for elem in self.dist.out_shape:
            self.test.assertTrue(isinstance(elem, int))
        for elem in self.dist.param_shape:
            self.test.assertTrue(isinstance(elem, int))

        samples = self.dist.sample(self._random_params())
        self.test.assertEqual(len(samples), self.batch_size)
        sample_shape = np.array(self.dist.to_vecs(samples)).shape
        self.test.assertEqual(sample_shape, self._out_batch_shape())

        param_placeholder = tf.placeholder(tf.float32, self._param_batch_shape())
        sample_placeholder = tf.placeholder(tf.float32, self._out_batch_shape())
        feed_dict = {
            param_placeholder: self._random_params(),
            sample_placeholder: self.dist.to_vecs(samples)
        }
        log_probs = self.dist.log_prob(param_placeholder, sample_placeholder)
        entropies = self.dist.entropy(param_placeholder)
        kls = self.dist.kl_divergence(param_placeholder, param_placeholder)
        outs = self.session.run([log_probs, entropies, kls], feed_dict=feed_dict)
        scalar_out_shape = (self.batch_size,)
        for out in outs:
            self.test.assertEqual(np.array(out).shape, scalar_out_shape)

    def test_entropy(self):
        """
        Test entropy with a sample estimate.
        """
        param_placeholder = tf.placeholder(tf.float32, self._param_batch_shape())
        sample_placeholder = tf.placeholder(tf.float32, self._out_batch_shape())
        log_probs = self.dist.log_prob(param_placeholder, sample_placeholder)
        entropy = tf.reduce_mean(self.dist.entropy(param_placeholder))
        for i in range(NUM_SAMPLE_TRIES):
            params = self._random_params(use_numpy=(i%2 == 0))
            samples = self.dist.sample(params)
            feed_dict = {
                param_placeholder: params,
                sample_placeholder: self.dist.to_vecs(samples)
            }
            log_probs_out = np.array(self.session.run(log_probs,
                                                      feed_dict=feed_dict))
            entropy_out = self.session.run(entropy, feed_dict=feed_dict)

            est_entropy = -np.mean(log_probs_out)
            self.test.assertTrue(abs(1 - entropy_out/est_entropy) < self.prec)

    def test_kl(self):
        """
        Test KL divergence with a sample estimate.
        """
        param_1_placeholder = tf.placeholder(tf.float32, self._param_batch_shape())
        param_2_placeholder = tf.placeholder(tf.float32, self._param_batch_shape())
        sample_placeholder = tf.placeholder(tf.float32, self._out_batch_shape())
        log_probs_1 = self.dist.log_prob(param_1_placeholder, sample_placeholder)
        log_probs_2 = self.dist.log_prob(param_2_placeholder, sample_placeholder)
        real_kl = tf.reduce_mean(self.dist.kl_divergence(param_1_placeholder,
                                                         param_2_placeholder))
        for i in range(NUM_SAMPLE_TRIES):
            params = self._random_params(use_numpy=(i%2 == 0))
            samples = self.dist.sample(params)
            feed_dict = {
                param_1_placeholder: params,
                param_2_placeholder: self._random_params(),
                sample_placeholder: self.dist.to_vecs(samples)
            }
            log_probs_out_1 = np.array(self.session.run(log_probs_1,
                                                        feed_dict=feed_dict))
            log_probs_out_2 = np.array(self.session.run(log_probs_2,
                                                        feed_dict=feed_dict))
            kl_out = self.session.run(real_kl, feed_dict=feed_dict)
            self.test.assertTrue(kl_out >= 0)

            est_kl = np.mean(log_probs_out_1 - log_probs_out_2)
            self.test.assertTrue(abs(1 - kl_out/est_kl) < self.prec)

    def _random_params(self, use_numpy=False):
        params = np.random.normal(size=self.dist.param_shape)
        batch = [params] * self.batch_size
        if use_numpy:
            batch = np.array(batch)
        return batch

    def _param_batch_shape(self):
        return (self.batch_size,) + self.dist.param_shape

    def _out_batch_shape(self):
        return (self.batch_size,) + self.dist.out_shape

class TestCategoricalSoftmax(unittest.TestCase):
    """
    Tests for the CategoricalSoftmax distribution.
    """
    def test_log_prob(self):
        """
        Test log probabilities on a known case.
        """
        with tf.Graph().as_default():
            with tf.Session() as sess:
                params = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                                     dtype=tf.float32)
                samples = tf.constant(np.array([[0, 0, 1],
                                                [1, 0, 0],
                                                [0, 1, 0]]),
                                      dtype=tf.float32)
                dist = CategoricalSoftmax(3)
                log_probs = np.array(sess.run(dist.log_prob(params, samples)))
                expected = np.array([0.665241, 0.090031, 0.244728])
                diff = np.amax(expected - np.exp(log_probs))
                self.assertTrue(diff < 1e-4)

    def test_generic(self):
        """
        Run generic tests with DistributionTester.
        """
        dist = CategoricalSoftmax(7, low=2)
        tester = DistributionTester(self, dist)
        tester.test_all()

class TestBoxGaussian(unittest.TestCase):
    """
    Tests for the BoxGaussian distribution.
    """
    def test_generic(self):
        """
        Run generic tests with DistributionTester.
        """
        dist = BoxGaussian(np.array([[-3, 7, 1], [1, 2, 3]]),
                           np.array([[5, 7.1, 3], [2, 3.1, 4]]))
        tester = DistributionTester(self, dist)
        tester.test_all()

class TestMultiBernoulli(unittest.TestCase):
    """
    Tests for the MultiBernoulli distribution.
    """
    def test_generic(self):
        """
        Run generic tests with DistributionTester.
        """
        dist = MultiBernoulli(3)
        tester = DistributionTester(self, dist)
        tester.test_all()

class TestTuple(unittest.TestCase):
    """
    Tests for tuples of distributions.
    """
    def test_unpack_shape(self):
        """
        Make sure that unpack_outs gives the correct
        shape.
        """
        box_dist = BoxGaussian(np.array([[-3, 7, 1], [1, 2, 3.7]]),
                               np.array([[5, 7.1, 1.5], [2, 2.5, 4]]))
        dist = TupleDistribution((MultiBernoulli(3), box_dist))
        vec = np.array([[0, 1, 0, 1, 2, 3, 4, 5, 6],
                        [1, 0, 1, 6, 5, 4, 3, 2, 1]])
        unpacked = dist.unpack_outs(vec)
        self.assertEqual(len(unpacked), 2)
        self.assertEqual(np.array(unpacked[0]).shape, (2, 3))
        self.assertEqual(np.array(unpacked[1]).shape, (2, 2, 3))

    def test_generic(self):
        """
        Run generic tests with DistributionTester.
        """
        box_dist = BoxGaussian(np.array([[-3, 7, 1], [1, 2, 3.7]]),
                               np.array([[5, 7.1, 1.5], [2, 2.5, 4]]))
        dist = TupleDistribution((MultiBernoulli(3), box_dist))
        tester = DistributionTester(self, dist)
        tester.test_all()

    def test_multi_discrete(self):
        """
        Basic tests for MultiDiscrete, which is internally
        implemented using a tuple.
        """
        space = gym.spaces.MultiDiscrete([[1, 3], [-1, 5]])
        dist = gym_space_distribution(space)
        tester = DistributionTester(self, dist)
        tester.test_all()
        for sample in dist.sample(np.zeros((50,) + dist.param_shape)):
            self.assertTrue(space.contains(sample))

if __name__ == '__main__':
    unittest.main()
