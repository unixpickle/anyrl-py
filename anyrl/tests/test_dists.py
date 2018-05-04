"""
Tests for probability distributions.
"""

# pylint: disable=E1129

import random

import gym
import numpy as np
import tensorflow as tf

from anyrl.spaces import (CategoricalSoftmax, NaturalSoftmax, BoxGaussian, BoxBeta, MultiBernoulli,
                          TupleDistribution, gym_space_distribution)

# Number of times to run sample-based tests.
NUM_SAMPLE_TRIES = 3


def test_cat_softmax_log_prob(dist_maker=CategoricalSoftmax):
    """
    Test log probabilities on a known case for softmax.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            params = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                                 dtype=tf.float32)
            samples = tf.constant(np.array([[0, 0, 1],
                                            [1, 0, 0],
                                            [0, 1, 0]]),
                                  dtype=tf.float32)
            dist = dist_maker(3)
            log_probs = np.array(sess.run(dist.log_prob(params, samples)))
            expected = np.array([0.665241, 0.090031, 0.244728])
            diff = np.amax(expected - np.exp(log_probs))
            assert diff < 1e-4


def test_cat_softmax_generic():
    """
    Run generic tests for CategoricalSoftmax.
    """
    dist = CategoricalSoftmax(7, low=2)
    tester = DistributionTester(dist)
    tester.test_all()


def test_nat_softmax_log_prob():
    """
    Test log probabilities of NaturalSoftmax.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dist = NaturalSoftmax(7)
            params = tf.constant(np.random.normal(size=(15, 7)), dtype=tf.float64)
            sampled = tf.one_hot([random.randrange(7) for _ in range(15)], 7,
                                 dtype=tf.float64)
            actual = sess.run(dist.log_prob(params, sampled))
            expected = sess.run(CategoricalSoftmax(7).log_prob(params, sampled))
            assert np.allclose(actual, expected)


def test_nat_softmax_determinism():
    """
    Make sure that the gradient of NaturalSoftmax doesn't
    change from run to run.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dist = NaturalSoftmax(7)
            params = tf.constant(np.random.normal(size=(15, 7)), dtype=tf.float64)
            sampled = tf.one_hot([random.randrange(7) for _ in range(15)], 7,
                                 dtype=tf.float64)
            batched_grad = tf.gradients(dist.log_prob(params, sampled), params)[0]
            first = sess.run(batched_grad)
            for _ in range(10):
                assert np.allclose(first, sess.run(batched_grad))


def test_nat_softmax_gradient():
    """
    Test NaturalSoftmax on random natural gradient
    cases.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for size in range(3, 9):
                dist = NaturalSoftmax(size, epsilon=0)
                softmax = CategoricalSoftmax(size)
                param_row = tf.constant(np.random.normal(size=(size,)), dtype=tf.float64)
                params = tf.stack([param_row])
                one_hot = np.zeros((1, size))
                one_hot[0, 1] = 1
                samples = tf.constant(one_hot, dtype=tf.float64)
                kl_div = softmax.kl_divergence(tf.stop_gradient(params), params)
                hessian = sess.run(tf.hessians(kl_div, param_row)[0])
                gradient = sess.run(tf.gradients(softmax.log_prob(params, samples),
                                                 params)[0][0])
                expected = np.matmul(np.array([gradient]), np.linalg.pinv(hessian))[0]
                actual = sess.run(tf.gradients(dist.log_prob(params, samples), params)[0][0])
                assert np.allclose(actual, expected)


def test_nat_softmax_batched():
    """
    Test that batched gradients from NaturalSoftmax
    give the same results as single gradients.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dist = NaturalSoftmax(7)
            params = tf.constant(np.random.normal(size=(15, 7)), dtype=tf.float64)
            sampled = tf.one_hot([random.randrange(7) for _ in range(15)], 7,
                                 dtype=tf.float64)
            batched_grad = tf.gradients(dist.log_prob(params, sampled), params)[0]
            single_grads = []
            for i in range(15):
                sub_params = params[i:i+1]
                prob = dist.log_prob(sub_params, sampled[i:i+1])
                single_grads.append(tf.gradients(prob, sub_params)[0])
            single_grad = tf.concat(single_grads, axis=0)
            batched, single = sess.run((batched_grad, single_grad))
            assert batched.shape == single.shape
            assert np.allclose(batched, single)


def test_box_gaussian_generic():
    """
    Run generic tests for BoxGaussian.
    """
    dist = BoxGaussian(np.array([[-3, 7, 1], [1, 2, 3]]),
                       np.array([[5, 7.1, 3], [2, 3.1, 4]]))
    tester = DistributionTester(dist)
    tester.test_all()


def test_box_beta_generic_simple():
    """
    Run generic tests for BoxBeta with an unscaled
    distribution.
    """
    dist = BoxBeta(np.array([0]), np.array([1]))
    tester = DistributionTester(dist)
    tester.test_all()


def test_box_beta_generic():
    """
    Run generic tests for BoxBeta.
    """
    dist = BoxBeta(np.array([[-3, 7, 1], [1, 2, 3]]),
                   np.array([[5, 7.1, 3], [2, 3.1, 4]]))
    tester = DistributionTester(dist, batch_size=400000)
    tester.test_all()


def test_box_beta_log_prob_simple():
    """
    Test log probs for a very simple BoxBeta.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dist = BoxBeta(np.array([0]), np.array([1]), softplus=False)
            actual = sess.run(dist.log_prob(np.array([[0.5, 0.5]]), np.array([0.5])))
            expected = np.array([_beta_log_prob(0.5, 0.5, 0.5)])
            assert np.allclose(actual, expected, atol=1e-4)


def test_box_beta_log_prob():
    """
    Test log probs for known situations of BoxBeta.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dist = BoxBeta(np.array([[0], [-2]]), np.array([[1], [3]]), softplus=False)
            actual = sess.run(dist.log_prob(np.array([[[0.1, 0.5]], [[0.3, 0.7]]]),
                                            np.array([[0.4], [-0.5]])))
            expected = np.array([_beta_log_prob(0.1, 0.5, 0.4),
                                 _beta_log_prob(0.3, 0.7, 0.3) - np.log(5)])
            assert np.allclose(actual, expected, atol=1e-4)


def _beta_log_prob(alpha, beta, value):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            gamma_a, gamma_b, gamma_ab = sess.run((tf.lgamma(alpha), tf.lgamma(beta),
                                                   tf.lgamma(alpha + beta)))
    log_denom = gamma_a + gamma_b - gamma_ab
    log_num = np.log((value ** (alpha - 1)) * ((1 - value) ** (beta - 1)))
    return log_num - log_denom


def test_multi_bernoulli_generic():
    """
    Run generic tests for MultiBernoulli.
    """
    dist = MultiBernoulli(3)
    tester = DistributionTester(dist)
    tester.test_all()


def test_tuple_unpack_shape():
    """
    Make sure that TupleDistribution.unpack_outs()
    gives the correct shape.
    """
    box_dist = BoxGaussian(np.array([[-3, 7, 1], [1, 2, 3.7]]),
                           np.array([[5, 7.1, 1.5], [2, 2.5, 4]]))
    dist = TupleDistribution((MultiBernoulli(3), box_dist))
    vec = np.array([[0, 1, 0, 1, 2, 3, 4, 5, 6],
                    [1, 0, 1, 6, 5, 4, 3, 2, 1]])
    unpacked = dist.unpack_outs(vec)
    assert len(unpacked) == 2
    assert np.array(unpacked[0]).shape == (2, 3)
    assert np.array(unpacked[1]).shape == (2, 2, 3)


def test_tuple_generic():
    """
    Run generic tests for TupleDistribution.
    """
    box_dist = BoxGaussian(np.array([[-3, 7, 1], [1, 2, 3.7]]),
                           np.array([[5, 7.1, 1.5], [2, 2.5, 4]]))
    dist = TupleDistribution((MultiBernoulli(3), box_dist))
    tester = DistributionTester(dist)
    tester.test_all()


def test_multi_discrete():
    """
    Basic tests for MultiDiscrete.
    """
    space = gym.spaces.MultiDiscrete([2, 2])
    dist = gym_space_distribution(space)
    tester = DistributionTester(dist)
    tester.test_all()
    for sample in dist.sample(np.zeros((50,) + dist.param_shape)):
        assert space.contains(sample)


class DistributionTester:
    """
    Various generic tests for Distributions.
    """

    def __init__(self, dist, batch_size=200000, prec=3e-2):
        self.prec = prec
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
                self.test_mode()

    def test_shapes(self):
        """
        Test the reported shapes.
        """
        for elem in self.dist.out_shape:
            assert isinstance(elem, int)
        for elem in self.dist.param_shape:
            assert isinstance(elem, int)

        samples = self.dist.sample(self._random_params())
        assert len(samples) == self.batch_size
        sample_shape = np.array(self.dist.to_vecs(samples)).shape
        assert sample_shape == self._out_batch_shape()

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
            assert np.array(out).shape == scalar_out_shape

    def test_entropy(self):
        """
        Test entropy with a sample estimate.
        """
        param_placeholder = tf.placeholder(tf.float32, self._param_batch_shape())
        sample_placeholder = tf.placeholder(tf.float32, self._out_batch_shape())
        log_probs = self.dist.log_prob(param_placeholder, sample_placeholder)
        entropy = tf.reduce_mean(self.dist.entropy(param_placeholder))
        for i in range(NUM_SAMPLE_TRIES):
            params = self._random_params(use_numpy=(i % 2 == 0))
            samples = self.dist.sample(params)
            feed_dict = {
                param_placeholder: params,
                sample_placeholder: self.dist.to_vecs(samples)
            }
            log_probs_out = np.array(self.session.run(log_probs,
                                                      feed_dict=feed_dict))
            entropy_out = self.session.run(entropy, feed_dict=feed_dict)

            est_entropy = -np.mean(log_probs_out)
            assert abs(1 - entropy_out/est_entropy) < self.prec

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
            params = self._random_params(use_numpy=(i % 2 == 0))
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
            assert kl_out >= 0

            est_kl = np.mean(log_probs_out_1 - log_probs_out_2)
            assert abs(1 - kl_out/est_kl) < self.prec

    def test_mode(self):
        """
        Test that the mode is truly the most likely.
        """
        param_batch = self._random_params(use_numpy=True)
        samples = tf.constant(self.dist.to_vecs(self.dist.sample(param_batch)))
        modes = self.dist.to_vecs(self.dist.mode(param_batch))
        modes = tf.constant(modes)
        param_batch = tf.constant(param_batch)

        sample_probs = self.session.run(self.dist.log_prob(param_batch, samples))
        mode_probs = self.session.run(self.dist.log_prob(param_batch, modes))
        assert (sample_probs <= mode_probs).all()

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
