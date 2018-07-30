"""
Tests for various environment APIs.
"""

import sys

import numpy as np
import pytest

from anyrl.envs import batched_gym_env, AsyncGymEnv
from anyrl.tests import SimpleEnv

SUB_BATCH_SIZE = 4
NUM_SUB_BATCHES = 3
NUM_STEPS = 100
SHAPE = (3, 8)


@pytest.mark.parametrize('dtype', ['uint8', 'float32'])
def test_dummy_equiv_dtype(dtype):
    """
    Test that batched_gym_env() gives something
    equivalent to a synchronous environment.
    """
    def make_fn(seed):
        """
        Get an environment constructor with a seed.
        """
        return lambda: SimpleEnv(seed, SHAPE, dtype)
    fns = [make_fn(i) for i in range(SUB_BATCH_SIZE * NUM_SUB_BATCHES)]
    real = batched_gym_env(fns, num_sub_batches=NUM_SUB_BATCHES)
    dummy = batched_gym_env(fns, num_sub_batches=NUM_SUB_BATCHES, sync=True)
    try:
        _assert_resets_equal(dummy, real)
        np.random.seed(1337)
        for _ in range(NUM_STEPS):
            joint_shape = (SUB_BATCH_SIZE,) + SHAPE
            actions = np.array(np.random.randint(0, 0x100, size=joint_shape),
                               dtype=dtype)
            _assert_steps_equal(actions, dummy, real)
    finally:
        dummy.close()
        real.close()


def _assert_resets_equal(env1, env2):
    # Non-overlapping resets.
    for i in range(env1.num_sub_batches):
        env1.reset_start(sub_batch=i)
        env2.reset_start(sub_batch=i)
        _assert_reset_waits_equal(env1, env2, i)

    # Overlapping & async resets.
    for i in range(env1.num_sub_batches):
        env1.reset_start(sub_batch=i)
        env2.reset_start(sub_batch=i)
    for i in range(env1.num_sub_batches):
        _assert_reset_waits_equal(env1, env2, i)


def _assert_reset_waits_equal(env1, env2, sub_batch):
    out1 = np.array(env1.reset_wait(sub_batch=sub_batch))
    out2 = np.array(env2.reset_wait(sub_batch=sub_batch))
    assert (out1 == out2).all()


def _assert_steps_equal(actions, env1, env2):
    # Non-overlapping steps.
    for i in range(env1.num_sub_batches):
        env1.step_start(actions, sub_batch=i)
        env2.step_start(actions, sub_batch=i)
        _assert_step_waits_equal(env1, env2, i)

    # Overlapping & async steps.
    for i in range(env1.num_sub_batches):
        env1.step_start(actions, sub_batch=i)
        env2.step_start(actions, sub_batch=i)
    for i in range(env1.num_sub_batches):
        _assert_step_waits_equal(env1, env2, i)


def _assert_step_waits_equal(env1, env2, sub_batch):
    outs1 = env1.step_wait(sub_batch=sub_batch)
    outs2 = env2.step_wait(sub_batch=sub_batch)
    for out1, out2 in zip(outs1[:3], outs2[:3]):
        assert (np.array(out1) == np.array(out2)).all()
    assert outs1[3] == outs2[3]


def test_env_exit():
    """
    Test an environment that straightup exits.
    """
    try:
        AsyncGymEnv(lambda: sys.exit(1), None)
    except RuntimeError:
        return
    pytest.fail('should have gotten exception')


def test_env_exception():
    """
    Test an environment that throws.
    """
    try:
        def raiser():
            raise ValueError('hello world')
        AsyncGymEnv(raiser, None)
    except RuntimeError:
        return
    pytest.fail('should have gotten exception')


def test_async_creation_exit():
    """
    Test that an exception is forwarded when the
    environment constructor exits.
    """
    try:
        batched_gym_env([lambda: sys.exit(1)] * 4)
    except RuntimeError:
        return
    pytest.fail('should have gotten exception')


def test_async_creation_exception():
    """
    Test that an exception is forwarded when the
    environment constructor fails.
    """
    try:
        def raiser():
            raise ValueError('hello world')
        batched_gym_env([raiser] * 4)
    except RuntimeError:
        return
    pytest.fail('should have gotten exception')
