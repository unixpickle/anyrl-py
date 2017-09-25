"""
Tests for various environment APIs.
"""

import unittest

import numpy as np

from anyrl.rollouts import batched_gym_env
from anyrl.tests import SimpleEnv

SUB_BATCH_SIZE = 4
NUM_SUB_BATCHES = 3
NUM_STEPS = 100
SHAPE = (3, 8)

class BatchedGymEnvTest(unittest.TestCase):
    """
    Tests for the environment produced by batched_gym_env.
    """
    def test_dummy_equiv_bytes(self):
        """
        Test equivalence with uint8 observations.

        Intended to test shared memory.
        """
        self._test_dummy_equiv_dtype('uint8')

    def test_dummy_equiv_floats(self):
        """
        Test equivalence with float32 observations.
        """
        self._test_dummy_equiv_dtype('float32')

    def _test_dummy_equiv_dtype(self, dtype):
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
            self._assert_resets_equal(dummy, real)
            np.random.seed(1337)
            for _ in range(NUM_STEPS):
                joint_shape = (SUB_BATCH_SIZE,) + SHAPE
                actions = np.array(np.random.randint(0, 0x100, size=joint_shape),
                                   dtype=dtype)
                self._assert_steps_equal(actions, dummy, real)
        finally:
            dummy.close()
            real.close()

    def _assert_resets_equal(self, env1, env2):
        # Non-overlapping resets.
        for i in range(env1.num_sub_batches):
            env1.reset_start(sub_batch=i)
            env2.reset_start(sub_batch=i)
            self._assert_reset_waits_equal(env1, env2, i)

        # Overlapping & async resets.
        for i in range(env1.num_sub_batches):
            env1.reset_start(sub_batch=i)
            env2.reset_start(sub_batch=i)
        for i in range(env1.num_sub_batches):
            self._assert_reset_waits_equal(env1, env2, i)

    def _assert_reset_waits_equal(self, env1, env2, sub_batch):
        out1 = np.array(env1.reset_wait(sub_batch=sub_batch))
        out2 = np.array(env2.reset_wait(sub_batch=sub_batch))
        self.assertTrue((out1 == out2).all())

    def _assert_steps_equal(self, actions, env1, env2):
        # Non-overlapping steps.
        for i in range(env1.num_sub_batches):
            env1.step_start(actions, sub_batch=i)
            env2.step_start(actions, sub_batch=i)
            self._assert_step_waits_equal(env1, env2, i)

        # Overlapping & async steps.
        for i in range(env1.num_sub_batches):
            env1.step_start(actions, sub_batch=i)
            env2.step_start(actions, sub_batch=i)
        for i in range(env1.num_sub_batches):
            self._assert_step_waits_equal(env1, env2, i)

    def _assert_step_waits_equal(self, env1, env2, sub_batch):
        outs1 = env1.step_wait(sub_batch=sub_batch)
        outs2 = env2.step_wait(sub_batch=sub_batch)
        for out1, out2 in zip(outs1[:3], outs2[:3]):
            self.assertTrue((np.array(out1) == np.array(out2)).all())
        self.assertEqual(outs1[3], outs2[3])

if __name__ == '__main__':
    unittest.main()
