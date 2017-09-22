"""
Tests for various environment APIs.
"""

import unittest

import gym
import numpy as np

from anyrl.rollouts import BatchedEnv, batched_gym_env

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
        equivalent to _DummyBatchedEnv.
        """
        def make_fn(seed):
            """
            Get an environment constructor with a seed.
            """
            return lambda: _SimpleEnv(seed, SHAPE, dtype)
        fns = [make_fn(i) for i in range(SUB_BATCH_SIZE * NUM_SUB_BATCHES)]
        real = batched_gym_env(fns, num_sub_batches=NUM_SUB_BATCHES)
        dummy = _DummyBatchedEnv(fns, NUM_SUB_BATCHES)
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

class _SimpleEnv(gym.Env):
    """
    An environment with a pre-determined observation space
    and RNG seed.
    """
    def __init__(self, seed, shape, dtype):
        np.random.seed(seed)
        self._dtype = dtype
        self._start_obs = np.array(np.random.randint(0, 0x100, size=shape),
                                   dtype=dtype)
        self._max_steps = seed + 1
        self._cur_obs = None
        self._cur_step = 0
        self.action_space = gym.spaces.Box(0, 0xff, shape=shape)
        self.observation_space = self.action_space

    def _step(self, action):
        self._cur_obs += np.array(action, dtype=self._dtype)
        self._cur_step += 1
        done = self._cur_step >= self._max_steps
        reward = self._cur_step / self._max_steps
        return self._cur_obs, reward, done, {'foo': 'bar' + str(reward)}

    def _reset(self):
        self._cur_obs = self._start_obs
        self._cur_step = 0
        return self._cur_obs

class _DummyBatchedEnv(BatchedEnv):
    """
    The simplest possible batched environment.
    """
    def __init__(self, env_fns, num_sub_batches):
        env_fn_batches = []
        batch = len(env_fns) // num_sub_batches
        for i in range(num_sub_batches):
            env_fn_batches.append(env_fns[i*batch : (i+1)*batch])
        self._envs = [[f() for f in fs] for fs in env_fn_batches]
        self._step_actions = [None] * len(self._envs)

    @property
    def num_sub_batches(self):
        return len(self._envs)

    @property
    def num_envs_per_sub_batch(self):
        return len(self._envs[0])

    def reset_start(self, sub_batch=0):
        pass

    def reset_wait(self, sub_batch=0):
        return [env.reset() for env in self._envs[sub_batch]]

    def step_start(self, actions, sub_batch=0):
        self._step_actions[sub_batch] = actions

    def step_wait(self, sub_batch=0):
        obses, rews, dones, infos = ([], [], [], [])
        for env, action in zip(self._envs[sub_batch], self._step_actions[sub_batch]):
            obs, rew, done, info = env.step(action)
            if done:
                obs = env.reset()
            obses.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        return obses, rews, dones, infos

    def close(self):
        for batch in self._envs:
            for env in batch:
                env.close()

if __name__ == '__main__':
    unittest.main()
