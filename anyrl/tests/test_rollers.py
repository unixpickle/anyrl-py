"""
Test various Roller implementations.
"""

from anyrl.envs import batched_gym_env
from anyrl.rollouts import BasicRoller, TruncatedRoller, EpisodeRoller, Rollout
from anyrl.tests import SimpleEnv, SimpleModel
import numpy as np
import pytest


@pytest.mark.parametrize('stateful,state_tuple', [(False, False), (True, False), (True, True)])
def test_trunc_basic_equivalence(stateful, state_tuple):
    """
    Test that TruncatedRoller is equivalent to BasicRoller
    for batches of one environment when the episodes end
    cleanly.
    """
    def env_fn():
        return SimpleEnv(3, (4, 5), 'uint8')

    env = env_fn()
    model = SimpleModel(env.action_space.low.shape,
                        stateful=stateful,
                        state_tuple=state_tuple)
    basic_roller = BasicRoller(env, model, min_episodes=5)
    expected = basic_roller.rollouts()
    total_timesteps = sum([x.num_steps for x in expected])

    batched_env = batched_gym_env([env_fn], sync=True)
    trunc_roller = TruncatedRoller(batched_env, model, total_timesteps)
    actual = trunc_roller.rollouts()
    _compare_rollout_batch(actual, expected)


@pytest.mark.parametrize('stateful,state_tuple', [(False, False), (True, False), (True, True)])
def test_truncation(stateful, state_tuple):
    """
    Test sequence truncation for TruncatedRoller with a
    batch of one environment.
    """
    def env_fn():
        return SimpleEnv(7, (5, 3), 'uint8')

    env = env_fn()
    model = SimpleModel(env.action_space.low.shape,
                        stateful=stateful,
                        state_tuple=state_tuple)
    basic_roller = BasicRoller(env, model, min_episodes=5)
    expected = basic_roller.rollouts()
    total_timesteps = sum([x.num_steps for x in expected])

    batched_env = batched_gym_env([env_fn], sync=True)
    trunc_roller = TruncatedRoller(batched_env, model, total_timesteps // 2 + 1)
    actual1 = trunc_roller.rollouts()
    assert actual1[-1].trunc_end
    actual2 = trunc_roller.rollouts()
    expected1, expected2 = _artificial_truncation(expected,
                                                  len(actual1) - 1,
                                                  actual1[-1].num_steps)
    assert len(actual2) == len(expected2) + 1
    actual2 = actual2[:-1]
    _compare_rollout_batch(actual1, expected1)
    _compare_rollout_batch(actual2, expected2)


@pytest.mark.parametrize('stateful,state_tuple', [(False, False), (True, False), (True, True)])
def test_trunc_batches(stateful, state_tuple):
    """
    Test that TruncatedRoller produces the same result for
    batches as it does for individual environments.
    """
    env_fns = [lambda seed=x: SimpleEnv(seed, (5, 3), 'uint8') for x in range(15)]
    model = SimpleModel((5, 3),
                        stateful=stateful,
                        state_tuple=state_tuple)

    unbatched_rollouts = []
    for env_fn in env_fns:
        batched_env = batched_gym_env([env_fn], sync=True)
        trunc_roller = TruncatedRoller(batched_env, model, 17)
        for _ in range(3):
            unbatched_rollouts.extend(trunc_roller.rollouts())

    batched_rollouts = []
    batched_env = batched_gym_env(env_fns, num_sub_batches=3, sync=True)
    trunc_roller = TruncatedRoller(batched_env, model, 17)
    for _ in range(3):
        batched_rollouts.extend(trunc_roller.rollouts())

    _compare_rollout_batch(unbatched_rollouts, batched_rollouts, ordered=False)


def test_trunc_drop_states():
    """
    Test TruncatedRoller with drop_states=True.
    """
    env_fns = [lambda seed=x: SimpleEnv(seed, (5, 3), 'uint8') for x in range(15)]
    model = SimpleModel((5, 3), stateful=True, state_tuple=True)

    expected_rollouts = []
    batched_env = batched_gym_env(env_fns, num_sub_batches=3, sync=True)
    trunc_roller = TruncatedRoller(batched_env, model, 17)
    for _ in range(3):
        expected_rollouts.extend(trunc_roller.rollouts())
    for rollout in expected_rollouts:
        for model_out in rollout.model_outs:
            model_out['states'] = None

    actual_rollouts = []
    trunc_roller = TruncatedRoller(batched_env, model, 17, drop_states=True)
    for _ in range(3):
        actual_rollouts.extend(trunc_roller.rollouts())

    _compare_rollout_batch(actual_rollouts, expected_rollouts)


@pytest.mark.parametrize('stateful,state_tuple', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('limits', [{'min_episodes': 5}, {'min_steps': 7}])
def test_ep_basic_equivalence(stateful, state_tuple, limits):
    """
    Test that EpisodeRoller is equivalent to a
    BasicRoller when run on a single environment.
    """
    def env_fn():
        return SimpleEnv(3, (4, 5), 'uint8')

    env = env_fn()
    model = SimpleModel(env.action_space.low.shape,
                        stateful=stateful,
                        state_tuple=state_tuple)
    basic_roller = BasicRoller(env, model, **limits)
    expected = basic_roller.rollouts()

    batched_env = batched_gym_env([env_fn], sync=True)
    ep_roller = EpisodeRoller(batched_env, model, **limits)
    actual = ep_roller.rollouts()
    _compare_rollout_batch(actual, expected)


@pytest.mark.parametrize('stateful,state_tuple', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('limits', [{'min_episodes': 5}, {'min_steps': 7}])
def test_ep_batches(stateful, state_tuple, limits):
    """
    Test that EpisodeRoller is equivalent to a
    BasicRoller when run on a batch of envs.
    """
    def env_fn():
        return SimpleEnv(3, (4, 5), 'uint8')

    model = SimpleModel((4, 5),
                        stateful=stateful,
                        state_tuple=state_tuple)

    batched_env = batched_gym_env([env_fn]*21, num_sub_batches=7, sync=True)
    ep_roller = EpisodeRoller(batched_env, model, **limits)
    actual = ep_roller.rollouts()

    total_steps = sum([r.num_steps for r in actual])
    assert len(actual) >= ep_roller.min_episodes
    assert total_steps >= ep_roller.min_steps

    if 'min_steps' not in limits:
        num_eps = ep_roller.min_episodes + batched_env.num_envs - 1
        assert len(actual) == num_eps

    basic_roller = BasicRoller(env_fn(), model, min_episodes=len(actual))
    expected = basic_roller.rollouts()

    _compare_rollout_batch(actual, expected)


def test_ep_multiple_batches():
    """
    Make sure calling EpisodeRoller.rollouts()
    multiple times works.
    """
    def env_fn():
        return SimpleEnv(3, (4, 5), 'uint8')

    env = env_fn()
    try:
        model = SimpleModel(env.action_space.low.shape)
    finally:
        env.close()
    batched_env = batched_gym_env([env_fn], sync=True)
    try:
        ep_roller = EpisodeRoller(batched_env, model, min_episodes=5, min_steps=7)
        first = ep_roller.rollouts()
        for _ in range(3):
            _compare_rollout_batch(first, ep_roller.rollouts())
    finally:
        batched_env.close()


def _compare_rollout_batch(rs1, rs2, ordered=True):
    """
    Assert that batches of rollouts are the same.
    """
    assert len(rs1) == len(rs2)
    if ordered:
        for rollout1, rollout2 in zip(rs1, rs2):
            assert _rollout_hash(rollout1) == _rollout_hash(rollout2)
    else:
        hashes1 = [_rollout_hash(r) for r in rs1]
        hashes2 = [_rollout_hash(r) for r in rs2]
        for hash1 in hashes1:
            assert hash1 in hashes2
            hashes2.remove(hash1)


def _rollout_hash(rollout):
    """
    Generate a string that uniquely identifies a rollout.
    """
    # Prevent ellipsis.
    np.set_printoptions(threshold=1e6)

    res = ''
    res += str(rollout.trunc_start)
    res += str(rollout.trunc_end)
    res += str(rollout.prev_steps)
    res += '|'
    res += str(rollout.prev_reward)
    res += str(rollout.observations)
    res += str(rollout.rewards)
    res += str(rollout.infos)
    for out in rollout.model_outs:
        res += 'output'
        for key in sorted(out.keys()):
            res += 'kv'
            res += key
            res += str(out[key])
    return res


def _artificial_truncation(rollouts, rollout_idx, timestep_idx):
    """
    Split up the rollouts into two batches by artificially
    truncating (and splitting) the given rollout before
    the given timestep.

    Returns (left_rollouts, right_rollouts)
    """
    to_split = rollouts[rollout_idx]
    left = Rollout(observations=to_split.observations[:timestep_idx+1],
                   model_outs=to_split.model_outs[:timestep_idx+1],
                   rewards=to_split.rewards[:timestep_idx],
                   start_state=to_split.start_state,
                   infos=to_split.infos[:timestep_idx])
    right = Rollout(observations=to_split.observations[timestep_idx:],
                    model_outs=to_split.model_outs[timestep_idx:],
                    rewards=to_split.rewards[timestep_idx:],
                    start_state=to_split.model_outs[timestep_idx-1]['states'],
                    prev_steps=timestep_idx,
                    prev_reward=left.total_reward,
                    infos=to_split.infos[timestep_idx:])
    return rollouts[:rollout_idx]+[left], [right]+rollouts[rollout_idx+1:]
