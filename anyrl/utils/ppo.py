"""
Helpers for vanilla PPO training runs.
"""

import itertools

from anyrl.algos import GAE
from anyrl.algos.mpi import MPIOptimizer, mpi_ppo
from anyrl.rollouts import TruncatedRoller
import tensorflow as tf

from .tf_state import load_vars, save_vars


def ppo_cli_args(parser):
    """
    Add standard PPO arguments to an ArgumentParser.

    This is useful for scripts that run training.
    """
    parser.add_argument('--ppo-lr', help='Adam learning rate', default=0.0003, type=float)
    parser.add_argument('--ppo-horizon', help='PPO horizon', default=128, type=int)
    parser.add_argument('--ppo-iters', help='PPO iterations', default=16, type=int)
    parser.add_argument('--ppo-epsilon', help='PPO clipping fraction', default=0.1, type=float)
    parser.add_argument('--ppo-discount', help='PPO discount factor', default=0.99, type=float)
    parser.add_argument('--ppo-lambda', help='PPO lambda', default=0.95, type=float)
    parser.add_argument('--ppo-entropy', help='PPO entropy bonus', default=0.01, type=float)
    parser.add_argument('--ppo-reward-scale', help='PPO reward scale', default=1.0, type=float)
    parser.add_argument('--save-path', help='path to PPO save file', default='ppo_agent.pkl')
    parser.add_argument('--save-interval', help='iterations per save', default=5, type=int)


def ppo_kwargs(args):
    """
    Convert parsed CLI arguments into parameters for a new
    anyrl PPO instance.

    Returns:
      A dict meant to be passed as kwargs to the PPO
        constructor.
    """
    return {
        'entropy_reg': args.ppo_entropy,
        'epsilon': args.ppo_epsilon,
        'adv_est': GAE(discount=args.ppo_discount, lam=args.ppo_lambda),
    }


def ppo_loop_kwargs(args):
    """
    Convert parsed CLI arguments into parameters for a
    call to mpi_ppo_loop().

    Returns:
      A dict meant to be passed as kwargs to mpi_ppo_loop.
    """
    return {
        'horizon': args.ppo_horizon,
        'lr': args.ppo_lr,
        'num_iters': args.ppo_iters,
        'reward_scale': args.ppo_reward_scale,
        'save_path': args.save_path,
        'save_interval': args.save_interval,
    }


def mpi_ppo_loop(ppo,
                 env,
                 horizon=128,
                 lr=0.0003,
                 num_iters=16,
                 num_batches=4,
                 reward_scale=1.0,
                 save_path=None,
                 save_interval=5,
                 load_fn=None,
                 rollout_fn=None):
    """
    Run PPO forever on an environment.

    Args:
      ppo: an anyrl PPO instance.
      env: a batched environment.
      horizon: the number of timesteps per segment.
      lr: the Adam learning rate.
      num_iters: the number of training iterations.
      num_batches: the number of mini-batches per training
        epoch.
      reward_scale: a scale to bring rewards into a
        reasonable range.
      save_path: the variable state file.
      save_interval: outer loop iterations per save.
      load_fn: a function to call to load any extra TF
        variables before syncing and training.
      rollout_fn: a function that is called with every
        batch of rollouts before the rollouts are used.
    """
    from .mpi import is_mpi_root, mpi_log

    sess = ppo.model.session

    roller = TruncatedRoller(env, ppo.model, horizon)
    optimizer = MPIOptimizer(tf.train.AdamOptimizer(learning_rate=lr), -ppo.objective,
                             var_list=ppo.variables)

    mpi_log('Initializing optimizer variables...')
    sess.run([v.initializer for v in optimizer.optimizer_vars])

    if save_path and is_mpi_root():
        load_vars(sess, save_path, var_list=ppo.variables)

    if load_fn is not None:
        load_fn()

    mpi_log('Syncing parameters...')
    optimizer.sync_from_root(sess)

    mpi_log('Training...')
    for i in itertools.count():
        mpi_ppo_round(ppo,
                      optimizer,
                      roller,
                      num_iters=num_iters,
                      num_batches=num_batches,
                      reward_scale=reward_scale,
                      rollout_fn=rollout_fn)

        if save_path and i % save_interval == 0 and is_mpi_root():
            save_vars(sess, save_path, var_list=ppo.variables)

        mpi_log('done iteration %d' % i)


def mpi_ppo_round(ppo,
                  optimizer,
                  roller,
                  num_iters=16,
                  num_batches=4,
                  reward_scale=1.0,
                  rollout_fn=None):
    """
    Run a round of PPO.

    Args:
      ppo: an anyrl PPO instance.
      optimizer: an anyrl MPIOptimizer.
      roller: an anyrl Roller.
      num_iters: the number of training iterations.
      num_batches: the number of mini-batches per training
        epoch.
      reward_scale: a scale to bring rewards into a
        reasonable range.
      rollout_fn: a function that is called with every
        batch of rollouts before the rollouts are used.
    """
    from .mpi import mpi_log

    rollouts = roller.rollouts()
    for rollout in rollouts:
        if not rollout.trunc_end:
            msg = 'reward=%f steps=%d' % (rollout.total_reward, rollout.total_steps)
            if 'env_name' in rollout.infos[0]:
                msg += ' env_name=%s' % rollout.infos[0]['env_name']
            print(msg)
    for rollout in rollouts:
        rollout.rewards = [r * reward_scale for r in rollout.rewards]

    if rollout_fn is not None:
        rollout_fn(rollouts)

    total_steps = sum(r.num_steps for r in rollouts)
    results = mpi_ppo(ppo, optimizer, rollouts,
                      batch_size=total_steps // num_batches,
                      num_iter=num_iters)
    mpi_log('explained=%f final_explained=%f clipped=%f entropy=%f' %
            (results['explained_var'][0], results['explained_var'][-1],
             results['clipped'][-1], results['entropy'][0]))
