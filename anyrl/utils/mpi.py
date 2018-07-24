"""
Helpers for MPI.
"""

# pylint: disable=E1101

import os
import platform
import subprocess

from mpi4py import MPI


def setup_mpi_gpus(env_var=None):
    """
    Set CUDA_VISIBLE_DEVICES using MPI.

    Args:
      env_var: if specified, use this as an environment
        variable that specifies the number of GPUs.
    """
    num_gpus = count_gpus(env_var=env_var)
    node_id = platform.node()
    nodes = MPI.COMM_WORLD.allgather(node_id)
    local_rank = len([n for n in nodes[:MPI.COMM_WORLD.Get_rank()] if n == node_id])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % num_gpus)


def is_mpi_root():
    """Check if this worker is the root node."""
    return MPI.COMM_WORLD.Get_rank() == 0


def mpi_mean(scalar):
    """Compute the mean of a scalar across all workers."""
    return MPI.COMM_WORLD.allreduce(scalar) / MPI.COMM_WORLD.Get_size()


def mpi_log(*args, **kwargs):
    """Call print() only for the root node."""
    if is_mpi_root():
        print(*args, **kwargs)


def count_gpus(env_var=None):
    """
    Count the number of GPUs on the system.

    Args:
      env_var: if specified, this is an environment
        variable that is treated as an integer.

    Returns:
      An integer number of GPUs.

    Raises:
      An exception if the environment variable points to a
        non-integer value, or if nvidia-smi is not present
        and an environment variable was not specified.
    """
    if env_var is not None and env_var in os.environ:
        return int(os.environ[env_var])
    output = subprocess.check_output(['nvidia-smi', '-q'])
    line = next(x for x in str(output, 'utf-8').split('\n') if x.startswith('Attached GPUs'))
    return int(line.split(' ')[-1])
