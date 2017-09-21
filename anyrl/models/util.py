"""
Helpers for implementing models.
"""

import numpy as np

def mini_batches(size_per_index, batch_size=None):
    """
    Generate mini-batches of size batch_size.

    The size_per_index list is the size of each batch
    element.
    Batches are generated such that the sum of the sizes
    of the batch elements is at least batch_size.
    """
    if batch_size is None or sum(size_per_index) <= batch_size:
        while True:
            yield list(range(len(size_per_index)))
    cur_indices = []
    cur_size = 0
    for idx in _infinite_random_shuffle(len(size_per_index)):
        cur_indices.append(idx)
        cur_size += size_per_index[idx]
        if cur_size >= batch_size:
            yield cur_indices
            cur_indices = []
            cur_size = 0

def _infinite_random_shuffle(num_elements):
    """
    Continually permute the elements and yield all of the
    permuted indices.
    """
    while True:
        for elem in np.random.permutation(num_elements):
            yield elem

def product(vals):
    """
    Compute the product of values in a list-like object.
    """
    prod = 1
    for val in vals:
        prod *= val
    return prod
