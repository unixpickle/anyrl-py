"""
Atomic file writing.
"""

import os
import pickle


def atomic_pickle(obj, path, **kwargs):
    """
    Pickle an object and write it atomically to a file.

    Args:
      obj: the object to pickle.
      path: the destination filepath.
      kwargs: arguments for pickle.dump().
    """
    tmp_path = os.path.join(path + '.anyrl.swp')
    try:
        with open(tmp_path, 'wb+') as out_file:
            pickle.dump(obj, out_file, **kwargs)
        os.rename(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
