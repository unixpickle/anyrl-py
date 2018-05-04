"""
Tests for all space implementations.
"""

from gym.spaces import Box
import numpy as np

from anyrl.spaces import StackedBoxSpace


def test_stacked_box_space_json():
    """
    Test JSON conversions for StackedBoxSpace.
    """
    box_space = Box(low=np.array([[1., 2.], [3., 4.]]),
                    high=np.array([[1.3, 4.9], [3.5, 5.]]))
    space = StackedBoxSpace(box_space, 2)
    samples = [box_space.sample() for _ in range(5)]
    jsoned = space.to_jsonable(samples)
    assert space.to_jsonable(space.from_jsonable(jsoned)) == jsoned
    in_data = [[np.array([[1.1, 2.1], [3.1, 4.1]]), np.array([[1.2, 2.2], [3.2, 4.2]])],
               [np.array([[1.11, 2.11], [3.3, 4.]]), np.array([[1.21, 2.2], [3.23, 4.21]])]]
    assert (space.to_jsonable(in_data) ==
            [[[[1.1, 2.1], [3.1, 4.1]], [[1.11, 2.11], [3.3, 4.]]],
             [[[1.2, 2.2], [3.2, 4.2]], [[1.21, 2.2], [3.23, 4.21]]]])
    inverted = space.from_jsonable(space.to_jsonable(in_data))
    for idx1, idx2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert np.allclose(in_data[idx1][idx2], inverted[idx1][idx2])
