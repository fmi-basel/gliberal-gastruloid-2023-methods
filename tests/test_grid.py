import pytest
import numpy as np

from goid.grid import split_path


def test_split_path():
    '''Check that splitting a circular path correctly connects the idx 0 and -1'''

    fake_path = np.arange(101)
    fake_path[-1] = 0  # close the loop
    subpaths = split_path(fake_path, [10, 20])

    np.testing.assert_array_equal(subpaths[0], np.arange(10, 20))
    np.testing.assert_array_equal(
        subpaths[1], np.concatenate([np.arange(20, 100),
                                     np.arange(0, 10)]))


# TODO test grid extraction + segments label generation
# save visually validated example --> test that result can be reproduced
