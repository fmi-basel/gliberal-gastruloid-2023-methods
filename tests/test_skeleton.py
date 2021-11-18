import pytest
import numpy as np

from goid.skeleton import find_terminus, find_longest_skeleton_path


def test_find_terminus():
    '''test finding skeleton terminus'''

    # single line skel
    skel = np.zeros((100, 100), dtype=bool)
    skel[47, 20:80] = True
    termini = [tuple(p) for p in find_terminus(skel)]

    assert (47, 20) in termini
    assert (47, 79) in termini

    # add a corner (L-shaped skel)
    skel[47:90, 80] = True
    termini = [tuple(p) for p in find_terminus(skel)]
    assert (47, 20) in termini
    assert (89, 80) in termini


def test_find_longest_skeleton_path():
    '''check that returned path is the longest possible for any pairs of termini'''

    # draw an assymetric cross
    skel = np.zeros((100, 100), dtype=bool)
    skel[50, 40:90] = True
    skel[30:60, 50] = True

    assert len(find_longest_skeleton_path(skel)) == 39 + 20

    # make another branch longer
    skel[30:80, 50] = True
    assert len(find_longest_skeleton_path(skel)) == 39 + 29
