import pytest
import numpy as np

from goid.props import SkeletonFeatureExtractor, GridFeatureExtractor, BorderFeatureExtractor


def test_skeleton_feature_extractor():

    # draw an asymmetric cross
    skel = np.zeros((100, 100), dtype=bool)
    skel[50, 40:90] = True
    skel[30:60, 50] = True

    extractor = SkeletonFeatureExtractor()

    props = extractor({'skel': skel}, None)

    assert props.loc[0, 'feature_value'] == 4


def test_grid_feature_extractor():

    extractor = GridFeatureExtractor()

    grid = np.stack(np.meshgrid(np.arange(101) * 0.1,
                                np.arange(31) * 2,
                                indexing='ij'),
                    axis=-1)
    # 10 x 60

    props = extractor({'grid': grid}, None).set_index('feature_name')

    np.testing.assert_almost_equal(props.loc['width_mean', 'feature_value'],
                                   10)
    np.testing.assert_almost_equal(props.loc['length_mean', 'feature_value'],
                                   60)


def test_border_feature_extractor():

    extractor = BorderFeatureExtractor()

    obj = np.zeros((100, 100), dtype=np.uint16)
    obj[:10, :10] = 1
    obj[10:20, -1] = 2
    obj[10:21, 0] = 3
    obj[-1, 40:52] = 4
    obj[20:30, 20:30] = 5

    props = extractor({'obj': obj}, None)
    props.set_index('object_id', inplace=True)

    assert props.loc[1].feature_value == 19
    assert props.loc[2].feature_value == 10
    assert props.loc[3].feature_value == 11
    assert props.loc[4].feature_value == 12
    assert props.loc[5].feature_value == 0
