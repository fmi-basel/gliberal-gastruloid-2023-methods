import itertools
import luigi
import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.graph import route_through_array
from skimage.morphology import medial_axis
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.graph import route_through_array
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread, imsave

from goid.luigi_utils import BaseGoidTask
from goid.foreground_model.predict import PredictForegroundTask
from goid.separator_model.predict import PredictSeparatorTask


def clear_border(mask):
    ''' # artificially add 1 px around border in case the mask is touching it'''

    mask[:1] = False
    mask[-1:] = False
    mask[:, :1] = False
    mask[:, -1:] = False

    return mask


def find_terminus(skel):
    '''Find endpoints of skeleton'''

    kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    res = convolve2d(skel, kernel, mode='same', boundary='fill', fillvalue=0)
    return np.array(np.where(res == 9)).T


def find_longest_skeleton_path(skel):
    '''Find the longest path between all possible pairs of terminuses'''

    terminus = find_terminus(skel)

    paths = []
    path_lengths = []

    for start, end in itertools.combinations(terminus, 2):
        path, _ = route_through_array(1 - skel,
                                      start,
                                      end,
                                      fully_connected=True,
                                      geometric=True)
        paths.append(path)
        path_lengths.append(len(path))

    return np.array(paths[np.argmax(path_lengths)])


def smooth_medial_axis(mask, separator=None, sigma=25., debug_plot=False):
    '''Smooth mask edges by applying a gaussian filter to the distance transform before medial axis.
    Additionaly a separator map can be subtracted at this stage to get skeleton of curled gastuloids'''

    if mask.max() < 1:
        return mask

    truncate = 4.
    pad_width = int(sigma * truncate) + 1
    mask = np.pad(mask, pad_width)
    dist = distance_transform_edt(mask)

    # apply separator to the distance transform
    # (separator applied to mask tend to get filled during smoothing)
    if separator is not None:
        sep_max = separator.max()

        if sep_max > 500:  # separator train range 0-1000
            separator = separator * dist.max() / sep_max
            separator = np.pad(separator, pad_width)
            dist -= separator

    dist = gaussian_filter(dist, sigma, truncate=truncate)
    # TODO modifiy Sk-image to directly input the distance transform instead of a binary mask
    mask = dist > 1 / sigma * dist.max()
    mask_filled = binary_fill_holes(
        mask)  # in case separator only introdcue a hole, fill it

    if np.any(mask_filled ^ mask):
        separator_create_hole = True
    else:
        separator_create_hole = False
    mask = mask_filled

    if debug_plot:
        plt.imshow(dist)
        plt.show()

        plt.imshow(mask)
        plt.show()

    mask = clear_border(mask)

    skel = medial_axis(mask)
    skel = skel[pad_width:-pad_width, pad_width:-pad_width]

    return skel, separator_create_hole


def separator_pred_to_binary(separator,
                             mask,
                             separator_create_hole,
                             threshold=500):
    '''Converts a predicted separator map to binary single pixel wide separator. 
    Ensures it connects to the bakground'''

    # if create hole, ignore separator even if above threshold
    if separator_create_hole:
        return np.zeros_like(mask)

    binary_separator = medial_axis(separator > 500)

    # if separator exist and overlap with mask
    if np.any(binary_separator) and np.any(mask[binary_separator]):

        # ensure separator is connectect to the background by adding the shortest path to the outside
        dist_to_bg = distance_transform_edt(mask & (~binary_separator))
        start_point = np.argwhere(~mask)[0]  # any point in background
        end_point = np.argwhere(binary_separator)[0]  # anypoint on separator

        bridge_path, _ = route_through_array(dist_to_bg,
                                             start_point,
                                             end_point,
                                             fully_connected=True)

        # extend separator with bridge path
        binary_separator[tuple(np.array(bridge_path).T)] = True

        # keep only longest path (in case connection to outside didn't exactly start at one of skeleton's end)
        longest_separator_path = find_longest_skeleton_path(binary_separator)
        binary_separator[:] = False
        binary_separator[tuple(np.array(longest_separator_path).T)] = True

        # keep separator only over background
        binary_separator[~mask] = False

    return binary_separator


class SkeletonTask(BaseGoidTask):
    ''''''

    use_separator = luigi.BoolParameter(
        True,
        'If true, uses the predicted curled gastruloid separator to compute the skeleton'
    )

    def requires(self):
        req = {'mask': PredictForegroundTask(dc_mip=self.dc_mip)}
        if self.use_separator:
            req['separator'] = PredictSeparatorTask(dc_mip=self.dc_mip)
        return req

    def output(self):
        row = dict(self.dc_mip[0])
        row['subdir'] = 'SKELETON'

        row_mask = row.copy()
        row_mask['subdir'] = 'FG_MASK_SEPARATOR'

        return {
            'skeleton': luigi.LocalTarget(self.dcrow_to_path(row)),
            'mask_with_sep': luigi.LocalTarget(self.dcrow_to_path(row_mask))
        }

    def run(self):

        mask = imread(self.input()['mask'].path)
        if self.use_separator:
            separator = imread(self.input()['separator'].path)
        else:
            separator = None

        if np.any(mask):
            skel, separator_create_hole = smooth_medial_axis(mask, separator)
            if self.use_separator:
                binary_separator = separator_pred_to_binary(
                    separator, mask, separator_create_hole)
                mask[binary_separator] = False
        else:
            skel = np.zeros_like(mask)

        skel_path = self.output()['skeleton'].path
        out_dir = os.path.dirname(skel_path)
        os.makedirs(out_dir, exist_ok=True)
        imsave(skel_path, skel.astype(np.uint16), compress=9)

        mask_sep_path = self.output()['mask_with_sep'].path
        out_dir = os.path.dirname(mask_sep_path)
        os.makedirs(out_dir, exist_ok=True)
        imsave(mask_sep_path, mask.astype(np.uint16), compress=9)
