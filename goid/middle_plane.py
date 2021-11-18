import numpy as np
import logging
import luigi
from luigi.util import inherits
import os
from glob import glob

from skimage.io import imread, imsave
from scipy.ndimage.morphology import distance_transform_edt
from goid.shading import estimate_shading_mask

from improc.morphology import clean_up_mask

from goid.luigi_utils import BaseGoidTask, ExternalInputFile
from goid.foreground_model.predict import PredictForegroundTask


def middle_slice_center_surround(max_proj_mask, stack):

    if not np.any(max_proj_mask):
        return np.zeros_like(max_proj_mask, dtype=np.uint16), 0

    stack_max_proj = stack.max(axis=0)

    # find object middle slice from stack
    stack_fg = stack.copy()
    stack_fg[~np.broadcast_to(max_proj_mask[None], stack_fg.shape)] = 0
    middle_slice = np.argmax(stack_fg.mean(axis=(1, 2)))

    # refine the mask on middle slice (remove dark regions)
    slice_mask = max_proj_mask & ~estimate_shading_mask(
        stack[middle_slice], max_proj_mask)
    slice_mask = clean_up_mask(slice_mask,
                               fill_holes=True,
                               size_threshold=1000,
                               keep_largest=True)

    # build inner/outer label from normalized distance transform
    dist = distance_transform_edt(slice_mask)
    dist /= dist.max()
    labels = np.zeros_like(slice_mask, dtype=np.uint16)
    labels[(dist > 0)] = 1
    labels[dist > 0.5] = 2

    # yokogawa slice naming start at 1
    return labels, middle_slice + 1


class MiddlePlaneTask(BaseGoidTask):
    '''Segments object middle plane based on intensity and split the mask in center/surround regions'''

    channel = luigi.Parameter(description='channel id or stain')

    def requires(self):
        row = dict(self.filter_dc_mip(ch=self.channel)[0],
                   subdir='TIF_OVR',
                   zslice='*')
        row['pattern'] = row['pattern'].replace('zslice:02d', 'zslice')
        matching_files = sorted(glob(self.dcrow_to_path(row)))

        if len(matching_files) <= 0:
            logger = logging.getLogger('luigi-interface')
            logger.error('zplanes matching MIP not found: {}'.format(
                self.dcrow_to_path(row)))

        return {
            'mask': PredictForegroundTask(dc_mip=self.dc_mip),
            'zplanes': [ExternalInputFile(path=p) for p in matching_files]
        }

    def output(self):
        row = dict(self.filter_dc_mip(ch=self.channel)[0],
                   zslice=0,
                   subdir='CENTER_SURROUND')

        row_glob = dict(row, zslice='*')
        row_glob['pattern'] = row_glob['pattern'].replace(
            'zslice:02d', 'zslice')
        matching_files = glob(self.dcrow_to_path(row_glob))

        if len(matching_files) > 0:
            return luigi.LocalTarget(matching_files[0])
        else:
            return luigi.LocalTarget(self.dcrow_to_path(row))

    def run(self):
        mask = imread(self.input()['mask'].path).astype(bool)
        stack = np.stack(
            [imread(t.path, img_num=0) for t in self.input()['zplanes']],
            axis=0)
        labels, middle_slice = middle_slice_center_surround(mask, stack)

        # override output with middle plane index determined during processing
        row_out = dict(self.filter_dc_mip(ch=self.channel)[0],
                       zslice=middle_slice,
                       subdir='CENTER_SURROUND')
        out_path = self.dcrow_to_path(row_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imsave(out_path, labels.astype(np.uint16), compress=9)
