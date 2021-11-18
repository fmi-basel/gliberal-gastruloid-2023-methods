import os
import luigi
import numpy as np

from skimage.exposure import rescale_intensity
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt2d
from skimage.io import imread, imsave

from goid.plate_layout import cached_experiment_layout_parser
from goid.luigi_utils import BaseGoidTask, ExternalInputFile
from goid.skeleton import PredictForegroundTask


def estimate_shading_mask(img, mask, thresh=0.3):
    '''Return of foreground object close to background level (likely shaded by debris)'''

    # convert to 8 bit (needed for median filter)
    img = rescale_intensity(img,
                            in_range=tuple(
                                np.quantile(img, [0.001, 0.999]).tolist()),
                            out_range=np.uint8).astype(np.uint8)

    low = np.quantile(img[~mask], 0.1)  # 10 percentile of background
    high = np.quantile(img[mask], 0.9)  # 90 percentile of foreground

    blurred_img = img.copy()
    # replace outside by foreground values to maintain contour when blurring
    blurred_img[~mask] = high
    # median blur: removes small holes between nuclei but keep sligthly larger/ long debris
    blurred_img = medfilt2d(blurred_img, 25)

    absolute_threshold = thresh * (high - low) + low

    shading_mask = blurred_img < absolute_threshold
    shading_mask[~mask] = False

    return shading_mask


class ShadingMaskTask(BaseGoidTask):
    ''''''

    shading_channel = luigi.Parameter(
        description='channel id or stain used to estimate shaded areas')

    def requires(self):
        row = self.filter_dc_mip(ch=self.shading_channel)[0]

        return {
            'intensities': ExternalInputFile(path=self.dcrow_to_path(row)),
            'mask': PredictForegroundTask(dc_mip=self.dc_mip)
        }

    def output(self):
        row = dict(self.dc_mip[0])
        row['subdir'] = 'SHADING_MASK'

        return luigi.LocalTarget(self.dcrow_to_path(row))

    def run(self):

        mask = imread(self.input()['mask'].path).astype(bool)
        intensities = imread(self.input()['intensities'].path, img_num=0)

        if not np.any(mask):
            shading_mask = np.zeros_like(mask)

        else:
            shading_mask = estimate_shading_mask(intensities, mask)

        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        imsave(self.output().path, shading_mask.astype(np.uint16), compress=9)
