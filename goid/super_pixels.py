import numpy as np
import logging
import luigi
from luigi.util import inherits
import os
from glob import glob

from skimage.io import imread, imsave

from improc.morphology import clean_up_mask

from goid.luigi_utils import BaseGoidTask, ExternalInputFile
from goid.foreground_model.predict import PredictForegroundTask
from goid.middle_plane import MiddlePlaneTask

from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
from skimage.segmentation import slic


def create_superpixels(image: np.ndarray, mask_image: np.ndarray,
                       cell_area: int) -> np.ndarray:
    '''loose re-implementation of Urs' IntensityFeatureSuperpixel.
    
    Returns a labelmap of the superpixels.

    '''
    OUTSIDE = 0

    def _estimate_segment_count(crop: np.ndarray):
        count = np.prod(crop.shape) // cell_area
        assert count >= 1
        return count

    def rescale(img):
        lower, upper = img.min(), img.max()
        return (img.astype(float) - lower) / (upper - lower)

    assert np.all(image.shape == mask_image.shape)

    image = rescale(image)
    image[np.logical_not(mask_image)] = 0
    spx = slic(image,
               n_segments=_estimate_segment_count(image),
               compactness=0.1,
               multichannel=False,
               enforce_connectivity=True,
               max_iter=50)

    # Only consider superpixels within mask
    spx[np.logical_not(mask_image)] = OUTSIDE
    # Remove small, cut superpixels
    remove_small_objects(spx, min_size=cell_area // 4, in_place=True)
    spx = relabel_sequential(spx)[0]
    return spx


class SuperPixelTask(BaseGoidTask):
    '''Super-pixel extraction on middle slice'''

    cell_area = luigi.IntParameter(
        description=
        'rough estimate of cell size. Used to initialize superpixel grid.')
    channel = luigi.Parameter(description='channel id or stain')

    def requires(self):
        center_surround = MiddlePlaneTask(self.dc_mip)
        req = {'center_surround': center_surround}

        if center_surround.output().exists():
            cs_path = center_surround.output().path
            zslice = self.parse_path(self.dc_mip[0], cs_path)['zslice']
            if zslice > 0:
                row = dict(self.filter_dc_mip(ch=self.channel)[0],
                           zslice=zslice,
                           subdir='TIF_OVR')
                req['mid_intensity'] = ExternalInputFile(
                    path=self.dcrow_to_path(row))
            # else, zslice zeros does not exist
            # fg mask was empty and no middle slice was found --> simply output empty superpixels

        return req

    def output(self):
        out_path = self.input()['center_surround'].path.replace(
            'CENTER_SURROUND', 'SUPER_PX')
        return luigi.LocalTarget(out_path)

    def run(self):
        middle_mask = imread(self.input()['center_surround'].path) > 0
        img_target = self.input().get('mid_intensity', None)
        if img_target:
            img = imread(img_target.path, img_num=0)
            spx = create_superpixels(img, middle_mask, self.cell_area)

        else:
            spx = np.zeros_like(middle_mask)

        self.output().makedirs()
        imsave(self.output().path, spx.astype(np.uint16), compress=9)


class MIPSuperPixelTask(BaseGoidTask):
    '''Super-pixel extraction on MIP'''
    cell_area = luigi.IntParameter(
        description=
        'rough estimate of cell size. Used to initialize superpixel grid.')
    channel = luigi.Parameter(description='channel id or stain')

    def requires(self):
        req = {'mask': PredictForegroundTask(dc_mip=self.dc_mip)}
        row = dict(self.filter_dc_mip(ch=self.channel)[0],
                   subdir='TIF_OVR_MIP')
        req['intensity'] = ExternalInputFile(path=self.dcrow_to_path(row))

        return req

    def output(self):
        out_path = self.input()['mask'].path.replace('FG_MASK', 'SUPER_PX_MIP')
        return luigi.LocalTarget(out_path)

    def run(self):

        mask = imread(self.input()['mask'].path).astype(bool)
        img = imread(self.input()['intensity'].path, img_num=0)
        spx = create_superpixels(img, mask, self.cell_area)

        self.output().makedirs()
        imsave(self.output().path, spx.astype(np.uint16), compress=9)
