import os
import luigi
import numpy as np

from improc.morphology import clean_up_mask
from improc.resample import match_spacing
from skimage.transform import resize
from skimage.io import imread

from goid.luigi_utils import ExternalInputFile, BaseGoidGPUTask
from goid.foreground_model.predict import PredictForegroundTask


def preproc(paths):
    img = imread(paths[0], img_num=0)
    fg_mask = imread(paths[1]).astype(bool)
    orig_shape = img.shape
    img = match_spacing(img, 1, 4, image_type='greyscale')
    img = img.astype(np.float32)
    img -= np.quantile(img, 0.01, axis=(0, 1), keepdims=True)
    img /= np.quantile(img, 0.999, axis=(0, 1), keepdims=True)
    img -= 0.5
    return img[None, ..., None], {'orig_shape': orig_shape, 'fg_mask': fg_mask}


def postproc(pred, orig_shape, fg_mask):
    debris = pred > 0.5
    debris = resize(debris, output_shape=orig_shape, preserve_range=True)
    debris[~fg_mask] = False

    return debris.astype(np.uint16)


class PredictDebrisTask(BaseGoidGPUTask):
    ''''''

    model_dir = luigi.Parameter(
        description='directory of tensorflow debris prediction model')
    channel = luigi.Parameter(1, description='channel id or stain')

    resources = {'gpu_submitter': 1}

    def requires(self):
        row = self.filter_dc_mip(ch=self.channel)[0]

        return ExternalInputFile(path=self.dcrow_to_path(row)), \
               PredictForegroundTask(dc_mip=self.dc_mip)

    def output(self):
        row = dict(self.filter_dc_mip(ch=self.channel)[0])
        row['subdir'] = 'DEBRIS_MASK'
        return luigi.LocalTarget(self.dcrow_to_path(row))

    def _get_gpu_task(self):
        return {
            'task_id': id(self),
            'model_dir': self.model_dir,
            'input_paths': [val.path for val in self.input()],
            'preproc_fun': preproc,
            'postproc_fun': postproc,
            'output_path': self.output().path
        }
