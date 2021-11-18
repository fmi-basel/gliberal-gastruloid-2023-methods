import os
import luigi
import numpy as np

from improc.morphology import clean_up_mask
from improc.resample import match_spacing
from skimage.transform import resize
from skimage.io import imread

from goid.luigi_utils import ExternalInputFile, BaseGoidGPUTask


def preproc(paths):
    img = imread(paths[0], img_num=0)
    orig_shape = img.shape
    img = match_spacing(img, 1, 4, image_type='greyscale')
    img = img.astype(np.float32)
    img -= np.quantile(img, 0.01)
    img /= np.quantile(img, 0.95)
    img -= 0.5
    return img[None, ..., None], {'orig_shape': orig_shape}


def postproc(pred, orig_shape, min_object_size=1000):
    fg = pred > 0.5
    fg = resize(fg, output_shape=orig_shape, preserve_range=True)

    fg = clean_up_mask(fg,
                       fill_holes=True,
                       size_threshold=min_object_size,
                       keep_largest=True)

    return fg.astype(np.uint16)


class PredictForegroundTask(BaseGoidGPUTask):
    ''''''

    min_object_size = luigi.IntParameter(
        1000,
        description=
        'Minimum object area in pixels to be considered a gastruloid.')
    model_dir = luigi.Parameter(
        description='directory of tensorflow foreground prediction model')
    channel = luigi.Parameter(1, description='channel id or stain')

    resources = {'gpu_submitter': 1}

    def requires(self):
        row = self.filter_dc_mip(ch=self.channel)[0]
        return ExternalInputFile(path=self.dcrow_to_path(row))

    def output(self):
        row = dict(self.filter_dc_mip(ch=self.channel)[0])
        row['subdir'] = 'FG_MASK'
        return luigi.LocalTarget(self.dcrow_to_path(row))

    def _get_gpu_task(self):
        return {
            'task_id': id(self),
            'model_dir': self.model_dir,
            'input_paths': [self.input().path],
            'preproc_fun': preproc,
            'postproc_fun': postproc,
            'output_path': self.output().path,
            'postproc_params': {
                'min_object_size': self.min_object_size
            }
        }
