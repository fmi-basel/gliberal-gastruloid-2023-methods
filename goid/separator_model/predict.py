import os
import luigi
import numpy as np

from improc.morphology import clean_up_mask
from improc.resample import match_spacing
from skimage.transform import resize
from skimage.io import imread

from goid.luigi_utils import ExternalInputFile, BaseGoidGPUTask


def preproc(paths):
    img = np.stack([imread(p, img_num=0) for p in paths], axis=-1)
    orig_shape = img.shape
    img = match_spacing(img, 1, (4, 4, 1), image_type='greyscale')
    img = img.astype(np.float32)
    img -= np.quantile(img, 0.01, axis=(0, 1), keepdims=True)
    img /= np.quantile(img, 0.95, axis=(0, 1), keepdims=True)
    img -= 0.5

    return img[None], {'orig_shape': orig_shape}


def postproc(pred, orig_shape):
    separator_map = resize(pred,
                           output_shape=orig_shape[:-1],
                           preserve_range=True)
    separator_map = separator_map.clip(0, 1000)
    return separator_map.astype(np.uint16)


class PredictSeparatorTask(BaseGoidGPUTask):
    ''''''

    model_dir = luigi.Parameter(
        description='directory of tensorflow debris prediction model')
    channels = luigi.ListParameter([1, 2, 3, 4],
                                   description='channel id or stain')

    resources = {'gpu_submitter': 1}

    def requires(self):
        rows = [self.filter_dc_mip(ch=ch)[0] for ch in self.channels]
        return [
            ExternalInputFile(path=self.dcrow_to_path(row)) for row in rows
        ]

    def output(self):
        row = dict(self.filter_dc_mip(ch=self.channels[0])[0])
        row['subdir'] = 'SEPARATOR'
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
