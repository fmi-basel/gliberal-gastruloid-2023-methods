import luigi
import logging
from luigi.util import requires
import numpy as np
from improc.resample import match_spacing
from scipy.ndimage.filters import gaussian_filter

from goid.training.record import BuildTraingRecordBaseTask
from goid.training.training import ModelFittingBaseTask, JaccardLossParams, InferenceModelExportBaseTask


def standardize(img):
    '''custom "standardization".'''

    img = img.astype(np.float32)

    # align histogram based on quantiles, channel independent
    img -= np.quantile(img, 0.01, axis=(0, 1), keepdims=True)
    img /= np.quantile(img, 0.95, axis=(0, 1), keepdims=True)

    img -= 0.5

    return img


def smooth_separator(sep, sigma=10.):

    sep = gaussian_filter(sep.astype(np.float32), sigma)

    if sep.max() > 0:
        sep /= sep.max()
    sep *= 1000.

    return sep


class BuildSeparatorTraingRecordTask(BuildTraingRecordBaseTask):

    raw_filter = luigi.Parameter(
        'subdir=="raw"',
        description="pandas query string to get the input image")
    annot_filter = luigi.Parameter(
        'subdir=="annot"',
        description="pandas query string to get the annot image")
    input_downsampling = luigi.IntParameter(
        "downsampling factor at which the NN model operates")

    def _data_gen(self, inputs):
        '''generator to build the training record'''

        logger = logging.getLogger('luigi-interface')

        for idx, subdf in inputs:

            raw_row = subdf.query(self.raw_filter)
            annot_row = subdf.query(self.annot_filter)

            if not (len(raw_row) == 1 and len(annot_row) == 1):
                continue

            logger.info(
                'adding raw|annot pair to training record: {} - {} to {}'.
                format(raw_row.dc.path[0], annot_row.dc.path[0],
                       self.record_name))

            raw = raw_row.dc.read()[0]
            annot = annot_row.dc.read()[0]

            raw = match_spacing(
                raw,
                1, (self.input_downsampling, self.input_downsampling, 1),
                image_type='greyscale')
            annot = match_spacing(annot,
                                  1,
                                  self.input_downsampling,
                                  image_type='label_nearest')

            #normalize input image
            raw = standardize(raw)

            # replace annot by smooth map
            annot = smooth_separator(annot == 2)

            # add channel dim
            yield raw, annot[..., None]

    def _get_serialization_fun(self):
        '''Returns a record parser serialization function'''
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.float32,
                                               fixed_ndim=3).serialize


@requires(BuildSeparatorTraingRecordTask)
class SeparatorModelFittingTask(ModelFittingBaseTask):
    def split_samples(self, data):
        import tensorflow as tf

        return data['image'], {'separator': data['segm']}

    def _get_parser_fun(self):
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.float32,
                                               fixed_ndim=3).parse

    def get_training_losses(self):
        import tensorflow as tf

        return {'separator': tf.keras.losses.MSE}


@requires(SeparatorModelFittingTask)
class SeparatorModelExportTask(InferenceModelExportBaseTask):
    pass


def main():
    luigi.run(main_task_cls=SeparatorModelExportTask, local_scheduler=True)


if __name__ == "__main__":
    main()
