import luigi
import logging
from luigi.util import requires
import numpy as np
from improc.resample import match_spacing

from goid.training.record import BuildTraingRecordBaseTask
from goid.training.training import ModelFittingBaseTask, JaccardLossParams, InferenceModelExportBaseTask


def standardize(img):
    '''custom "standardization".'''

    img = img.astype(np.float32)

    # align histogram based on quantiles, channel independent
    img -= np.quantile(img, 0.01, axis=(0, 1), keepdims=True)
    img /= np.quantile(img, 0.999, axis=(0, 1), keepdims=True)

    img -= 0.5

    return img


class BuildDebrisTraingRecordTask(BuildTraingRecordBaseTask):

    raw_filter = luigi.Parameter(
        'subdir=="raw"',
        description="pandas query string to get the input image")
    annot_filter = luigi.Parameter(
        'subdir=="annot"',
        description="pandas query string to get the annot image")
    mask_filter = luigi.Parameter(
        'subdir=="mask"',
        description="pandas query string to get the binary mask")
    input_downsampling = luigi.IntParameter(
        "downsampling factor at which the NN model operates")

    def _data_gen(self, inputs):
        '''generator to build the training record'''

        logger = logging.getLogger('luigi-interface')

        for idx, subdf in inputs:

            raw_row = subdf.query(self.raw_filter)
            mask_row = subdf.query(self.mask_filter)
            annot_row = subdf.query(self.annot_filter)

            if not (len(raw_row) == 1 and len(mask_row) == 1):
                continue

            logger.info('adding: {} to training record {}'.format(
                raw_row.dc.path[0], self.record_name))

            raw = raw_row.dc.read()[0]
            mask = mask_row.dc.read()[0].astype(bool)

            try:
                annot = annot_row.dc.read()[0].astype(np.int8)
            except Exception as e:  # missing annot
                annot = np.zeros_like(raw, dtype=np.int8)

            # debris over background were not annotated --> no loss during training
            annot = annot.clip(0)
            annot[~mask] = -1

            raw = match_spacing(
                raw,
                1, (self.input_downsampling, self.input_downsampling),
                image_type='greyscale')
            annot = match_spacing(annot,
                                  1,
                                  self.input_downsampling,
                                  image_type='label_nearest')

            #normalize input image
            raw = standardize(raw)

            # add channel dim
            yield raw[..., None], annot[..., None]

    def _get_serialization_fun(self):
        '''Returns a record parser serialization function'''
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.int8,
                                               fixed_ndim=3).serialize


@requires(BuildDebrisTraingRecordTask)
class DebrisModelFittingTask(JaccardLossParams, ModelFittingBaseTask):
    def split_samples(self, data):
        import tensorflow as tf

        data['segm'] = tf.minimum(tf.cast(data['segm'], tf.int32), 1)

        return data['image'], {'binary_seg': tf.cast(data['segm'], tf.float32)}

    def _get_parser_fun(self):
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.int8,
                                               fixed_ndim=3).parse

    def get_training_losses(self):
        from dlutils.losses.jaccard_loss import HingedBinaryJaccardLoss

        return {
            'binary_seg':
            HingedBinaryJaccardLoss(hinge_thresh=self.jaccard_hinge,
                                    eps=self.jaccard_eps)
        }


@requires(DebrisModelFittingTask)
class DebrisModelExportTask(InferenceModelExportBaseTask):
    pass


def main():
    luigi.run(main_task_cls=DebrisModelExportTask, local_scheduler=True)


if __name__ == "__main__":
    main()
