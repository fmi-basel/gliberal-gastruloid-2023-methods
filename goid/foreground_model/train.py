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

    # align histogram based on quantiles
    img -= np.quantile(img, 0.01)
    img /= np.quantile(img, 0.95)

    img -= 0.5

    return img


class BuildFGTraingRecordTask(BuildTraingRecordBaseTask):

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

            raw = match_spacing(raw,
                                1,
                                self.input_downsampling,
                                image_type='greyscale')
            annot = match_spacing(annot,
                                  1,
                                  self.input_downsampling,
                                  image_type='label_nearest')

            #normalize input image
            raw = standardize(raw)

            # remove negative value (this dataset should not have unannotated regions)
            annot = annot.clip(0, 1).astype(np.uint8)

            # add channel dim
            yield raw[..., None], annot[..., None]

    def _get_serialization_fun(self):
        '''Returns a record parser serialization function'''
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.uint8,
                                               fixed_ndim=3).serialize


@requires(BuildFGTraingRecordTask)
class FGModelFittingTask(JaccardLossParams, ModelFittingBaseTask):
    def split_samples(self, data):
        import tensorflow as tf

        return data['image'], {'binary_seg': tf.cast(data['segm'], tf.float32)}

    def _get_parser_fun(self):
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.uint8,
                                               fixed_ndim=3).parse

    def get_training_losses(self):
        from dlutils.losses.jaccard_loss import HingedBinaryJaccardLoss

        return {
            'binary_seg':
            HingedBinaryJaccardLoss(hinge_thresh=self.jaccard_hinge,
                                    eps=self.jaccard_eps)
        }


@requires(FGModelFittingTask)
class FGModelExportTask(InferenceModelExportBaseTask):
    pass


def main():
    luigi.run(main_task_cls=FGModelExportTask, local_scheduler=True)


if __name__ == "__main__":
    main()
