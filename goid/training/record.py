import luigi
import abc
import os
import random
import numpy as np
from luigi.util import requires

from .training_collection import ParseCollectionTask


@requires(ParseCollectionTask)
class BuildTraingRecordBaseTask(luigi.Task):
    '''Base task to build train/valid/test records
    '''

    # yapf: disable
    sample_goup_key = luigi.ListParameter(['fname'], description="pandas grouping keys to get one sample per group from the input collection")
    record_name = luigi.Parameter()

    train_fraction = luigi.FloatParameter(default=0.8)
    valid_fraction = luigi.FloatParameter(default=0.1)
    seed = luigi.IntParameter()
    # yapf: enable

    def output(self):
        targets = {}
        for split_name in ['train', 'valid', 'test']:
            outpath = os.path.join(
                self.datadir, '{}_{}.tfrec'.format(self.record_name,
                                                   split_name))
            targets[split_name] = luigi.LocalTarget(outpath)

        return targets

    @abc.abstractmethod
    def _data_gen(self, inputs):
        '''generator to build the training record'''
        pass

    @abc.abstractmethod
    def _get_serialization_fun(self):
        '''Returns a record parser serialization function'''
        pass

    def _split_data(self):
        '''split annot/raw input pairs into train|valid|test sets'''

        if not 0. <= self.train_fraction <= 1.:
            raise ValueError(
                'training fraction should be within [0.,1.]. got {}'.format(
                    self.train_fraction))
        if not 0. <= self.valid_fraction <= 1.:
            raise ValueError(
                'training fraction should be within [0.,1.]. got {}'.format(
                    self.valid_fraction))

        df = self.input().load()

        # randomize sample order, e.g mixes timepoints
        record_input = list(df.groupby('fname'))
        random.seed(self.seed)
        random.shuffle(record_input)

        test_fraction = max(0., 1. - self.train_fraction - self.valid_fraction)
        n_samples = len(record_input)
        probs = [self.train_fraction, self.valid_fraction, test_fraction]
        split = np.random.choice([0, 1, 2],
                                 size=n_samples,
                                 replace=True,
                                 p=probs)
        sets = {}
        for idx, split_name in enumerate(['train', 'valid', 'test']):
            inputs = [p for s, p in zip(split, record_input) if s == idx]
            sets[split_name] = inputs

        return sets

    def run(self):
        from dlutils.dataset.tfrecords import tfrecord_from_iterable

        np.random.seed(self.seed)

        for split_name, inputs in self._split_data().items():

            with self.output()[split_name].temporary_path(
            ) as temp_output_path:
                tfrecord_from_iterable(temp_output_path,
                                       self._data_gen(inputs),
                                       self._get_serialization_fun(),
                                       verbose=False)
