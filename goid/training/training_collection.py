import os
import pandas as pd
import luigi
import logging

from improc.io import parse_collection, DCAccessor

DCAccessor.register()


class CollectionTarget(luigi.LocalTarget):
    extension = 'h5'

    def __init__(self, outdir, filename, *args, **kwargs):
        '''
        '''
        path = os.path.join(outdir, '{}.{}'.format(filename, self.extension))
        super().__init__(path, *args, **kwargs)

    def load(self):
        '''
        '''
        return pd.read_hdf(self.path, 'dc')

    def save(self, df, task_config, **kwargs):
        '''
        '''

        config = pd.Series(task_config)
        self.makedirs()

        with self.temporary_path() as path:
            df.to_hdf(path, key='dc', **kwargs)
            config.to_hdf(path, key='config', **kwargs)


class ParseCollectionTask(luigi.Task):
    ''''''

    datadir = luigi.Parameter()
    pattern = luigi.Parameter()
    index = luigi.ListParameter(["subdir", "fname"])

    filename = luigi.Parameter(
        'collection_input', description='filename of exported data collection')
    outdir = luigi.OptionalParameter(None)

    def requires(self):
        return None

    def output(self):
        if self.outdir is None:
            self.outdir = self.datadir

        return CollectionTarget(self.outdir, self.filename)

    def run(self):
        df = parse_collection(os.path.join(self.datadir, self.pattern),
                              list(self.index))

        self.output().save(df, self.param_kwargs)
