import luigi
import os
import logging
import pandas as pd
import time

from improc.io import DCAccessor
DCAccessor.register()

from goid.luigi_utils import get_gpu_processor

from goid.collection import ParseGoidCollectionTask
from goid.foreground_model.predict import PredictForegroundTask
from goid.debris_model.predict import PredictDebrisTask
from goid.separator_model.predict import PredictSeparatorTask
from goid.skeleton import SkeletonTask
from goid.grid import GridTask
from goid.shading import ShadingMaskTask
from goid.props import MIPPropTask, AggreatePropsTask, MiddleSlicePropTask
from goid.middle_plane import MiddlePlaneTask
from goid.super_pixels import SuperPixelTask, MIPSuperPixelTask


class CompleteWorkflow(luigi.Task):

    compute_mip_props = luigi.BoolParameter(True)
    compute_middle_props = luigi.BoolParameter(True)
    aggregate_existing_props = luigi.BoolParameter(True)

    datadir = luigi.Parameter(description='base path of the experiment')
    platedir = luigi.ListParameter([])
    plate_row = luigi.ListParameter([])
    plate_column = luigi.ListParameter([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.starting_time_str = time.strftime("%Y%m%d-%H%M%S")

    def requires(self):

        # setup logging to file. better way to do it with luigi?
        logger = logging.getLogger('luigi-interface')
        formatter = logging.Formatter(
            '[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
            '%d.%m.%Y %I:%M:%S')
        fh = logging.FileHandler(os.path.join(self.datadir,
                                              'luigi_errors.log'))
        fh.setLevel(logging.ERROR)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return ParseGoidCollectionTask(datadir=self.datadir,
                                       filename='input_collection.h5')

    def run(self):
        df = pd.read_hdf(self.input().path)

        # TODO enfore that index start with ["platedir", "subdir", "plate_row", "plate_column", "channel", zlice

        df = df.dc[list(self.platedir) or slice(None), :,
                   list(self.plate_row) or slice(None),
                   list(self.plate_column) or slice(None)]

        outputs = []
        for (platedir, plate_row, plate_column
             ), subdf in df.dc[:, 'TIF_OVR_MIP'].reset_index().groupby(
                 ['platedir', 'plate_row', 'plate_column']):
            dc_mip = [row[1].to_dict() for row in subdf.iterrows()]

            # ~outputs.append(PredictDebrisTask(dc_mip=dc_mip))
            # ~outputs.append(PredictForegroundTask(dc_mip=dc_mip))
            # ~outputs.append(PredictSeparatorTask(dc_mip=dc_mip))
            # ~outputs.append(SkeletonTask(dc_mip=dc_mip))
            # ~outputs.append(GridTask(dc_mip=dc_mip))
            # ~outputs.append(ShadingMaskTask(dc_mip=dc_mip))
            # ~outputs.append(MIPSuperPixelTask(dc_mip=dc_mip))
            if self.compute_mip_props:
                outputs.append(MIPPropTask(dc_mip=dc_mip))

            # ~outputs.append(MiddlePlaneTask(dc_mip=dc_mip))
            # ~outputs.append(SuperPixelTask(dc_mip=dc_mip))
            if self.compute_middle_props:
                outputs.append(MiddleSlicePropTask(dc_mip=dc_mip))

        yield outputs

        if self.aggregate_existing_props:
            outdir = os.path.join(
                self.datadir, 'workflow_out_{}'.format(self.starting_time_str))

            yield AggreatePropsTask(datadir=self.datadir, outdir=outdir)


def main():
    gpu_processor = get_gpu_processor()
    luigi.run(main_task_cls=CompleteWorkflow, local_scheduler=True)
    gpu_processor.soft_stop()


if __name__ == '__main__':
    main()
