import os
import luigi
import traceback

from improc.io import parse_collection
from goid.plate_layout import cached_experiment_layout_parser


def add_plate_info(df, plate_layout, sheet):

    df['stain'] = 'na'
    df['condition'] = 'na'

    def _get_plate_info(subdf):

        platedir, plate_row, plate_column, channel = subdf.reset_index(
        ).iloc[0][['platedir', 'plate_row', 'plate_column', 'channel']]

        plate_layout_path = os.path.join(
            subdf.iloc[[0], :].dc.path[0].split(platedir)[0], platedir,
            plate_layout)
        elp = cached_experiment_layout_parser(plate_layout_path, sheet)

        try:
            subdf['stain'] = elp.ch_to_stain[plate_row, plate_column, channel]
        except KeyError as e:
            traceback.print_exc()
            raise KeyError(
                'Stain mapping for plate {}, row {}, col {}, channel {}, not found'
                .format(platedir, plate_row, plate_column, channel))

        try:
            subdf['condition'] = elp.well_to_condition[plate_row, plate_column]
        except KeyError as e:
            traceback.print_exc()
            raise KeyError(
                'condition mapping for plate {}, row {}, col {}, not found'.
                format(platedir, plate_row, plate_column))

        return subdf

    return df.groupby(['platedir', 'plate_row', 'plate_column',
                       'channel']).apply(_get_plate_info)


class ParseCollectionTask(luigi.Task):
    ''''''

    datadir = luigi.Parameter()
    pattern = luigi.Parameter()
    index = luigi.ListParameter([
        "platedir", "subdir", "plate_row", "plate_column", "channel", "zslice"
    ])
    filename = luigi.Parameter(
        'parsed_collection.h5',
        description='output filename for parsed collection')
    outdir = luigi.OptionalParameter(None)

    def requires(self):
        return None

    def output(self):
        if self.outdir is None:
            self.outdir = self.datadir
        return luigi.LocalTarget(os.path.join(self.outdir, self.filename))

    def run(self):
        df = parse_collection(os.path.join(self.datadir, self.pattern),
                              list(self.index))

        self.output().makedirs()
        df.to_hdf(self.output().path, key='dc')


class ParseGoidCollectionTask(ParseCollectionTask):
    ''''''
    plate_layout = luigi.OptionalParameter(
        None,
        description='path of .xlsx plate layout file, relative to platedir')
    plate_layout_sheet = luigi.Parameter(
        '384-well plate',
        description=
        'name of sheet containing the plate layout in .xlsx template')

    def run(self):
        df = parse_collection(os.path.join(self.datadir, self.pattern),
                              list(self.index))

        if self.plate_layout is not None:
            df = add_plate_info(df, self.plate_layout, self.plate_layout_sheet)

        self.output().makedirs()
        df.to_hdf(self.output().path, key='dc')  #, format='table')
