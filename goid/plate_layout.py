import pandas as pd
import numpy as np
from threading import RLock
from functools import lru_cache

_NOT_FOUND = object()


class cached_property:
    # cached_property from functools python 3.8
    # https://github.com/python/cpython/blob/master/Lib/functools.py#L1169
    # NOTE functools.lru_cache() applied on class method/properties creates
    # class level cache and prevents garbage collection

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r}).")

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (f"No '__dict__' attribute on {type(instance).__name__!r} "
                   f"instance to cache {self.attrname!r} property.")
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    try:
                        cache[self.attrname] = val
                    except TypeError:
                        msg = (
                            f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {self.attrname!r} property."
                        )
                        raise TypeError(msg) from None
        return val


class ExperimentLayoutParser():
    '''Basic experiment layout parser. Can be used to get stain to channel
    mapping and vice versa.
    
    Args:
        path: path of .xlsx layout
        sheet: name of sheet to parse in .xlsx layout
    '''

    # TODO read condition section
    # TODO handle exception, inform which section does not comply with template
    # TODO handle edge cases (empty array, etc.)

    def __init__(self, path, sheet):
        self.path = path
        self.sheet = sheet

        self.barcode, self.plate_df, self.condition_df, self.stain_df = self._parse_sheet(
        )

    def _parse_sheet(self):
        # load raw sheet array
        template = pd.read_excel(self.path,
                                 sheet_name=self.sheet,
                                 header=None,
                                 index_col=None,
                                 engine='openpyxl').values

        # find top left corner (and trim if necessary)
        row0, col0 = np.argwhere(template == 'Barcode')[0]
        template = template[row0:, col0:]

        # read barcode
        barcode = template[0, 1]

        # read plate section
        plate_row0 = np.argwhere(template[:, 0] == 'A')[0][0] - 1
        plate_index, stain_nrows = self._trim_at_first_nan(
            template[plate_row0 + 1:, 0])
        plate_columns, stain_ncols = self._trim_at_first_nan(
            template[plate_row0, 1:])
        plate_df = pd.DataFrame(template[plate_row0 + 1:plate_row0 +
                                         stain_nrows + 1, 1:stain_ncols + 1],
                                index=plate_index,
                                columns=plate_columns.astype(int))

        # read condition section
        condition_row0 = np.argwhere(template[:, 0] == 'Well*')[0][0]
        condition_index, condition_nrows = self._trim_at_first_nan(
            template[condition_row0 + 1:, 0])
        condition_columns, condition_ncols = self._trim_at_first_nan(
            template[condition_row0, 1:])
        condition_df = pd.DataFrame(
            template[condition_row0 + 1:condition_row0 + 1 + condition_nrows,
                     1:condition_ncols + 1],
            index=condition_index,
            columns=condition_columns)

        # read stain section
        stain_row0 = np.argwhere(template[:, 0] == 'Well*')[-1][0]
        stain_index, stain_nrows = self._trim_at_first_nan(
            template[stain_row0 + 1:, 0])
        stain_columns, stain_ncols = self._trim_at_first_nan(
            template[stain_row0, 1:])
        stain_df = pd.DataFrame(template[stain_row0 + 1:stain_row0 + 1 +
                                         stain_nrows, 1:stain_ncols + 1],
                                index=stain_index,
                                columns=stain_columns)

        return barcode, plate_df, condition_df, stain_df

    @cached_property
    def condition_plate_df(self):
        '''Flat dataframe with 'col', 'row', 'condition_id' columns'''

        # yapf: disable
        df = (self.plate_df
                  .applymap(lambda x: x.split('-')[0].strip() if isinstance(x, str) else np.nan)
                  .unstack()
                  .dropna()
                  .reset_index())
        # yapf: enable
        df.columns = ['col', 'row', 'condition_id']
        return df

    @cached_property
    def stain_plate_df(self):
        '''Flat Dataframe with 'col', 'row', 'stain_id' columns'''

        # yapf: disable
        df = (self.plate_df
                  .applymap(lambda x: x.split('-')[1].strip() if isinstance(x, str) else np.nan)
                  .unstack()
                  .dropna()
                  .astype(np.int)
                  .reset_index())
        # yapf: enable
        df.columns = ['col', 'row', 'stain_id']
        return df

    @cached_property
    def condition_mapping_df(self):
        '''Dataframe mapping plate row,col to experimental conditions'''
        # lookup staining for each well
        df = self.condition_df.loc[
            self.condition_plate_df.condition_id].reset_index(drop=True)

        # add columns indicating plate row/col
        df = self.condition_plate_df[['col', 'row']].join(df)

        return df.set_index(['row', 'col'])

    @cached_property
    def well_to_condition(self):
        '''condition series with ['row', 'col'] multi-index'''

        return self.condition_mapping_df['Condition*'].rename('condition')

    @cached_property
    def stain_mapping_df(self):
        '''Dataframe mapping plate row,col to staining conditions'''
        # lookup staining for each well
        df = self.stain_df.loc[self.stain_plate_df.stain_id].reset_index(
            drop=True)

        # add columns indicating plate row/col
        df = self.stain_plate_df[['col', 'row']].join(df)

        return df.set_index(['row', 'col'])

    @cached_property
    def ch_to_stain(self):
        '''stain series with ['row', 'col', 'channel'] multi-index'''

        df = (self.stain_mapping_df.rename(
            columns={
                'Channel01*': 1,
                'Channel02*': 2,
                'Channel03*': 3,
                'Channel04*': 4
            })[[1, 2, 3, 4]].stack())

        df.index.names = ['row', 'col', 'channel']
        df.name = 'stain'

        return df

    @cached_property
    def stain_to_ch(self):
        '''channel series with ['row', 'col', 'stain'] multi-index'''

        return self.ch_to_stain.reset_index().set_index(
            ['row', 'col', 'stain']).iloc[:, 0]

    def get_first_matching_stain_to_ch(self, row, col, stain_candidates):
        '''Returns the channel id of the first matched staining or None if there is no match'''

        for stain in stain_candidates:
            try:
                return self.stain_to_ch[row, col, stain]
            except KeyError as e:
                pass
        return None

    @staticmethod
    def _trim_at_first_nan(arr):
        '''trim a 1d array (e.g. index or columns) at the first encountered nan.
        returns the trimmed array and its length'''

        arr_length = len(arr)
        nan_idxs = np.nonzero(pd.isnull(arr))[0]
        if len(nan_idxs) > 0:
            arr_length = nan_idxs[0]

        return arr[:arr_length], arr_length


@lru_cache(maxsize=32)
def cached_experiment_layout_parser(path, sheet):
    return ExperimentLayoutParser(path, sheet)
