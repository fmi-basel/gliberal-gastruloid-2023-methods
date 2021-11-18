import numpy as np
import luigi
import os
import pandas as pd
import itertools
import re

from improc.regionprops import BaseFeatureExtractor, QuantilesFeatureExtractor, IntensityFeatureExtractor, DistanceTransformFeatureExtractor, SKRegionPropFeatureExtractor
from improc.regionprops import DerivedFeatureCalculator, HybridDerivedFeatureCalculator, RegionDerivedFeatureCalculator
from improc.regionprops import GlobalFeatureExtractor
from skimage.measure import regionprops, regionprops_table
from scipy.ndimage import find_objects
from skimage.io import imread, imsave
from skimage.feature import greycomatrix, greycoprops
from skimage.exposure import rescale_intensity
from scipy.stats import pearsonr

from goid.skeleton import find_terminus
from goid.luigi_utils import BaseGoidTask, ExternalInputFile
from goid.skeleton import SkeletonTask
from goid.foreground_model.predict import PredictForegroundTask
from goid.debris_model.predict import PredictDebrisTask
from goid.shading import ShadingMaskTask
from goid.grid import GridTask
from goid.middle_plane import MiddlePlaneTask
from goid.super_pixels import SuperPixelTask, MIPSuperPixelTask
from goid.collection import ParseGoidCollectionTask


class SkeletonFeatureExtractor(BaseFeatureExtractor):
    '''Extract skeleton features. Expects skeletons to be passed as "labels"'''

    _features_functions = {
        'n_branches': lambda x: len(find_terminus(x)),
    }
    _implemented_features = set(_features_functions.keys())

    def __init__(self, features=['n_branches'], *args, **kwargs):

        # override channel target
        try:
            del kwargs['channel_targets']
        except KeyError:
            pass
        super().__init__(channel_targets=None, *args, **kwargs)

        for f in set(features) - self._implemented_features:
            raise NotImplementedError('feature {} not implemented'.format(f))

        self.features = features

    def _extract_features(self, labels, intensity):

        unique_l = np.unique(labels)
        unique_l = unique_l[unique_l != 0]

        props = {f: [] for f in self.features}
        unique_l = []

        locs = find_objects(labels)
        for l, loc in enumerate(locs, start=1):
            if loc:
                unique_l.append(l)
                crop = np.pad(labels[loc] > 0, 1)

                for f in self.features:
                    props[f].append(self._features_functions[f](crop))

        props['label'] = unique_l

        return props


class GridFeatureExtractor(BaseFeatureExtractor):
    '''Extract grid features. Expects grid to be passed as "labels"'''
    def __init__(self, *args, **kwargs):

        # override channel target
        try:
            del kwargs['channel_targets']
        except KeyError:
            pass
        super().__init__(channel_targets=None, *args, **kwargs)

    def _extract_features(self, labels, intensity):
        grid = labels

        widths = np.linalg.norm(np.diff(grid, axis=0), axis=-1).sum(axis=0)
        lengths = np.linalg.norm(np.diff(grid, axis=1), axis=-1).sum(axis=1)

        props = {}

        quantiles = [0., 0.25, 0.5, 0.75, 1.]

        props.update({
            'width_q{:.3f}'.format(qt).replace('.', '_'): qt_val
            for qt, qt_val in zip(quantiles, np.quantile(widths, quantiles))
        })
        props['width_mean'] = widths.mean()
        props['width_std'] = widths.std()

        props.update({
            'length_q{:.3f}'.format(qt).replace('.', '_'): qt_val
            for qt, qt_val in zip(quantiles, np.quantile(lengths, quantiles))
        })
        props['length_mean'] = lengths.mean()
        props['length_std'] = lengths.std()

        props['label'] = [1]

        return props


class BorderFeatureExtractor(BaseFeatureExtractor):
    '''Extract features related to image border'''
    def _extract_features(self, labels, intensity):

        unique_l = np.unique(labels)
        unique_l = unique_l[unique_l != 0]

        props = {'border_pixels': []}
        for l in unique_l:
            mask = labels == l
            border_count = mask[0,:].sum() + \
                           mask[-1,:].sum() + \
                           mask[1:-1,0].sum() + \
                           mask[1:-1,-1].sum()

            props['border_pixels'].append(border_count)

        props['label'] = unique_l

        return props


class CorrelationFeatureExtractor(BaseFeatureExtractor):
    '''Extract Pearson's correlation over labels for pairs of channel supplied as "intensity'''
    def _extract_features(self, labels, intensity):

        unique_l = np.unique(labels)
        unique_l = unique_l[unique_l != 0]
        locs = find_objects(labels)

        props = {'PearsonR': []}

        for l, loc in zip(unique_l, locs):
            if loc:
                img1 = intensity[0][loc]
                img2 = intensity[1][loc]
                mask = labels[loc] == l

                props['PearsonR'] = pearsonr(img1[mask], img2[mask])[0]

        props['label'] = unique_l

        return props


class GLCMFeatureExtractor(BaseFeatureExtractor):
    '''Extract features related to grey level co-occurrence matrices.
    
    Simplifed version for gastruloid analysis that only return the mean correlation over 8 orientations'''
    def __init__(self, distances=(1, ), n_levels=256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.distances = distances
        self.n_levels = n_levels

    def _extract_features(self, labels, intensity):

        unique_l = np.unique(labels)
        unique_l = unique_l[unique_l != 0]
        locs = find_objects(labels)
        correlations = []

        for l, loc in zip(unique_l, locs):
            if loc:
                img = intensity[loc]
                mask = labels[loc] == l

                obj_vals = img[mask]
                img = rescale_intensity(
                    img,
                    in_range=(obj_vals.min(), obj_vals.max()),
                    out_range=(0, self.n_levels - 1)).astype(np.uint8)
                glcm = greycomatrix(
                    img,
                    distances=self.distances,
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=self.n_levels,
                    symmetric=True,
                    normed=True)

                # "ignore zero levels" as done in cell profiler
                # glcm[0] = 0
                # glcm[:,0] = 0

                # mean over directions
                correlations.append(
                    greycoprops(glcm, 'correlation').mean(axis=1))

        correlations = np.stack(correlations, axis=-1)
        props = {
            'GLCM_correlation_dist{}'.format(d): correlations[d_idx]
            for d_idx, d in enumerate(self.distances)
        }
        props['label'] = unique_l

        return props


class SPXFeatureExtractor(BaseFeatureExtractor):
    '''Extract features related to super pixels'''
    def __init__(self, *args, **kwargs):

        # override channel target
        try:
            del kwargs['channel_targets']
        except KeyError:
            pass
        super().__init__(channel_targets=None, *args, **kwargs)

    def _extract_features(self, spx_labels, _):
        # center of entire mask to compute relative coordinates
        center = np.asarray(
            regionprops((spx_labels >= 1).astype(np.uint8))[0].centroid)
        props = regionprops_table(spx_labels,
                                  properties=['centroid', 'label'],
                                  separator='-')

        props['center_x'] = props.pop('centroid-0') - center[0]
        props['center_y'] = props.pop('centroid-1') - center[1]
        props['center_radius'], props['center_phi'] = self.cartesian_to_polar(
            props['center_x'], props['center_y'])

        return props

    @staticmethod
    def cartesian_to_polar(x, y):
        return np.sqrt(x**2 + y**2), np.arctan2(y, x)


class GoidDerivedFeatureCalculator(DerivedFeatureCalculator):
    @staticmethod
    def aspect_ratio(length_q0_500, width_q0_500):
        '''grid aspect ratio'''

        return length_q0_500 / width_q0_500


class GoidRegionDerivedFeatureCalculator(RegionDerivedFeatureCalculator):
    @staticmethod
    def shading_fraction(shading__area, obj__area):
        return shading__area / obj__area

    @staticmethod
    def debris_fraction(debris__area, obj__area):
        return debris__area / obj__area


class GastruloidFeatureExtractor():
    '''Custom feature extractor for gastruloid'''

    # NOTE as opposed to closure, class is pickable for multiprocessing

    segments_pattern = re.compile("l[0-9]+seg")

    def __init__(self):

        # yapf: disable
        self.feature_extractor = GlobalFeatureExtractor(extractors=[
            QuantilesFeatureExtractor(
                label_targets=['img', 'obj_corr'],
                channel_targets=[1, 2, 3, 4],
                quantiles=[0., 0.001, 0.25, 0.5, 0.75, 0.999, 1.0]),

            QuantilesFeatureExtractor(
                label_targets=['l3seg_corr', 'l100seg_corr'],
                channel_targets=[1, 2, 3, 4],
                quantiles=[0., 0.25, 0.5, 0.75, 1.0]),

            IntensityFeatureExtractor(
                features=['mean', 'std', 'mad'],
                label_targets=['img', 'obj_corr', 'l3seg_corr', 'l100seg_corr'],
                channel_targets=[1, 2, 3, 4],),

            SKRegionPropFeatureExtractor(
                features=['weighted_centroid'],
                label_targets=['obj_corr'],
                channel_targets=[1, 2, 3, 4],
                 physical_coords=False),

            SKRegionPropFeatureExtractor(
                features=['area', 'centroid', 'minor_axis_length', 'major_axis_length',
                    'eccentricity', 'perimeter', 'convex_area', 'convex_perimeter',
                    'solidity'],
                label_targets=['obj'],
                channel_targets=None,
                physical_coords=False),

            BorderFeatureExtractor(
                label_targets=['obj'],
                channel_targets=None,
                ),

            SKRegionPropFeatureExtractor(
                features=['area'],
                label_targets=['l3seg', 'l100seg'],
                channel_targets=None,
                physical_coords=False),

            DistanceTransformFeatureExtractor(
                features=['mean_radius', 'max_radius', 'median_radius'],
                label_targets=['obj'],
                channel_targets=None,
                physical_coords=False),

            GridFeatureExtractor(label_targets=['grid']),

            SkeletonFeatureExtractor(label_targets=['skel']),

            SPXFeatureExtractor(label_targets=['spxmip']),

            IntensityFeatureExtractor(features=['mean'],
                                      label_targets=['spxmip'],
                                      channel_targets=[1, 2, 3, 4]),
        ])

        self.feature_calculators = [
            DerivedFeatureCalculator(
                features=['convexity', 'form_factor'],
                label_targets=['obj']),

            HybridDerivedFeatureCalculator(
                features=['mass_displacement'],
                label_targets=['obj']),

            GoidRegionDerivedFeatureCalculator(
                features=['shading_fraction', 'debris_fraction'],
                label_targets=['obj', 'shading', 'debris']),

            GoidDerivedFeatureCalculator(
                features=['aspect_ratio'],
                label_targets=['grid']),
        ]
        # yapf: enable

    @staticmethod
    def _correct_labels(labels, shading, debris):
        '''Returns a copy of labels with shaded and debris region set to zero'''

        labels_corr = labels.copy()
        labels_corr[shading.astype(bool) | debris.astype(bool)] = 0
        return labels_corr

    def _add_debris_area(self, props, debris):
        '''Manually add debris area, even if empty mask'''

        return props.append(
            pd.DataFrame([{
                'channel': 'na',
                'region': 'debris',
                'object_id': 1,
                'feature_name': 'area',
                'feature_value': debris.sum()
            }]))

    def _add_shading_area(self, props, shading):
        '''Manually add shadimg area, even if empty mask'''

        return props.append(
            pd.DataFrame([{
                'channel': 'na',
                'region': 'shading',
                'object_id': 1,
                'feature_name': 'area',
                'feature_value': shading.sum()
            }]))

    def __call__(self, labels, channels):

        labels['obj_corr'] = self._correct_labels(labels['obj'],
                                                  labels['shading'],
                                                  labels['debris'])

        for key in list(labels.keys()):
            if bool(self.segments_pattern.match(key)):
                labels[key + '_corr'] = self._correct_labels(
                    labels[key], labels['shading'], labels['debris'])

        props = self.feature_extractor(labels, channels)

        props['region'] = props['region'].str.replace('_corr', '')

        props = self._add_shading_area(props, labels['shading'])
        props = self._add_debris_area(props, labels['debris'])

        for calculator in self.feature_calculators:
            props = props.append(calculator(props))

        return props


class MIPPropTask(BaseGoidTask):
    ''''''

    mask_debris = luigi.BoolParameter(
        True,
        description=
        'If true, ignore regions marked as debris when computing intensity based features'
    )
    spx_props = luigi.BoolParameter(
        False, description='If true, extract super pixels featurs on MIPs')
    ch_to_stain = luigi.BoolParameter(
        False,
        description='If true, replaces channel IDs by stain name (if available)'
    )

    def requires(self):
        req = {
            'channels': {
                row['channel']: ExternalInputFile(path=self.dcrow_to_path(row))
                for row in self.dc_mip
            },
            'skel_out': SkeletonTask(dc_mip=self.dc_mip),
            'mask': PredictForegroundTask(dc_mip=self.dc_mip),
            'shading_mask': ShadingMaskTask(dc_mip=self.dc_mip),
            'grid_out': GridTask(dc_mip=self.dc_mip)
        }

        if self.mask_debris:
            req['debris_mask'] = PredictDebrisTask(dc_mip=self.dc_mip)

        if self.spx_props:
            req['spxmip'] = MIPSuperPixelTask(dc_mip=self.dc_mip)

        return req

    def output(self):
        row = dict(self.dc_mip[0])
        row['subdir'] = 'PROPS_MIP'
        row['ext'] = 'csv'

        return luigi.LocalTarget(self.dcrow_to_path(row))

    def run(self):

        channels = {
            ch: imread(t.path, img_num=0)
            for ch, t in self.input()['channels'].items()
        }
        labels = {}
        labels['obj'] = imread(self.input()['mask'].path)
        labels['shading'] = imread(self.input()['shading_mask'].path)
        labels['skel'] = imread(self.input()['skel_out']['skeleton'].path)
        labels['img'] = np.ones_like(labels['obj'])

        if self.mask_debris:
            labels['debris'] = imread(self.input()['debris_mask'].path)
        else:
            labels['debris'] = np.zeros_like(labels['obj'])

        if self.spx_props:
            labels['spxmip'] = imread(self.input()['spxmip'].path)

        for (axis, n_seg), target in self.input()['grid_out'][0].items():
            labels['{}{}seg'.format(axis, n_seg)] = imread(target.path)
        labels['grid'] = np.load(self.input()['grid_out'][1].path)['arr']

        feature_extractor = GastruloidFeatureExtractor()
        props = feature_extractor(labels, channels)

        if self.ch_to_stain:
            props = self.map_ch_to_stain(props)

        props['platedir'] = self.dc_mip[0]['platedir']
        props['plate_row'] = self.dc_mip[0]['plate_row']
        props['plate_column'] = self.dc_mip[0]['plate_column']

        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        props.to_csv(self.output().path, index=False)

    def map_ch_to_stain(self, props):
        lut = {ch: ch for ch in props.channel.unique()}
        lut.update({
            row['channel']: row.get('stain', row['channel'])
            for row in self.dc_mip
        })
        props.channel = props.channel.map(lut)

        return props


class MiddleSlicePropTask(BaseGoidTask):
    ''''''

    ch_to_stain = luigi.BoolParameter(
        False,
        description='If true, replaces channel IDs by stain name (if available)'
    )

    def requires(self):
        center_surround = MiddlePlaneTask(self.dc_mip)
        req = {
            'center_surround': center_surround,
            'spx': SuperPixelTask(self.dc_mip)
        }

        if center_surround.output().exists():
            cs_path = center_surround.output().path
            zslice = self.parse_path(self.dc_mip[0], cs_path)['zslice']
            if zslice > 0:
                rows = [
                    dict(row, zslice=zslice, subdir='TIF_OVR')
                    for row in self.dc_mip
                ]
                req['channels'] = {
                    row['channel']:
                    ExternalInputFile(path=self.dcrow_to_path(row))
                    for row in rows
                }

        return req

    def output(self):
        cs_path = self.input()['center_surround'].path
        row = dict(self.parse_path(self.dc_mip[0], cs_path),
                   subdir='PROPS_MIDDLE',
                   ext='csv')

        return luigi.LocalTarget(self.dcrow_to_path(row))

    def run(self):
        channels = {
            ch: imread(t.path, img_num=0)
            for ch, t in self.input().get('channels', {}).items()
        }

        # build pairs of channels to compute correlation features
        for cha, chb in itertools.combinations(list(channels.keys()), 2):
            channels['ch{}.ch{}'.format(cha,
                                        chb)] = (channels[cha], channels[chb])

        labels = {}
        labels['center_surround'] = imread(
            self.input()['center_surround'].path)
        labels['spx'] = imread(self.input()['spx'].path)
        labels['obj_middle'] = labels['center_surround'] > 0

        feature_extractor = self._build_feature_extractor()
        props = feature_extractor(labels, channels)

        # add middle z plane to feature list
        cs_path = self.input()['center_surround'].path
        row = dict(self.parse_path(self.dc_mip[0], cs_path),
                   subdir='PROPS_MIDDLE',
                   ext='csv')
        props = props.append(
            {
                'channel': row['channel'],
                'region': 'center_surround',
                'object_id': 1,
                'feature_name': 'z_slice',
                'feature_value': float(row['zslice'])
            },
            ignore_index=True)

        if self.ch_to_stain:
            props = self.map_ch_to_stain(props)

        props['platedir'] = self.dc_mip[0]['platedir']
        props['plate_row'] = self.dc_mip[0]['plate_row']
        props['plate_column'] = self.dc_mip[0]['plate_column']

        self.output().makedirs()
        props.to_csv(self.output().path, index=False)

    def map_ch_to_stain(self, props):
        lut = {ch: ch for ch in props.channel.unique()}
        stain_lut = {
            row['channel']: row.get('stain', row['channel'])
            for row in self.dc_mip
        }

        for (cha, cha_stain), (chb, chb_stain) in itertools.combinations(
                list(stain_lut.items()), 2):
            stain_lut['ch{}.ch{}'.format(cha, chb)] = '{}.{}'.format(
                cha_stain, chb_stain)

        lut.update(stain_lut)

        props.channel = props.channel.map(lut)

        return props

    @staticmethod
    def _build_feature_extractor():
        return GlobalFeatureExtractor(extractors=[
            IntensityFeatureExtractor(features=['mean'],
                                      label_targets=['center_surround', 'spx'],
                                      channel_targets=[1, 2, 3, 4]),
            SKRegionPropFeatureExtractor(features=['area'],
                                         label_targets=['center_surround'],
                                         channel_targets=None,
                                         physical_coords=False),
            GLCMFeatureExtractor(distances=(10, 100, 200),
                                 n_levels=256,
                                 label_targets=['obj_middle'],
                                 channel_targets=[1, 2, 3, 4]),
            CorrelationFeatureExtractor(label_targets=['obj_middle'],
                                        channel_targets=[
                                            'ch1.ch2', 'ch1.ch3', 'ch1.ch4',
                                            'ch2.ch3', 'ch2.ch4', 'ch3.ch4'
                                        ]),
            SPXFeatureExtractor(label_targets=['spx']),
        ])


def clean_feature_names(name):
    '''Clean up feature name obtained by flattening column multi-index'''

    return (name.replace('img001', 'img').replace('obj001', 'obj').replace(
        'grid001', 'grid').replace('skel001', 'skel').replace(
            'shading001',
            'shading').replace('debris001',
                               'debris').replace('na001_',
                                                 '').replace('chna_', ''))


class AggreatePropsTask(luigi.Task):

    outdir = luigi.Parameter()
    datadir = luigi.Parameter(description='base path of the experiment')
    props_extra_index = luigi.ListParameter(
        description=
        'extends props index with optional fields available in data collection. e.g. timepoint or staining'
    )

    def requires(self):
        return ParseGoidCollectionTask(datadir=self.datadir,
                                       outdir=self.outdir,
                                       filename='output_collection.h5')

    def output(self):
        return {
            'all_multi':
            luigi.LocalTarget(os.path.join(self.outdir,
                                           'aggregated_props.h5')),
            'all_noindex':
            luigi.LocalTarget(os.path.join(self.outdir,
                                           'aggregated_props.csv')),
            'l100':
            luigi.LocalTarget(os.path.join(self.outdir, 'l100seg_props.h5')),
            'spx':
            luigi.LocalTarget(os.path.join(self.outdir, 'spx_props.h5')),
            'flat_props':
            luigi.LocalTarget(os.path.join(self.outdir, 'flat_props.h5')),
        }

    def run(self):
        os.makedirs(self.outdir, exist_ok=True)

        df = pd.read_hdf(self.input().path)

        all_props = df.dc[:, ['PROPS_MIDDLE', 'PROPS_MIP', 'PROPS']].dc.read()
        all_props = pd.concat(all_props, sort=True)
        all_props = all_props.astype({
            'channel': str,
            'feature_name': str,
            'feature_value': 'float32',
            'object_id': 'int32',
            'plate_column': 'int32',
            'plate_row': str,
            'platedir': str,
            'region': str
        })

        # extra index lookup table (e.g. timepoint, condition)
        extra_index = list(self.props_extra_index)
        lut = df.reset_index()[['platedir', 'plate_column', 'plate_row'] +
                               extra_index].drop_duplicates().set_index(
                                   ['platedir', 'plate_row', 'plate_column'])
        extra_vals = lut.loc[pd.MultiIndex.from_frame(
            all_props[['platedir', 'plate_row', 'plate_column']])]
        all_props = pd.concat([
            all_props.reset_index(drop=True),
            extra_vals.reset_index(drop=True)
        ],
                              axis=1)
        all_props.to_csv(self.output()['all_noindex'].path, index=False)

        # pivot table to save memory and loading time
        all_props = all_props.pivot_table(
            index=['platedir', 'plate_row', 'plate_column'] + extra_index,
            columns=['region', 'object_id', 'channel', 'feature_name'],
            values='feature_value')

        # save all props as hdf5
        all_props.to_hdf(self.output()['all_multi'].path, 'props', complevel=5)

        # save 100seg features
        l100_props = all_props.loc(axis=1)[['l100seg']].stack('object_id')
        l100_props.columns = [
            clean_feature_names('{}_ch{}_{}'.format(*idx))
            for idx in l100_props.columns.to_flat_index()
        ]
        l100_props.sort_index(axis=1, inplace=True)
        l100_props.to_hdf(self.output()['l100'].path, 'props')

        # save super pixels features
        try:
            spx_props = all_props.loc(
                axis=1)[['spx', 'spxmip'], :].stack('object_id')
        except:
            # middle slice/spx props probably not computed
            spx_props = pd.DataFrame()
        spx_props.columns = [
            clean_feature_names('{}_ch{}_{}'.format(*idx))
            for idx in spx_props.columns.to_flat_index()
        ]
        spx_props.sort_index(axis=1, inplace=True)
        spx_props.to_hdf(self.output()['spx'].path, 'props')

        # save rest of props with a flat column index
        regions = list(
            set(all_props.columns.levels[0].tolist()) -
            {'spx', 'spxmip', 'l100seg'})
        flat_props = all_props.loc(axis=1)[regions]
        flat_props.columns = [
            clean_feature_names('{}{:03d}_ch{}_{}'.format(*idx))
            for idx in flat_props.columns.to_flat_index()
        ]
        flat_props.sort_index(axis=1, inplace=True)
        flat_props.to_hdf(self.output()['flat_props'].path, 'props')
