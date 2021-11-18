'''Luigi.Task mixin for data augmentation.

'''
import luigi


def pad_to_minsize(patch_size):
    '''
    '''
    import tensorflow as tf

    patch_size = tf.convert_to_tensor(patch_size, name='patch_size')

    @tf.function
    def _padder(input_dict):

        key0 = next(iter(input_dict.keys()))
        input_shape = tf.shape(input_dict[key0])
        tf.assert_equal(
            len(input_shape),
            len(patch_size),
            message=
            f'image.shape={input_shape} and patch_size={patch_size} are not compatible'
        )
        if tf.reduce_all(tf.greater_equal(input_shape, patch_size)):
            return input_dict

        pad = patch_size - input_shape
        pad = tf.where(pad <= 0, 0, pad)
        lhs = tf.cast(pad / 2, tf.int32)
        pad = tf.stack([lhs, pad - lhs], axis=1)

        def get_const_val(vals):
            if vals.dtype == tf.float32:
                return tf.reduce_min(vals)
            elif vals.dtype == tf.int32:
                return -1
            else:
                raise NotImplementedError(
                    'constant value for dtype {} not implemented'.format(
                        vals.dtype))

        return {
            key: tf.pad(vals,
                        pad,
                        mode='CONSTANT',
                        constant_values=get_const_val(vals))
            for key, vals in input_dict.items()
        }

    return _padder


def random_rot90(input_dict):
    import tensorflow as tf

    rot_k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    return {
        key: tf.image.rot90(val, k=rot_k)
        for key, val in input_dict.items()
    }


class AugmentationMixin:
    '''Adds parameters for augmentation functions to task.

    '''
    # augmentation params
    # yapf: disable
    augmentation_with_flips = luigi.BoolParameter(default=False)
    augmentation_with_rot90 = luigi.BoolParameter(default=False)
    augmentation_gaussian_noise_sigma = luigi.FloatParameter(default=0)
    augmentation_gaussian_noise_mu = luigi.FloatParameter(default=0)
    augmentation_offset_sigma = luigi.FloatParameter(default=0)
    augmentation_intensity_scaling = luigi.TupleParameter(default=(1., 1.))

    augmentation_angle = luigi.FloatParameter(default=0)
    augmentation_shear = luigi.FloatParameter(default=0)
    augmentation_zoom = luigi.TupleParameter(default=(1, 1))
    augmentation_zoom_x = luigi.TupleParameter(default=(1, 1))
    augmentation_zoom_y = luigi.TupleParameter(default=(1, 1))

    augmentation_warp_max_amplitude = luigi.FloatParameter(default=0)
    augmentation_warp_smoothness_factor = luigi.FloatParameter(default=4)

    # determines which entries are to be treated as inputs
    augmentation_input_keys = luigi.ListParameter(default=['image'])
    augmentation_nearest_keys = luigi.ListParameter(default=['segm'])
    augmentation_linear_keys = luigi.ListParameter(default=[])
    # yapf: enable

    def _get_dimensions(self):
        '''guess along how many dimensions to flip.

        '''
        try:
            return len(self.patch_size)
        except Exception:
            print('Patch size not known. Assuming 2D images.')
            return 2

    def _annot_to_int32(self, input_dict):
        import tensorflow as tf

        return {
            key: tf.cast(val, tf.int32)
            if val.dtype in [tf.int16, tf.uint16, tf.int8, tf.uint8] else val
            for key, val in input_dict.items()
        }

    def get_augmentations(self):
        '''get a list of augmentation functions parametrized by the given
        values.

        '''
        import tensorflow as tf
        from dlutils.dataset.augmentations import random_axis_flip
        from dlutils.dataset.augmentations import random_gaussian_noise
        from dlutils.dataset.augmentations import random_gaussian_offset
        from dlutils.dataset.augmentations import random_intensity_scaling
        from dlutils.dataset.augmentations import random_affine_transform
        from dlutils.dataset.augmentations import random_warp

        linear_keys = self.augmentation_linear_keys + self.augmentation_input_keys

        interpolations = {k: 'BILINEAR' for k in linear_keys}
        interpolations.update(
            {k: 'NEAREST'
             for k in self.augmentation_nearest_keys})

        fill_modes = {k: 'REFLECT' for k in linear_keys}
        fill_modes.update(
            {k: 'CONSTANT'
             for k in self.augmentation_nearest_keys})

        cvals = {k: 0 for k in linear_keys}
        cvals.update({k: -1 for k in self.augmentation_nearest_keys})

        augmentations = [self._annot_to_int32]
        if self.augmentation_with_flips:
            for axis in range(self._get_dimensions()):
                augmentations.append(random_axis_flip(axis, 0.5))

        if self.augmentation_with_rot90:
            augmentations.append(random_rot90)

        if (self.augmentation_angle, self.augmentation_shear,
                self.augmentation_zoom, self.augmentation_zoom_x,
                self.augmentation_zoom_y) != (0, 0, (1, 1), (1, 1), (1, 1)):

            augmentations.append(
                random_affine_transform(interpolations,
                                        angle=self.augmentation_angle,
                                        shear=self.augmentation_shear,
                                        zoom=self.augmentation_zoom,
                                        zoomx=self.augmentation_zoom_x,
                                        zoomy=self.augmentation_zoom_y,
                                        fill_mode=fill_modes,
                                        cvals=cvals))

        if self.augmentation_warp_max_amplitude > 0:
            augmentations.append(
                random_warp(self.augmentation_warp_max_amplitude,
                            interpolations,
                            fill_mode=fill_modes,
                            cvals=cvals,
                            smoothness_factor=self.
                            augmentation_warp_smoothness_factor))

        if self.augmentation_gaussian_noise_mu > 0. or \
           self.augmentation_gaussian_noise_sigma > 0.:
            augmentations.append(
                random_gaussian_noise(self.augmentation_gaussian_noise_mu,
                                      self.augmentation_gaussian_noise_sigma,
                                      self.augmentation_input_keys))

        if self.augmentation_intensity_scaling != (1., 1.):
            augmentations.append(
                random_intensity_scaling(self.augmentation_intensity_scaling,
                                         self.augmentation_input_keys))

        if abs(self.augmentation_offset_sigma) >= 1e-8:
            augmentations.append(
                random_gaussian_offset(self.augmentation_offset_sigma,
                                       self.augmentation_input_keys))

        print('\nAdded augmentations: ')
        for augmentation in augmentations:
            print('\t', augmentation)
        print()
        return augmentations
