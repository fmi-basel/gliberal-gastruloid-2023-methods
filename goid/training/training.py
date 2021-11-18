import luigi
import os
import logging
import abc
import numpy as np
from multiprocessing import Lock, Manager

from .augmentations import pad_to_minsize, AugmentationMixin
from .model import BuildModelMixin


def plot_instance_dataset(path, tf_dataset, n_samples=10):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import inter_view.color

    def create_single_page(raw, y_true):

        zslice = None
        splots = {'raw': raw.numpy().squeeze()}

        for key, val in y_true.items():
            if key in ['embeddings', 'binary_seg']:
                splots[key] = val.numpy().squeeze().astype(np.int16)

            elif key == 'semantic_class':
                one_hot_classes = val.numpy()
                classes = np.argmax(one_hot_classes,
                                    axis=-1) + one_hot_classes.sum(axis=-1) - 1
                splots[key] = classes.astype(np.int16)

            elif key == 'separator':
                splots[key] = val.numpy().squeeze().astype(np.float32)

                # special case, raw input for separator prediction has 4 channels, only show the mean
                splots['raw'] = splots['raw'].mean(axis=-1)

            else:
                raise KeyError(
                    'unrecognized dataset key: {}, expected ["embeddings", "binary_seg", "semantic_class"]'
                    .format(key))

        if splots['raw'].ndim >= 3:
            zslice = splots['raw'].shape[0] // 2

            for key in splots.keys():
                splots[key] = splots[key][zslice]

        fig, axs = plt.subplots(1, len(splots), figsize=(len(splots) * 6, 6))
        for ax, (key, val) in zip(axs.flat, splots.items()):
            if key == 'raw':
                ax.imshow(val, cmap='Greys_r')
            elif key == 'separator':
                ax.imshow(val, cmap='viridis')
            else:
                ax.imshow(val,
                          cmap='blk_glasbey_hv',
                          interpolation='nearest',
                          vmin=-1,
                          vmax=254)

            ax.set_xlabel(key)
            ax.set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
                val.min(), val.max()))

        plt.tight_layout()

    with PdfPages(path) as pdf:
        for raw, y_true in tf_dataset.unbatch().take(n_samples):
            create_single_page(raw, y_true)
            pdf.savefig(bbox_inches='tight')
            plt.close()


class TrainingMixin:
    '''Adds for default training parameters as luigi task parameters
    and provides the common_callbacks() method.

    '''

    # yapf: disable
    traindir = luigi.Parameter(description='training base directory.')
    train_batch_size = luigi.IntParameter(description='training batch size.')
    valid_batch_size = luigi.IntParameter(description='validation batch size.')
    epochs = luigi.IntParameter(description='number of training epochs.')
    lr_min = luigi.FloatParameter(description='minimum learning rate.')
    lr_max = luigi.FloatParameter(description='maximum learning rate.')
    epoch_to_restart_growth = luigi.FloatParameter(2., description='growth factor of the number of epochs after each restart')
    n_restarts = luigi.IntParameter(1, description='number of restarts for the cosine annealing scheduler')
    patience = luigi.IntParameter(200, description='number of epochs without improvment to wait before stopping')
    patch_size = luigi.TupleParameter(description='training patch size')
    resume_weights = luigi.OptionalParameter(None, description='path to weigths used to resume training')
    plot_dataset = luigi.BoolParameter(True, description='plot samples from from the training set to pdf at the beginning of training', parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    # yapf: enable

    def common_callbacks(self, output_folder):
        '''creates several keras callbacks to be used in model.fit or
        model.fit_generator.
    
        '''

        from dlutils.training.callbacks import ModelConfigSaver
        from tensorflow.keras.callbacks import ModelCheckpoint
        from tensorflow.keras.callbacks import TensorBoard
        from tensorflow.keras.callbacks import LearningRateScheduler
        from tensorflow.keras.callbacks import TerminateOnNaN
        from tensorflow.keras.callbacks import EarlyStopping

        from dlutils.training.scheduler import CosineAnnealingSchedule

        n_restarts_factor = sum(self.epoch_to_restart_growth**x
                                for x in range(self.n_restarts))

        epochs_to_restart = (self.epochs + 1) / n_restarts_factor
        if epochs_to_restart < 1:
            raise ValueError(
                'Initial epoch_to_restart ({}) < 1. Decrease n_restarts ({}) or epoch_to_restart_growth ({})'
                .format(epochs_to_restart, self.n_restarts,
                        self.epoch_to_restart_growth))

        epochs_to_restart = int(np.ceil(epochs_to_restart))

        callbacks = []

        if self.lr_max != self.lr_min:
            callbacks.append(
                LearningRateScheduler(
                    CosineAnnealingSchedule(
                        lr_max=self.lr_max,
                        lr_min=self.lr_min,
                        epoch_max=epochs_to_restart,
                        epoch_max_growth=self.epoch_to_restart_growth,
                        reset_decay=1.)))

        callbacks.extend([
            TerminateOnNaN(),
            TensorBoard(os.path.join(output_folder, 'tensorboard-logs'),
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        histogram_freq=0),
            ModelCheckpoint(os.path.join(output_folder, 'weights_best.h5'),
                            save_best_only=True,
                            save_weights_only=True),
            ModelCheckpoint(os.path.join(output_folder, 'weights_latest.h5'),
                            save_best_only=False,
                            save_weights_only=True),
        ])

        if self.patience is not None and self.patience >= 1:
            callbacks.append(EarlyStopping(patience=self.patience))

        return callbacks

    @abc.abstractmethod
    def split_samples(self, data):
        '''Function to apply after data augmentation to get training input/output pair'''

        pass


# NOTE alternatively check actual usage with nvidia-smi, (e.g. nvgpu python wrapper)
# (tensorflow would have to be initialized and allocate memory before releasing lock)
class GPUTask(luigi.Task):

    _lock = Lock()
    _used_gpus = Manager().dict()

    resources = {'gpu_task': 1}

    def _acquire_gpu(self):

        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')

        self.gpu_idx = -1
        gpu_device = []
        with GPUTask._lock:
            for idx, device in enumerate(physical_devices):
                if not GPUTask._used_gpus.get(idx, False):
                    GPUTask._used_gpus[idx] = True
                    self.gpu_idx = idx
                    gpu_device = [device]
                    break

        if self.gpu_idx < 0:
            raise RuntimeError(
                'no available GPU found. Check that luigi resources "gpu_task" matches the number of physical GPU'
            )
            # TODO try get "gpu_task" from luigi config and compare to number of available gpu
            # log warning instead and attempt to run on cpu?

        # print('Placing on GPU {}'.format(self.gpu_idx))
        tf.config.set_visible_devices(gpu_device, 'GPU')

        # to be able to estimate VRAM usage with nvidia-smi
        tf.config.experimental.set_memory_growth(
            physical_devices[self.gpu_idx], True)

    def _release_gpu(self):
        if hasattr(self, 'gpu_idx'):
            with GPUTask._lock:
                GPUTask._used_gpus[self.gpu_idx] = False

    def run(self):
        self._acquire_gpu()
        self.gpu_run()
        self._release_gpu()

    def on_failure(self, exception):
        self._release_gpu()
        return super().on_failure(exception)

    @abc.abstractmethod
    def gpu_run(self):
        pass


class JaccardLossParams(luigi.Config):
    jaccard_hinge = luigi.FloatParameter(
        0.3, description='lower hinge for binary Jaccard loss')
    jaccard_eps = luigi.FloatParameter(
        1., description='epsilon/smoothing parameter for binary Jaccard loss')


class ModelFittingBaseTask(GPUTask, BuildModelMixin, AugmentationMixin,
                           TrainingMixin):
    '''

    '''
    foreground_weight = luigi.FloatParameter(default=1)
    draw_dataset = luigi.BoolParameter(
        True,
        description=
        'plot samples from from the training set to pdf at the beginning of training',
        parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.traindir, self.model_name))

    @abc.abstractmethod
    def get_training_losses(self):
        '''Function to apply after data augmentation to get training input/output pair'''

        pass

    @abc.abstractmethod
    def _get_parser_fun(self):
        '''Returns a record parser function'''
        pass

    def gpu_run(self):

        import tensorflow as tf
        from dlutils.dataset.dataset import create_dataset

        with self.output().temporary_path() as model_dir:
            logger = logging.getLogger('luigi-interface')
            logger.info('Starting training model: {}'.format(model_dir))

            augmentations = self.get_augmentations() + [self.split_samples]

            trainset = create_dataset(self.input()['train'].path,
                                      batch_size=self.train_batch_size,
                                      parser_fn=self._get_parser_fun(),
                                      transforms=augmentations,
                                      shuffle_buffer=500,
                                      shuffle=True,
                                      drop_remainder=False,
                                      cache_after_parse=False,
                                      patch_size=self.patch_size)

            validset = create_dataset(self.input()['valid'].path,
                                      batch_size=self.valid_batch_size,
                                      parser_fn=self._get_parser_fun(),
                                      transforms=[self.split_samples],
                                      drop_remainder=False,
                                      cache_after_parse=False,
                                      patch_size=self.patch_size)

            model = self.construct_model()
            model.save(model_dir)

            if self.plot_dataset:
                logger = logging.getLogger('luigi-interface')
                logger.info('plotting nuclei training examples to pdf')
                plot_instance_dataset(
                    os.path.join(model_dir, 'training_samples.pdf'), trainset,
                    100)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_max),
                loss=self.get_training_losses(),
                metrics=None)

            if self.resume_weights:
                model.load_weights(self.resume_weights)

            history = model.fit(trainset,
                                validation_data=validset,
                                epochs=self.epochs,
                                callbacks=self.common_callbacks(model_dir))


class InferenceModelExportBaseTask(luigi.Task):
    def run(self):
        import tensorflow as tf

        model = tf.keras.models.load_model(self.input().path, compile=False)
        model.load_weights(os.path.join(self.input().path,
                                        'weights_latest.h5'))

        with self.output().temporary_path() as temp_output_path:
            tf.saved_model.save(model,
                                export_dir=temp_output_path,
                                signatures=self.serve_signatures(model))

    def serve_signatures(self, model):
        return {}

    def output(self):
        return luigi.LocalTarget(self.input().path + '_inference')
