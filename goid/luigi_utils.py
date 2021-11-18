import luigi
import os
import time
import parse
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, Manager

from skimage.io import imsave


def get_gpu_processor():
    # TODO write GPUProcessor as proper singleton or find a cleaner way
    # currently gpuprocessor must be available globaly in this module

    global gpu_processor

    if 'gpu_processor' not in globals():
        gpu_processor = GPUProcessor()

    return gpu_processor


class GPUProcessor():
    def __init__(self):
        self.queue = Queue()

        # shared dict containing status of submitted task
        manager = Manager()
        self.status = manager.dict()

        self._process = Process(target=self.process_queue,
                                args=(self.queue, self.status))
        self._process.start()

    def submit(self, gpu_job):
        # instead of hacky shared status dict,
        # would be nice to put one end of a Pipe() in the queue, along with the task and return the other
        # doesn't seem to be possible. only found this for python 2...
        # https://stackoverflow.com/questions/1446004/python-2-6-send-connection-object-over-queue-pipe-etc

        existing_task_id = self.status.get(gpu_job['task_id'], None)
        if existing_task_id is not None:
            raise ValueError(
                'gpu job with the same task id already exist: {}'.format(
                    existing_task_id))

        self.status[gpu_job['task_id']] = 'pending'
        self.queue.put(gpu_job)
        return self.status

    @staticmethod
    def process_queue(queue, status):
        # import tensorflow here?
        import tensorflow as tf

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(
            physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        models = {}

        while True:
            gpu_job = queue.get()
            if (gpu_job == 'TERMINATE'):
                break

            try:
                task_id = gpu_job['task_id']
                model_dir = gpu_job['model_dir']

                input_paths = gpu_job['input_paths']
                output_path = gpu_job['output_path']

                preproc_fun = gpu_job['preproc_fun']
                postproc_fun = gpu_job['postproc_fun']
                postproc_params = gpu_job.get('postproc_params', {})

                # build/get model
                model = models.get(model_dir, None)
                if model is None:
                    # ~model = tf.keras.models.load_model(model_dir)
                    model = tf.saved_model.load(model_dir)
                    # save for future use
                    models[model_dir] = model

                # load/preproc input and pred
                img_in, preproc_out_params = preproc_fun(input_paths)
                # ~pred = model.predict_on_batch(img_in).squeeze()
                pred = model(img_in).numpy().squeeze()
                pred_post = postproc_fun(pred, **preproc_out_params,
                                         **postproc_params)

                # save output
                out_dir = os.path.dirname(output_path)
                os.makedirs(out_dir, exist_ok=True)
                imsave(output_path, pred_post, compress=9)

                status[task_id] = 'complete'
            except Exception as e:
                # TODO propagate error trace
                status[task_id] = 'failed'
                raise

    def soft_stop(self):
        '''Stops process after current queue has completed'''
        self.queue.put('TERMINATE')


class ExternalInputFile(luigi.ExternalTask):
    path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path)


class BaseGoidTask(luigi.Task):
    dc_mip = luigi.ListParameter(
        description='MIP data collection formatted as list of dict. \
    contains info necessary to build any requirement/output path')

    def filter_dc_mip(self, ch=None, zslices=None):

        rows = self.dc_mip
        if ch:
            rows = [
                row for row in rows
                if str(ch) in [str(row['channel']),
                               row.get('stain', 'na')]
            ]

        if zslices:
            rows = [row for row in rows if row['zslice'] in zslices]

        if len(rows) <= 0:
            raise ValueError(
                'no matching mip was found for ch={}, z={} in dc_mip {}'.
                format(ch, zslices, self.dc_mip))

        return rows

    @staticmethod
    def dcrow_to_path(row):
        '''Converts a data collection row dict to path'''
        return row['pattern'].format(**row)

    @staticmethod
    def parse_path(dc_row, path):
        '''parses the path with the same format as dc_row and returns data collection row dict.
        i.e. a dictionary of the parsed fields + the pattern necessary to rebuild the path'''

        pattern = dc_row['pattern']

        # ~basedir = os.path.dirname(pattern.split('{', 1)[0])
        # ~pattern = pattern.replace(basedir, '{basedir}')

        compiled_pattern = parse.compile(
            pattern.replace('{basedir}', dc_row['basedir']))
        row = compiled_pattern.parse(path).named
        row['pattern'] = pattern
        row['basedir'] = dc_row['basedir']
        return row


class BaseGoidGPUTask(BaseGoidTask, ABC):
    '''Luigi task that manages dependencies but delegates processing to 
    a GPUProcessor instance'''
    @abstractmethod
    def _get_gpu_task(self):
        '''Returns a dictionnary defining the gpu task. e.g.:
        
        {'task_id': id(self),
         'model_dir': 'path',
         'input_paths': [self.input().path],
         'preproc_fun': preproc_fg,
         'postproc_fun': postproc_fg,
         'output_path': self.output().path,
         'postproc_params':{'min_object_size':self.min_object_size}}
        
        '''
        pass

    def run(self):
        gpu_task = self._get_gpu_task()

        status = gpu_processor.submit(gpu_task)

        # wait for status change
        while True:
            if status[gpu_task['task_id']] == 'complete':
                break
            elif status[gpu_task['task_id']] == 'failed':
                # TODO propagate error trace
                raise ValueError('gpu task failed')

            time.sleep(0.1)
