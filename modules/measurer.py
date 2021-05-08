from time import time, sleep
import pickle


class TimeMeasurer:
    def __init__(self, save_path=''):
        self.path = save_path
        self.current_block = None
        self.times_global = []
        self.times_local = dict()

        self.time_sample_global = 0
        self.time_sample_local = 0

    def begin_sample(self, name):
        self.time_sample_global = time()
        self.times_local = {'experiment_name': name}

    def end_sample(self):
        self.times_local.update({'total_time': time() - self.time_sample_global})
        self.times_global.append(self.times_local)

    def start_measure_local(self, block_name):
        self.current_block = block_name
        self.time_sample_local = time()

    def finish_measure_local(self,):
        self.times_local.update({self.current_block: time() - self.time_sample_local})

    def finish_logging_time(self):
        with open(f'{self.path}time_profiling.pickle', 'wb') as handle:
            pickle.dump(self.times_global, handle, protocol=pickle.HIGHEST_PROTOCOL)