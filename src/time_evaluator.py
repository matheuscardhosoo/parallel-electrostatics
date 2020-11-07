import json

import matplotlib.pyplot as plt

from cuda.cuda_helper import cuda_args
from cuda.parallel_electric_field_wrapper import ParallelElectricFieldWrapper
from wrapper.electric_field_wrapper import ElectricFieldWrapper


class TimeEvaluator():

    def __init__(self, config_option, charges):
        self._electric_field = ElectricFieldWrapper(config_option, charges)
        self._parallel_electric_field = ParallelElectricFieldWrapper(config_option, charges)
        # To discard the compilation time.
        self._parallel_electric_field.time_it(sequential_time=0)

    def process(self, times, max_number_of_cores=1024):
        report = {}
        report.update(self._process_sequential_execution(times))
        report.update(self._process_parallel_execution(
            times, report['sequential_time'], max_number_of_cores))
        self._save_as_json_file(report)
        self.plot(report)

        return report

    def _process_sequential_execution(self, times):
        partial_report = {}
        partial_report['sequential_time'] = 0
        partial_report['sequential_samples'] = []
        for _ in range(times):
            sample = self._electric_field.time_it()
            partial_report['sequential_time'] += sample['total_time']/times
            partial_report['sequential_samples'].append(sample)
        return partial_report

    def _process_parallel_execution(self, times, sequential_time, max_number_of_cores=1024):
        partial_report = {}
        partial_report['parallel_time'] = []
        partial_report['parallel_speedup'] = []
        partial_report['parallel_efficiency'] = []
        partial_report['parallel_samples'] = []
        n = 0
        number_of_cores = 2**n
        while number_of_cores <= max_number_of_cores:
            time = 0
            speedup = 0
            efficiency = 0
            samples = []
            for _ in range(times):
                self._parallel_electric_field.number_of_cores = number_of_cores
                sample = self._parallel_electric_field.time_it(sequential_time=sequential_time)
                time += sample['total_time']/times
                speedup += sample['speedup']/times
                efficiency += sample['efficiency']/times
                samples.append(sample)
            partial_report['parallel_time'].append(time)
            partial_report['parallel_speedup'].append(speedup)
            partial_report['parallel_efficiency'].append(efficiency)
            partial_report['parallel_samples'].append(samples)
            n += 1
            number_of_cores = 2**n
        return partial_report

    @staticmethod
    def _save_as_json_file(report):
        f = open("tests.json", "w")
        json.dump(report, f)
        f.close()

    @staticmethod
    def plot(report):
        print(report['sequential_time'])
        x = [f'A{n}' for n in range(11)]
        TimeEvaluator._generic_plot(
            ['S'] + x, [report['sequential_time']] + report['parallel_time'],
            'Tempo de execução X Threads per block', 'Tempo', 'Grid X Blocks', 'time_plot.png')
        TimeEvaluator._generic_plot(x, report['parallel_speedup'], 'Speedup X Threads per block',
                                    'Tempo', 'Grid X Blocks', 'speedup_plot.png')
        TimeEvaluator._generic_plot(
            x, report['parallel_efficiency'],
            'Eficiência X Threads per block', 'Tempo', 'Grid X Blocks', 'efficiency_plot.png')

    @staticmethod
    def _generic_plot(x, y, title, y_label, x_label, file_name):
        plt.figure()
        plt.scatter(x, y)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid(True)
        plt.savefig(file_name)
