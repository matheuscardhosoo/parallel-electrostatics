"""
Parallel implementation of ElectricFieldWrapper.
"""
import time
from math import sqrt, log10
from numba import vectorize, cuda

import numpy as np
import electrostatics as es

from cuda.cuda_helper import cuda_args
from wrapper.electric_field_wrapper import ElectricFieldWrapper


class ParallelElectricFieldWrapper(ElectricFieldWrapper):
    """
    Parallel implementation of ElectricFieldWrapper.
    Args:
        config_option(object): ConfigOption object with the configuration values.
        charges(list): electric charges that generate the Electric Field.
    """

    def __init__(self, config_option, charges, number_of_cores=1024):
        super().__init__(config_option, charges)
        self._charges_array = self._charges_to_array(charges)
        self.number_of_cores = number_of_cores

    def time_it(self, **kwargs):
        """
        Calculate the matrix with Electric Field values.
        Arguments:
            kwargs(dict): 
        Returns:
            float: total execution time.
            list: execution time of each step.
        """
        sequential_algorithm_time = kwargs['sequential_time']

        start_time = time.time()
        partial, result, x, y = self._create_work_space()
        sequential_time_0 = time.time() - start_time

        sequential_time_1, parallel_time_1 = self._calculate_charges_electric_field_vectors(
            partial, x, y, self._charges)
        sequential_time_2, parallel_time_2 = self._calculate_electric_field_magnitudes(
            partial, result)

        sequential_time = sequential_time_0 + sequential_time_1 + sequential_time_2
        parallel_time = parallel_time_1 + parallel_time_2
        speedup = sequential_algorithm_time/parallel_time
        return {
            'total_time': sequential_time + parallel_time,
            'speedup': speedup,
            'efficiency': speedup/self.number_of_cores,
            'sequential_time': sequential_time,
            'sequential_times': [
                sequential_time_0,
                sequential_time_1,
                sequential_time_2
            ],
            'parallel_time': parallel_time,
            'parallel_times': [
                parallel_time_1,
                parallel_time_2
            ]
        }

    def _calculate_charges_electric_field_vectors(self, partial, x, y, charges):
        start_time = time.time()
        grid, block = cuda_args(partial, 3, self.number_of_cores)
        device_partial = cuda.to_device(partial)
        device_x = cuda.to_device(x)
        device_y = cuda.to_device(y)
        device_charges = cuda.to_device(self._charges_array)
        sequential_time = time.time() - start_time

        start_time = time.time()
        _calculate_charges_electric_field_vectors[grid, block](
            device_partial, device_x, device_y, device_charges)
        parallel_time = time.time() - start_time

        start_time = time.time()
        device_partial.copy_to_host(partial)
        sequential_time += time.time() - start_time
        return sequential_time, parallel_time

    def _calculate_electric_field_magnitudes(self, partial, result):
        start_time = time.time()
        grid, block = cuda_args(result, 2, self.number_of_cores)
        device_partial = cuda.to_device(partial)
        device_result = cuda.to_device(result)
        sequential_time = time.time() - start_time

        start_time = time.time()
        _calculate_electric_field_magnitudes[grid, block](device_partial, device_result)
        parallel_time = time.time() - start_time

        start_time = time.time()
        device_result.copy_to_host(result)
        sequential_time += time.time() - start_time
        return sequential_time, parallel_time

    @staticmethod
    def _charges_to_array(charges):
        chagres_array = np.zeros((len(charges), 7), dtype=np.float32)
        for i, charge in enumerate(charges):
            if isinstance(charge, es.PointCharge) or isinstance(charge, es.PointChargeFlatland):
                chagres_array[i][0] = 0 if isinstance(charge, es.PointChargeFlatland) else 1
                chagres_array[i][1] = charge.q
                chagres_array[i][2] = charge.x[0]
                chagres_array[i][3] = charge.x[1]
                chagres_array[i][4], chagres_array[i][5], chagres_array[i][6] = 0, 0, 0
            if isinstance(charge, es.LineCharge):
                chagres_array[i][0] = 2
                chagres_array[i][1] = charge.q
                chagres_array[i][2] = charge.x1[0]
                chagres_array[i][3] = charge.x1[1]
                chagres_array[i][4] = charge.x2[0]
                chagres_array[i][5] = charge.x2[1]
                chagres_array[i][6] = charge.lam
        return chagres_array


@cuda.jit('void(float32[:,:,:,:], float32[:,:], float32[:,:], float32[:,:])')
def _calculate_charges_electric_field_vectors(partial, x, y, charges):
    i, j, k = cuda.grid(3)
    if i >= partial.shape[0] or j >= partial.shape[1] or k >= partial.shape[2]:
        return

    charge_type, q, x0, y0 = charges[k][0], charges[k][1], charges[k][2], charges[k][3]

    if charge_type == 0 or charge_type == 1:
        dx = x[i][j] - x0
        dy = y[i][j] - y0
        b = dx**2 + dy**2 if charge_type == 0 else (dx**2 + dy**2)**1.5
        partial[i][j][k][0] = q * dx / b
        partial[i][j][k][1] = q * dy / b
    if charge_type == 2:
        x1 = charges[k][4]
        y1 = charges[k][5]
        lam = charges[k][6]


@cuda.jit('void(float32[:,:,:,:], float32[:,:])')
def _calculate_electric_field_magnitudes(partial, result):
    i, j = cuda.grid(2)
    if i >= result.shape[0] or j >= result.shape[1]:
        return

    field_vector_0, field_vector_1 = 0, 0
    for field_vector in partial[i][j]:
        field_vector_0 += field_vector[0]
        field_vector_1 += field_vector[1]
    result[i][j] = log10(sqrt(field_vector_0**2 + field_vector_1**2))
