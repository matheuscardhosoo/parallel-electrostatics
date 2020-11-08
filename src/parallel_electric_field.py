"""
Parallel implementation of SequentialElectricField.
"""
from math import acos, cos, fabs, log10, pi, sqrt
from time import time

from electrostatics import LineCharge, PointCharge, PointChargeFlatland
from numba import cuda
from numpy import float32, zeros

from src.sequential_electric_field import SequentialElectricField
from src.helper.cuda_helper import cuda_args


class ParallelElectricField(SequentialElectricField):
    """
    Parallel implementation of SequentialElectricField.
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

        start_time = time()
        partial, result, x, y = self._create_work_space()
        sequential_time_0 = time() - start_time

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
        start_time = time()
        grid, block = cuda_args(partial, 3, self.number_of_cores)
        device_partial = cuda.to_device(partial)
        device_x = cuda.to_device(x)
        device_y = cuda.to_device(y)
        device_charges = cuda.to_device(self._charges_array)
        sequential_time = time() - start_time

        start_time = time()
        # pylint: disable=E1136  # pylint/issues/3139
        _calculate_charges_electric_field_vectors[grid, block](
            device_partial, device_x, device_y, device_charges)
        parallel_time = time() - start_time

        start_time = time()
        device_partial.copy_to_host(partial)
        sequential_time += time() - start_time
        return sequential_time, parallel_time

    def _calculate_electric_field_magnitudes(self, partial, result):
        start_time = time()
        grid, block = cuda_args(result, 2, self.number_of_cores)
        device_partial = cuda.to_device(partial)
        device_result = cuda.to_device(result)
        sequential_time = time() - start_time

        start_time = time()
        # pylint: disable=E1136  # pylint/issues/3139
        _calculate_electric_field_magnitudes[grid, block](device_partial, device_result)
        parallel_time = time() - start_time

        start_time = time()
        device_result.copy_to_host(result)
        sequential_time += time() - start_time
        return sequential_time, parallel_time

    @staticmethod
    def _charges_to_array(charges):
        chagres_array = zeros((len(charges), 7), dtype=float32)
        for i, charge in enumerate(charges):
            if isinstance(charge, PointCharge) or isinstance(charge, PointChargeFlatland):
                chagres_array[i][0] = 0 if isinstance(charge, PointChargeFlatland) else 1
                chagres_array[i][1] = charge.q
                chagres_array[i][2] = charge.x[0]
                chagres_array[i][3] = charge.x[1]
                chagres_array[i][4], chagres_array[i][5], chagres_array[i][6] = 0, 0, 0
            if isinstance(charge, LineCharge):
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

    xp, yp = x[i][j], y[i][j]
    charge_type, q, x0, y0 = charges[k][0], charges[k][1], charges[k][2], charges[k][3]

    # PointChargeFlatland or PointCharge
    if charge_type == 0 or charge_type == 1:
        dx, dy = xp - x0, yp - y0
        b = dx**2 + dy**2 if charge_type == 0 else (dx**2 + dy**2)**1.5
        partial[i][j][k][0] = q * dx / b
        partial[i][j][k][1] = q * dy / b

    # LineCharge
    if charge_type == 2:
        x1 = charges[k][4]
        y1 = charges[k][5]
        lam = charges[k][6]

        # angle(p, v0, v1)
        dx_0p, dy_0p = x0 - xp, y0 - yp
        dx_01, dy_01 = x0 - x1, y0 - y1
        norm_0p = sqrt(dx_0p**2 + dy_0p**2)
        norm_01 = sqrt(dx_01**2 + dy_01**2)
        dot = dx_0p*dx_01 + dy_0p*dy_01
        theta_p01 = acos(dot/(norm_0p*norm_01))

        # angle(p, v1, v0)
        dx_1p, dy_1p = x1 - xp, y1 - yp
        dx_10, dy_10 = x1 - x0, y1 - y0
        norm_1p = sqrt(dx_1p**2 + dy_1p**2)
        norm_10 = sqrt(dx_10**2 + dy_10**2)
        dot = dx_1p*dx_10 + dy_1p*dy_10
        theta_p10 = pi - acos(dot/(norm_1p*norm_10))

        # point_line_distance(p, v0, v1)
        dx_p0, dy_p0 = xp - x0, yp - y0
        dx_p1, dy_p1 = xp - x1, yp - y1
        cross_p01 = dx_p0*dy_p1 - dy_p0*dx_p1
        point_line_distance_p01 = fabs(cross_p01)/norm_10

        # Calculate the parallel and perpendicular components
        # pylint: disable=invalid-name, invalid-unary-operand-type
        sign = 1 if dx_0p*dy_1p - dx_1p*dy_0p > 0 else -1
        Epara = lam*(1/norm_1p - 1/norm_0p)
        Eperp = -sign*lam*(cos(theta_p10) - cos(theta_p01))/point_line_distance_p01 \
            if point_line_distance_p01 != 0 else 0

        # Transform into the coordinate space and return
        ux_10, uy_10 = dx_10/norm_10, dy_10/norm_10
        partial[i][j][k][0] = Epara*ux_10 - Eperp*uy_10
        partial[i][j][k][1] = Eperp*ux_10 + Epara*uy_10



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
