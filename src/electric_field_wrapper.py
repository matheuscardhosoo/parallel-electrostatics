"""
Interface for electric field in Thomas J. Duck lib.

It condense the ElectricField inside of a single class that use the lib to generate
simulation values. 
"""
from time import time

from numpy import float32, meshgrid, zeros
from numpy import log10, sum
from electrostatics import norm

from helper.drawer import Drawer


class ElectricFieldWrapper():
    """
    Class to condense the use of ElectricField classes.
    Args:
        config_option(object): ConfigOption object with the configuration values.
        charges(list): electric charges that generate the Electric Field.
    """

    def __init__(self, config_option, charges):
        self._config_option = config_option
        self._charges = charges
        self._drawer = Drawer(self.calculate, config_option, charges)

    def draw(self, n_min, n_max, n_step, **kwargs):
        """
        Draw the matrix with Electric Field values.
        Arguments:
            n_min: superior limit for electricfield values.
            n_max: inferior limit for electricfield values.
            n_step: granularity between limits.
        """
        self._drawer.draw(n_min, n_max, n_step, **kwargs)

    def calculate(self, **kwargs):
        """
        Calculate the matrix with Electric Field values.
        Returns:
            numpy.array: matrix with calculated results.
            x: matrix with x-axis values.
            y: matrix with y-axis values.
        """
        partial, result, x, y = self._create_work_space()
        self._calculate_charges_electric_field_vectors(partial, x, y, self._charges)
        self._calculate_electric_field_magnitudes(partial, result)
        return result, x, y

    def time_it(self, **kwargs):
        """
        Calculate the matrix with Electric Field values.
        Arguments:
            kwargs(dict): 
        Returns:
            float: total execution time.
            list: execution time of each step.
        """
        start_time = time()
        partial, result, x, y = self._create_work_space()
        time_0 = time() - start_time

        start_time = time()
        self._calculate_charges_electric_field_vectors(partial, x, y, self._charges)
        time_1 = time() - start_time

        start_time = time()
        self._calculate_electric_field_magnitudes(partial, result)
        time_2 = time() - start_time

        return {
            'total_time': time_0 + time_1 + time_2,
            'sequential_time': time_0 + time_1 + time_2,
            'sequential_times': [
                time_0,
                time_1,
                time_2
            ]
        }

    def _create_work_space(self):
        x_axis = self._config_option.x_axis
        y_axis = self._config_option.y_axis
        x, y = meshgrid(x_axis, y_axis)
        result = zeros((len(y_axis), len(x_axis)), dtype=float32)
        partial = zeros((len(y_axis), len(x_axis), len(self._charges), 2), dtype=float32)
        return partial, result, x, y

    @staticmethod
    def _calculate_charges_electric_field_vectors(partial, x, y, charges):
        for i in range(partial.shape[0]):
            for j in range(partial.shape[1]):
                position = [x[i][j], y[i][j]]
                for charge_id in range(partial.shape[2]):
                    electric_field_vector = charges[charge_id].E(position)
                    partial[i][j][charge_id] = electric_field_vector

    @staticmethod
    def _calculate_electric_field_magnitudes(partial, result):
        for i in range(partial.shape[0]):
            for j in range(partial.shape[1]):
                electric_field_vector = sum(partial[i][j], axis=0)
                electric_field_magnitude = norm(electric_field_vector)
                normalized_electric_field_magnitude = log10(electric_field_magnitude)
                result[i][j] = normalized_electric_field_magnitude
