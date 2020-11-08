"""Unit test for SequentialElectricField."""
import unittest

from electrostatics import LineCharge, PointCharge, PointChargeFlatland
from numpy.testing import assert_array_almost_equal

from src.electric_field_wrapper import ElectricFieldWrapper
from src.sequential_electric_field import SequentialElectricField
from src.helper.config_option import ConfigOption


class TestSequentialElectricField(unittest.TestCase):
    """Unit test for SequentialElectricField."""

    @classmethod
    def setUpClass(cls):
        cls._config = ConfigOption(x_min=-40, x_max=40, x_offset=2, y_min=-30, y_max=30, y_offset=0,
                                   zoom=6, elements_between_limits=200)

    def test_with_only_flatland_point_charges_should_be_equal_to_original_results(self):
        charges = [PointChargeFlatland(2, [0, 0]),
                   PointChargeFlatland(-1, [2, 1]),
                   PointChargeFlatland(1, [4, 0])]
        original_electric_field = ElectricFieldWrapper(self._config, charges)
        original_result, _, __ = original_electric_field.calculate()
        sequential_electric_field = SequentialElectricField(self._config, charges)
        sequential_result, _, __ = sequential_electric_field.calculate()
        assert_array_almost_equal(original_result, sequential_result, decimal=5)

    def test_with_only_point_charges_should_be_equal_to_original_results(self):
        charges = [PointCharge(2, [0, 0]),
                   PointCharge(-1, [2, 1]),
                   PointCharge(1, [4, 0])]
        original_electric_field = ElectricFieldWrapper(self._config, charges)
        original_result, _, __ = original_electric_field.calculate()
        sequential_electric_field = SequentialElectricField(self._config, charges)
        sequential_result, _, __ = sequential_electric_field.calculate()
        assert_array_almost_equal(original_result, sequential_result, decimal=5)

    def test_with_only_line_charges_should_be_equal_to_original_results(self):
        charges = [LineCharge(1, [-1, -2], [-1, 2]),
                   LineCharge(-1, [1, 2], [1, -2])]
        original_electric_field = ElectricFieldWrapper(self._config, charges)
        original_result, _, __ = original_electric_field.calculate()
        sequential_electric_field = SequentialElectricField(self._config, charges)
        sequential_result, _, __ = sequential_electric_field.calculate()
        assert_array_almost_equal(original_result, sequential_result, decimal=5)

    def test_with_multiple_charge_types_should_be_equal_to_original_results(self):
        charges = [PointChargeFlatland(2, [0, 0]),
                   PointCharge(-1, [2, 1]),
                   LineCharge(1, [-1, -2], [-1, 2])]
        original_electric_field = ElectricFieldWrapper(self._config, charges)
        original_result, _, __ = original_electric_field.calculate()
        sequential_electric_field = SequentialElectricField(self._config, charges)
        sequential_result, _, __ = sequential_electric_field.calculate()
        assert_array_almost_equal(original_result, sequential_result, decimal=5)
