"""
Script for generate report plots of sequential and parallel execution using the execution time
and speedup as performance metrics.
You can use this as a script executed from the root of the repository. 
"""
from electrostatics import PointChargeFlatland
from src.helper.config_option import ConfigOption
from src.report.time_evaluator import TimeEvaluator

config = ConfigOption(x_min=-40, x_max=40, x_offset=2, y_min=-30, y_max=30, y_offset=0, zoom=6,
                      elements_between_limits=200)
charges = [PointChargeFlatland(2, [0, 0]),
           PointChargeFlatland(-1, [2, 0]),
           PointChargeFlatland(0, [4, 0])]

time_evaluator = TimeEvaluator(config, charges)
time_evaluator.process(times=10, max_number_of_cores=1024)
