from electrostatics import PointChargeFlatland

from time_evaluator import TimeEvaluator
from wrapper.config_option import ConfigOption

config = ConfigOption(x_min=-40, x_max=40, x_offset=2, y_min=-30, y_max=30, y_offset=0, zoom=6,
                      elements_between_limits=200)
charges = [PointChargeFlatland(2, [0, 0]),
           PointChargeFlatland(-1, [2, 0]),
           PointChargeFlatland(0, [4, 0])]

time_evaluator = TimeEvaluator(config, charges)
time_evaluator.process(times=1000, max_number_of_cores=1024)
