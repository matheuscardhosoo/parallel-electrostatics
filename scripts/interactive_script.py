# %%
# Interactive script used to validate the solutions.
import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))

from matplotlib import pyplot
from electrostatics import LineCharge, PointChargeFlatland, ElectricField, finalize_plot, init
from electric_field_wrapper import ElectricFieldWrapper
from parallel_electric_field_wrapper import ParallelElectricFieldWrapper
from helper.config_option import ConfigOption

config = ConfigOption(x_min=-40, x_max=40, x_offset=2, y_min=-30, y_max=30, y_offset=0, zoom=6,
                      elements_between_limits=200)
charges = [PointChargeFlatland(2, [0, 0]),
           PointChargeFlatland(-1, [2, 1]),
           PointChargeFlatland(1, [4, 0]),
           LineCharge(1, [-1, -2], [-1, 2])]

# %%
# Original version
XMIN, XMAX = -40, 40
YMIN, YMAX = -30, 30
ZOOM = 6
XOFFSET = 2
init(XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET)
field = ElectricField(charges)
f = pyplot.figure(figsize=(6, 4.5))
field.plot(-1.7, 0.8)
for charge in charges:
    charge.plot()
finalize_plot()
pyplot.savefig('image.png')

# %%
# Sequential version
electric_field = ElectricFieldWrapper(config, charges)
electric_field.draw(n_min=-1.7, n_max=0.8, n_step=0.2)
# electric_field.time_it()
# electric_field.calculate()

# %%
# Parallel version
electric_field = ParallelElectricFieldWrapper(config, charges, 16)
electric_field.draw(n_min=-1.7, n_max=0.8, n_step=0.2)
# electric_field.time_it()
# electric_field.calculate()

# %%
