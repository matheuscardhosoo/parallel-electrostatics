"""Functions to wrapper the electric field calculation using electrostatic lib."""
from electrostatics import ElectricField
from electrostatics import init, XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET
from numpy import linspace, log10, meshgrid, zeros_like


class ElectricFieldWrapper(ElectricField):
    """Wrapper for electrostatic.ElectricField class."""

    def __init__(self, config, charges):
        global XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET
        XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET = config.to_original_electrostatic_lib()
        init(XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET)
        super().__init__(charges)

    def calculate(self):
        """Calculate the field magnitude."""
        x, y = meshgrid(linspace(XMIN/ZOOM+XOFFSET, XMAX/ZOOM+XOFFSET, 200),
                        linspace(YMIN/ZOOM, YMAX/ZOOM, 200))
        z = zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i][j] = log10(self.magnitude([x[i][j], y[i][j]]))
        return z, x, y
