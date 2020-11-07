"""Drawer Class to normalize the plot generation."""
from matplotlib.pyplot import cm, contourf, figure, gca, savefig, subplots_adjust, xlim, ylim
from numpy import arange, clip


class Drawer:
    def __init__(self, calculate_function, config_option, charges):
        self._calculate_function = calculate_function
        self._config_option = config_option
        self._charges = charges

    def draw(self, n_min, n_max, n_step, **kwargs):
        """
        Draw the matrix with values.
        Arguments:
            n_min: superior limit for electricfield values.
            n_max: inferior limit for electricfield values.
            n_step: granularity between limits.
        """
        figure()
        result, x, y = self._calculate_function(**kwargs)
        self._plot_field(result, x, y, n_min, n_max, n_step)
        self._plot_charges()
        self._adjust_plot()
        savefig('image.png')

    def _plot_charges(self):
        for charge in self._charges:
            charge.plot()

    def _adjust_plot(self):
        ax = gca()
        ax.set_xticks([])
        ax.set_yticks([])
        xlim(self._config_option.fixed_x_min, self._config_option.fixed_x_max)
        ylim(self._config_option.fixed_y_min, self._config_option.fixed_y_max)
        subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    @staticmethod
    def _plot_field(result, x, y, n_min, n_max, n_step):
        levels = arange(n_min, n_max + n_step, n_step)
        color_map = cm.get_cmap('plasma')
        contourf(x, y, clip(result, n_min, n_max), 10, cmap=color_map, levels=levels, extend='both')
