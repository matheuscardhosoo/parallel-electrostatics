"""
Class to abstract the lib configuration parameters.
"""
import json

from numpy import float32, linspace


class ConfigOption():
    """
    Abstraction for config options.
    Args:
        x_min(int): Minimum x-axis value.
        x_max(int): Maximum x-axis value.
        x_offset(int): Offset value for x-axis.
        y_min(int): Minimum y-axis value.
        y_max(int): Maximum y-axis value.
        y_offset(int): Offset value for y-axis.
        zoom(int): Zoom value.
    """

    @property
    def x_min(self):
        """Minimum x-axis value."""
        return self.__x_min

    @property
    def x_max(self):
        """Maximum x-axis value."""
        return self.__x_max

    @property
    def x_offset(self):
        """Offset value for x-axis."""
        return self.__x_offset

    @property
    def x_axis(self):
        """X axis array."""
        return linspace(
            self.fixed_x_min, self.fixed_x_max, self.elements_between_limits, dtype=float32)

    @property
    def y_min(self):
        """Minimum y-axis value."""
        return self.__y_min

    @property
    def y_max(self):
        """Maximum y-axis value."""
        return self.__y_max

    @property
    def y_offset(self):
        """Offset value for y-axis."""
        return self.__y_offset

    @property
    def y_axis(self):
        """Y axis array."""
        return linspace(
            self.fixed_y_min, self.fixed_y_max, self.elements_between_limits, dtype=float32)

    @property
    def zoom(self):
        """Zoom value."""
        return self.__zoom

    @property
    def elements_between_limits(self):
        """Number of elements between one space unit."""
        return self.__elements_between_limits

    @property
    def fixed_x_min(self):
        """x_min fixed by zoom and offset."""
        return self.x_min / self.zoom + self.x_offset

    @property
    def fixed_x_max(self):
        """x_max fixed by zoom and offset."""
        return self.x_max / self.zoom + self.x_offset

    @property
    def fixed_y_min(self):
        """y_min fixed by zoom and offset."""
        return self.y_min / self.zoom + self.y_offset

    @property
    def fixed_y_max(self):
        """y_max fixed by zoom and offset."""
        return self.y_max / self.zoom + self.y_offset

    def __init__(self, x_min=-10, x_max=10, x_offset=0, y_min=-10, y_max=10, y_offset=0, zoom=1,
                 elements_between_limits=200):
        self.__x_min = x_min
        self.__x_max = x_max
        self.__x_offset = x_offset
        self.__y_min = y_min
        self.__y_max = y_max
        self.__y_offset = y_offset
        self.__zoom = zoom
        self.__elements_between_limits = elements_between_limits

    def __str__(self):
        x_str = f'X: [{self.x_min}, {self.x_max}] + {self.x_offset}'
        y_str = f'Y: [{self.y_min}, {self.y_max}] + {self.y_offset}'
        zoom_str = f'Zoom: {self.zoom}'
        elements_between_limits_str = f'Elements between units: {self.elements_between_limits}'
        return f'\n {x_str}\n {y_str}\n {zoom_str}\n {elements_between_limits_str} \n'

    @classmethod
    def from_dict(cls, configs_as_dict):
        """Create a ConfigOption object based on a dict."""
        return cls(
            x_min=configs_as_dict.get('x_min', -10),
            x_max=configs_as_dict.get('x_max', 10),
            x_offset=configs_as_dict.get('x_offset', 0),
            y_min=configs_as_dict.get('y_min', -10),
            y_max=configs_as_dict.get('y_max', 10),
            y_offset=configs_as_dict.get('y_offset', 0),
            zoom=configs_as_dict.get('zoom', 1),
            elements_between_limits=configs_as_dict.get('elements_between_limits', 500)
        )

    @classmethod
    def from_json(cls, configs_as_json_string):
        """Create a ConfigOption object based on a json string."""
        json_as_dict = json.loads(configs_as_json_string)
        return cls.from_dict(json_as_dict)
