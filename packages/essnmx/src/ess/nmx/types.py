import enum
from dataclasses import dataclass, field
from typing import NewType

import h5py
import numpy as np
import scipp as sc
import scippnexus as snx


def _create_field(
    group: snx.typing.H5Group,
    name: str,
    data: np.ndarray | sc.Variable,
    long_name: str = '',
    **kwargs,
) -> None:
    new_field = snx.create_field(group, name, data, **kwargs)
    if long_name:
        new_field.attrs['long_name'] = long_name


class Compression(enum.StrEnum):
    """Compression type of the output file.

    These options are written as enum for future extensibility.
    """

    NONE = 'NONE'
    BITSHUFFLE_LZ4 = 'BITSHUFFLE_LZ4'


TofSimulationMinWavelength = NewType("TofSimulationMinWavelength", sc.Variable)
"""Minimum wavelength for tof simulation to calculate look up table."""

TofSimulationMaxWavelength = NewType("TofSimulationMaxWavelength", sc.Variable)
"""Maximum wavelength for tof simulation to calculate look up table."""


@dataclass(kw_only=True)
class NMXSampleMetadata:
    nx_class = snx.NXsample

    crystal_rotation: sc.Variable
    sample_position: sc.Variable
    sample_name: str
    # Temporarily hardcoding some values
    # TODO: Remove hardcoded values
    sample_orientation_matrix: sc.Variable = field(
        default_factory=lambda: sc.array(
            dims=['i', 'j'],
            values=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            unit="dimensionless",
        )
    )
    sample_unit_cell: sc.Variable = field(
        default_factory=lambda: sc.array(
            dims=['i'],
            values=[1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
            unit="dimensionless",  # TODO: Add real data,
            # a, b, c, alpha, beta, gamma
        )
    )

    def __write_to_nexus_group__(self, group: h5py.Group):
        _create_field(
            group,
            'crystal_rotation',
            self.crystal_rotation,
            long_name='crystal rotation in Phi (XYZ)',
        )
        _create_field(
            group,
            'name',
            self.sample_name
            if isinstance(self.sample_name, str)
            else self.sample_name.value,
        )
        _create_field(group, 'orientation_matrix', self.sample_orientation_matrix)
        _create_field(group, 'unit_cell', self.sample_unit_cell)


@dataclass(kw_only=True)
class NMXSourceMetadata:
    nx_class = snx.NXsource
    source_position: sc.Variable

    def __write_to_nexus_group__(self, group: h5py.Group):
        _create_field(group, 'name', 'European Spallation Source')
        _create_field(group, 'type', 'Spallation Neutron Source')
        _create_field(group, 'distance', sc.norm(self.source_position))
        _create_field(group, 'probe', 'neutron')


@dataclass(kw_only=True)
class NMXMonitorMetadata:
    monitor_histogram: sc.DataArray
    tof_bin_coord: str = field(
        default='tof',
        metadata={
            "description": "Name of the time-of-flight coordinate "
            "in the monitor histogram."
        },
    )


@dataclass(kw_only=True)
class NMXDetectorMetadata:
    detector_name: str
    x_pixel_size: sc.Variable
    y_pixel_size: sc.Variable
    origin_position: sc.Variable
    fast_axis: sc.Variable
    slow_axis: sc.Variable
    distance: sc.Variable
    # TODO: Remove hardcoded values
    polar_angle: sc.Variable = field(default_factory=lambda: sc.scalar(0, unit='deg'))
    azimuthal_angle: sc.Variable = field(
        default_factory=lambda: sc.scalar(0, unit='deg')
    )
