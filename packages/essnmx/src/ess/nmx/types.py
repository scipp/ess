import enum
from dataclasses import dataclass, field
from typing import NewType

import h5py
import scipp as sc
import scippnexus as snx


class Compression(enum.StrEnum):
    """Compression type of the output file.

    These options are written as enum for future extensibility.
    """

    NONE = 'NONE'
    GZIP = 'GZIP'
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
        cr_field = snx.create_field(group, 'crystal_rotation', self.crystal_rotation)
        cr_field.attrs['long_name'] = 'crystal rotation in Phi (XYZ)'
        snx.create_field(group, 'name', self.sample_name)
        snx.create_field(group, 'orientation_matrix', self.sample_orientation_matrix)
        snx.create_field(group, 'unit_cell', self.sample_unit_cell)


@dataclass(kw_only=True)
class NMXSourceMetadata:
    nx_class = snx.NXsource
    source_position: sc.Variable

    def __write_to_nexus_group__(self, group: h5py.Group):
        snx.create_field(group, 'name', 'European Spallation Source')
        snx.create_field(group, 'type', 'Spallation Neutron Source')
        snx.create_field(group, 'distance', sc.norm(self.source_position))
        snx.create_field(group, 'probe', 'neutron')


@dataclass(kw_only=True)
class NMXMonitorMetadata:
    nx_class = snx.NXmonitor
    monitor_histogram: sc.DataArray
    tof_bin_coord: str = field(
        default='tof',
        metadata={
            "description": "Name of the time-of-flight coordinate "
            "in the monitor histogram."
        },
    )

    def __write_to_nexus_group__(self, group: h5py.Group):
        snx.create_field(group, 'mode', 'monitor')
        snx.create_field(group, 'preset', 0.0)
        data_field = snx.create_field(group, 'data', self.monitor_histogram.data)
        data_field.attrs['signal'] = 1
        data_field.attrs['primary'] = 1
        snx.create_field(
            group, 'time_of_flight', self.monitor_histogram.coords[self.tof_bin_coord]
        )


@dataclass(kw_only=True)
class NMXDetectorMetadata:
    nx_class = snx.NXdetector

    detector_name: str
    x_pixel_size: sc.Variable
    y_pixel_size: sc.Variable
    origin_position: sc.Variable
    fast_axis: sc.Variable
    slow_axis: sc.Variable
    distance: sc.Variable
    first_pixel_position: sc.Variable
    """First pixel position with respect to the sample.

    Additional field for DIALS. It should be a 3D vector.
    """
    # TODO: Remove hardcoded values
    polar_angle: sc.Variable = field(default_factory=lambda: sc.scalar(0, unit='deg'))
    azimuthal_angle: sc.Variable = field(
        default_factory=lambda: sc.scalar(0, unit='deg')
    )

    def __write_to_nexus_group__(self, group: h5py.Group):
        snx.create_field(group, 'x_pixel_size', self.x_pixel_size)
        snx.create_field(group, 'y_pixel_size', self.y_pixel_size)
        origin = snx.create_field(group, 'origin', self.origin_position)
        origin.attrs['first_pixel_position'] = self.first_pixel_position.values
        snx.create_field(group, 'fast_axis', self.fast_axis)
        snx.create_field(group, 'slow_axis', self.slow_axis)
        snx.create_field(group, 'distance', self.distance)
        snx.create_field(group, 'polar_angle', self.polar_angle)
        snx.create_field(group, 'azimuthal_angle', self.azimuthal_angle)
