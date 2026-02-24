import enum
from dataclasses import dataclass, field
from typing import Literal, NewType

import h5py
import numpy as np
import scipp as sc
import scippnexus as snx
from ess.reduce.time_of_flight.types import TofLookupTable

from ._display_helper import to_datagroup


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


class ControlMode(enum.StrEnum):
    """Control mode of counting.

    Based on the NXlauetof definition of `control`(NXmonitor) field.
    """

    monitor = 'monitor'
    """Count to a preset value based on received monitor counts."""
    timer = 'timer'
    """Count to a preset value based on clock time"""


def _unit_matrix() -> sc.Variable:
    return sc.spatial.linear_transform(
        value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        unit="dimensionless",
    )


def _uniform_unit_cell_length() -> sc.Variable:
    return sc.vector([1.0, 1.0, 1.0], unit='dimensionless')


def _cube_unit_cell_angle() -> sc.Variable:
    return sc.vector([90.0, 90.0, 90.0], unit='deg')


@dataclass(kw_only=True)
class NMXSampleMetadata:
    nx_class = snx.NXsample

    crystal_rotation: sc.Variable
    name: str
    position: sc.Variable
    # Temporarily hardcoding some values
    # TODO: Remove hardcoded values
    orientation_matrix: sc.Variable = field(default_factory=_unit_matrix)
    unit_cell_length: sc.Variable = field(default_factory=_uniform_unit_cell_length)
    unit_cell_angle: sc.Variable = field(default_factory=_cube_unit_cell_angle)

    @property
    def unit_cell(self) -> sc.Variable:
        """a, b, c, alpha, beta, gamma."""

        return np.concat([self.unit_cell_length.values, self.unit_cell_angle.values])

    def __write_to_nexus_group__(self, group: h5py.Group):
        cr_field = snx.create_field(group, 'crystal_rotation', self.crystal_rotation)
        cr_field.attrs['long_name'] = 'crystal rotation in Phi (XYZ)'
        snx.create_field(group, 'name', self.name)
        snx.create_field(group, 'position', self.position)
        snx.create_field(group, 'orientation_matrix', self.orientation_matrix)
        unit_cell = snx.create_field(group, 'unit_cell', self.unit_cell)
        unit_cell.attrs['length-unit'] = str(self.unit_cell_length.unit)
        unit_cell.attrs['angle-unit'] = str(self.unit_cell_angle.unit)


@dataclass(kw_only=True)
class NMXSourceMetadata:
    nx_class = snx.NXsource
    position: sc.Variable
    name: Literal['European Spallation Source'] = "European Spallation Source"
    type: Literal['Spallation Neutron Source'] = "Spallation Neutron Source"
    probe: Literal['neutron'] = "neutron"

    @property
    def distance(self) -> sc.Variable:
        return sc.norm(self.position)

    def __write_to_nexus_group__(self, group: h5py.Group):
        snx.create_field(group, 'name', self.name)
        snx.create_field(group, 'type', self.type)
        distance = snx.create_field(group, 'distance', self.distance)
        distance.attrs['position'] = self.position.values
        snx.create_field(group, 'probe', self.probe)


def _zero_float_count() -> sc.Variable:
    return sc.scalar(0.0, unit='count')


@dataclass(kw_only=True)
class NMXMonitorMetadata:
    nx_class = snx.NXmonitor
    data: sc.DataArray
    """Monitor counts."""

    @property
    def time_of_flight(self) -> sc.Variable:
        return self.data.coords[self.tof_bin_coord]

    tof_bin_coord: str = field(
        default='tof',
        metadata={
            "description": "Name of the time-of-flight coordinate "
            "in the monitor histogram."
        },
    )
    mode: ControlMode = field(
        default=ControlMode.monitor,
        metadata={"description": "Mode of counting. One of `monitor` or `timer`."},
    )
    preset: sc.Variable = field(
        default_factory=_zero_float_count,
        metadata={"description": "Preset value of counting for the `mode`."},
    )

    def __write_to_nexus_group__(self, group: h5py.Group):
        group.attrs['axes'] = self.data.dims
        group.attrs['tof_bin_coord'] = self.tof_bin_coord
        snx.create_field(group, 'mode', str(self.mode))
        snx.create_field(group, 'preset', self.preset)
        data_field = snx.create_field(group, 'data', self.data.data)
        data_field.attrs['signal'] = 1
        data_field.attrs['primary'] = 1
        snx.create_field(group, 'time_of_flight', self.time_of_flight)


@dataclass(kw_only=True)
class NMXDetectorMetadata:
    nx_class = snx.NXdetector

    detector_name: str
    x_pixel_size: sc.Variable
    y_pixel_size: sc.Variable
    origin_position: sc.Variable
    fast_axis: sc.Variable
    """Inner most dimension if the data is sorted by detector number.

    The index of the fast axis changes fast along the detector number.

    i.e. When detector numbers grows: ``0, 1, 2, 3, 4, 5, 6, ...``
    and the size of the fast axis is ``3``,
    the fast axis index will be: ``0, 1, 2, 0, 1, 2, 0 ...``
    for each detector number.

    """
    slow_axis: sc.Variable
    """Outer most dimension if the data is sorted by detector number.

    The index of the slow axis changes slowly along the detector number.

    i.e. When detector numbers grows: ``0, 1, 2, 3, 4, 5, 6, ...``
    and the size of the fast axis is ``3``,
    the slow axis index will be: ``0, 0, 0, 1, 1, 1, 2, ...``
    for each detector number.

    """
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


@dataclass(kw_only=True)
class NMXReducedDetector:
    """Reduced Detector data and metadata container.

    In an output file, all metadata fields are stored on the same level as the `data`.
    However, in this reduced detector data container, the `data` and `metadata` are
    separated with an extra hierarchy.
    It is because the `data` needs more control how to be stored,
    i.e. compression option.
    Also, the histogram may need chunk-wise processing
    and therefore metadata may need to be written in advance so that
    the `data` can be appended to the existing `NXdetector` HDF5 Group.

    """

    data: sc.DataArray | None = None
    """3D Histogram of the detector counts or its place holder."""
    metadata: NMXDetectorMetadata
    """NMX Detector metadata."""


@dataclass(kw_only=True)
class NMXInstrument:
    nx_class = snx.NXinstrument

    detectors: sc.DataGroup[NMXReducedDetector]
    name: str = "NMX"
    source: NMXSourceMetadata


@dataclass(kw_only=True)
class NMXLauetof:
    nx_class = "NXlauetof"

    control: NMXMonitorMetadata
    definitions: Literal['NXlauetof'] = 'NXlauetof'
    instrument: NMXInstrument
    sample: NMXSampleMetadata
    lookup_table: TofLookupTable | None = None

    def to_datagroup(self) -> sc.DataGroup:
        return to_datagroup(self)
