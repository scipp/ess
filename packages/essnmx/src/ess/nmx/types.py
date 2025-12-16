import enum
from dataclasses import dataclass, field
from typing import NewType

import scipp as sc


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
    crystal_rotation: sc.Variable
    sample_position: sc.Variable
    sample_name: sc.Variable | str
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


@dataclass(kw_only=True)
class NMXSourceMetadata:
    source_position: sc.Variable


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
