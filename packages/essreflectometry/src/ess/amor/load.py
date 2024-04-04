# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce import nexus

from ..reflectometry.types import (
    ChopperCorrectedTofEvents,
    DetectorPosition,
    FilePath,
    NeXusDetectorName,
    RawDetector,
    RawEvents,
    Run,
    SampleRotation,
)
from .types import ChopperFrequency, ChopperPhase


def load_detector(
    file_path: FilePath[Run], detector_name: NeXusDetectorName[Run]
) -> RawDetector[Run]:
    return nexus.load_detector(file_path=file_path, detector_name=detector_name)


def load_events(detector: RawDetector[Run]) -> RawEvents[Run]:
    # Recent versions of scippnexus no longer add variances for events by default, so
    # we add them here if they are missing.
    data = nexus.extract_detector_data(detector)
    if data.bins.constituents['data'].data.variances is None:
        data.bins.constituents['data'].data.variances = data.bins.constituents[
            'data'
        ].data.values
    return RawEvents[Run](data)


def compute_tof(
    events: RawEvents[Run], phase: ChopperPhase[Run], frequency: ChopperFrequency[Run]
) -> ChopperCorrectedTofEvents[Run]:
    data = events.copy(deep=False)
    dim = 'tof'
    data.bins.coords[dim] = data.bins.coords.pop('event_time_offset').to(
        unit='us', dtype='float64', copy=False
    )
    tof_unit = data.bins.coords[dim].bins.unit
    tau = sc.to_unit(1 / (2 * frequency), tof_unit)
    tof_offset = tau * phase / (180.0 * sc.units.deg)
    # Make 2 bins, one for each pulse
    edges = sc.concat([-tof_offset, tau - tof_offset, 2 * tau - tof_offset], dim)
    data = data.bin({dim: sc.to_unit(edges, tof_unit)})
    # Make one offset for each bin
    offset = sc.concat([tof_offset, tof_offset - tau], dim)
    # Apply the offset on both bins
    data.bins.coords[dim] += offset
    # Rebin to exclude second (empty) pulse range
    data = data.bin({dim: sc.concat([0.0 * sc.units.us, tau], dim)})
    return ChopperCorrectedTofEvents[Run](data)


def detector_position(
    events: RawEvents[Run], sample_rotation: SampleRotation[Run]
) -> DetectorPosition[Run]:
    position = sc.spatial.as_vectors(
        events.coords.pop('x_pixel_offset'),
        events.coords.pop('y_pixel_offset'),
        events.coords.pop('z_pixel_offset'),
    )
    # Ad-hoc correction described in
    # https://scipp.github.io/ess/instruments/amor/amor_reduction.html
    position.fields.y += position.fields.z * sc.tan(
        2.0 * sample_rotation - (0.955 * sc.units.deg)
    )
    return DetectorPosition[Run](position)


providers = (load_detector, load_events, compute_tof, detector_position)
