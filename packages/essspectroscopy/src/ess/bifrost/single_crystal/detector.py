# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Bragg peak monitor handling for BIFROST."""

import scipp as sc
import scippnexus as snx
from ess.spectroscopy.types import (
    Analyzer,
    DetectorPositionOffset,
    ElasticMonitor,
    EmptyDetector,
    NeXusComponent,
    NeXusData,
    NeXusTransformation,
    RawDetector,
    RunType,
)

from ess.reduce.nexus.types import MonitorPositionOffset

from ..detector import _assign_detector_position, get_base_calibrated_detector_bifrost


def get_calibrated_bragg_peak_monitor(
    monitor: NeXusComponent[ElasticMonitor, RunType],
    *,
    transform: NeXusTransformation[ElasticMonitor, RunType],
    offset: MonitorPositionOffset[RunType, ElasticMonitor],
) -> EmptyDetector[RunType]:
    """Extract the data array corresponding to the Bragg peak monitor's signal field.

    BIFROST's Bragg peak monitor is the elastic monitor (``cbm5``), written as an
    ``NXmonitor``. It has no pixel offsets, so its position is the transformed origin
    as in :func:`ess.reduce.nexus.workflow.get_calibrated_monitor`, rather than the
    per-pixel computation used for detectors. The position is assigned with the
    BIFROST-specific broadcasting because the monitor is mounted on the detector tank
    and therefore moves with the instrument angle.

    Parameters
    ----------
    monitor:
        Loaded NeXus monitor.
    transform:
        Transformation that determines the monitor position.
    offset:
        Offset to add to the monitor position.

    Returns
    -------
    :
        Monitor with geometry coordinates.
    """
    from ess.reduce.nexus import extract_signal_data_array

    da = extract_signal_data_array(monitor)
    unit = transform.value.unit
    position = transform.value * sc.vector([0.0, 0.0, 0.0], unit=unit) + offset.to(
        unit=unit
    )
    # A monitor carries no detector_number; the Bragg peak monitor is a single pixel,
    # so name it explicitly for the pixel-grouping done by the event assembly below.
    da = da.assign_coords(detector_number=sc.index(1))
    return EmptyDetector[RunType](_assign_detector_position(da, position))


def assemble_bragg_peak_monitor_data(
    monitor: EmptyDetector[RunType],
    data: NeXusData[ElasticMonitor, RunType],
) -> RawDetector[RunType]:
    """Combine the Bragg peak monitor's geometry with its event data.

    Parameters
    ----------
    monitor:
        Monitor geometry from :func:`get_calibrated_bragg_peak_monitor`.
    data:
        Monitor event data.

    Returns
    -------
    :
        Events with geometry coordinates.
    """
    from ess.reduce.nexus.workflow import assemble_detector_data

    return RawDetector[RunType](assemble_detector_data(monitor, data))


def get_calibrated_bragg_peak_detector(
    detector: NeXusComponent[snx.NXdetector, RunType],
    analyzer: Analyzer[RunType],
    *,
    transform: NeXusTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
) -> EmptyDetector[RunType]:
    """Extract the data array corresponding to a detector's signal field.

    Simulated data contains no Bragg peak monitor, so a bank ('triplet') of the
    regular inelastic detector stands in for it. Real data uses
    :func:`get_calibrated_bragg_peak_monitor` instead.

    Parameters
    ----------
    detector:
        Loaded NeXus detector.
    analyzer:
        Loaded analyzer parameters.
    transform:
        Transformation that determines the detector position.
    offset:
        Offset to add to the detector position.

    Returns
    -------
    :
        Detector with geometry coordinates.
    """
    return get_base_calibrated_detector_bifrost(
        detector, analyzer, transform=transform, offset=offset
    )


providers = (get_calibrated_bragg_peak_monitor, assemble_bragg_peak_monitor_data)
simulation_providers = (get_calibrated_bragg_peak_detector,)
