# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for computing real neutron time-of-flight for indirect geometry."""

from collections.abc import Iterable

import sciline

from ess.reduce import time_of_flight as reduce_time_of_flight
from ess.reduce.time_of_flight.lut import (
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    LtotalRange,
    PulsePeriod,
    PulseStride,
    SimulationResults,
    TimeResolution,
)
from ess.reduce.time_of_flight.types import DetectorLtotal

from ..types import (
    DataAtSample,
    DetectorData,
    DetectorTofData,
    L1Range,
    MonitorCoordTransformGraph,
    MonitorData,
    MonitorLtotal,
    MonitorTofData,
    MonitorType,
    PulseStrideOffset,
    RunType,
    TimeOfFlightLookupTable,
)


def TofWorkflow(
    *,
    run_types: Iterable[sciline.typing.Key],
    monitor_types: Iterable[sciline.typing.Key],
) -> sciline.Pipeline:
    workflow = reduce_time_of_flight.GenericTofWorkflow(
        run_types=run_types,
        monitor_types=monitor_types,
    )
    for provider in providers:
        workflow.insert(provider)
    return workflow


def TofLookupTableWorkflow() -> sciline.Pipeline:
    workflow = reduce_time_of_flight.lut.TofLookupTableWorkflow()
    workflow.insert(compute_tof_lookup_table)
    return workflow


def compute_tof_lookup_table(
    simulation: SimulationResults,
    l1_range: L1Range,
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride,
    error_threshold: LookupTableRelativeErrorThreshold,
) -> TimeOfFlightLookupTable:
    """Compute a lookup table for time-of-flight as a function of distance and
    time-of-arrival.

    This is a wrapper around :func:`ess.reduce.time_of_flight.compute_tof_lookup_table`
    for indirect geometry spectrometers.
    """
    return reduce_time_of_flight.lut.make_tof_lookup_table(
        simulation=simulation,
        ltotal_range=LtotalRange(l1_range),
        distance_resolution=distance_resolution,
        time_resolution=time_resolution,
        pulse_period=pulse_period,
        pulse_stride=pulse_stride,
        error_threshold=error_threshold,
    )


def detector_time_of_flight_data(
    sample_data: DataAtSample[RunType],
    lookup: TimeOfFlightLookupTable,
    pulse_stride_offset: PulseStrideOffset,
) -> DetectorTofData[RunType]:
    """
    Convert the time-of-arrival data to time-of-flight data using a lookup table.

    The output data will have a time-of-flight coordinate.

    This is a wrapper around
    :func:`ess.reduce.time_of_flight.detector_time_of_flight_data`
    for indirect geometry spectrometers.
    """
    result = reduce_time_of_flight.eto_to_tof.detector_time_of_flight_data(
        detector_data=DetectorData[RunType](sample_data),
        lookup=lookup,
        ltotal=DetectorLtotal(sample_data.coords['L1']),
        pulse_stride_offset=pulse_stride_offset,
    )
    # This is time-of-flight at the sample.
    result.bins.coords['sample_tof'] = result.bins.coords.pop('tof')
    del result.bins.coords['event_time_zero']
    return result


def monitor_time_of_flight_data(
    monitor_data: MonitorData[RunType, MonitorType],
    lookup: TimeOfFlightLookupTable,
    ltotal: MonitorLtotal[RunType, MonitorType],
    pulse_stride_offset: PulseStrideOffset,
) -> MonitorTofData[RunType, MonitorType]:
    """
    Convert the time-of-arrival data to time-of-flight data using a lookup table.

    The output data will have a time-of-flight coordinate.

    This is a wrapper around
    :func:`ess.reduce.time_of_flight.monitor_time_of_flight_data`
    for indirect geometry spectrometers.
    """
    result = reduce_time_of_flight.eto_to_tof.monitor_time_of_flight_data(
        monitor_data=monitor_data.rename(t='tof'),
        lookup=lookup,
        ltotal=ltotal,
        pulse_stride_offset=pulse_stride_offset,
    )
    return result


def compute_monitor_ltotal(
    monitor_data: MonitorData[RunType, MonitorType],
    coord_transform_graph: MonitorCoordTransformGraph,
) -> MonitorLtotal[RunType, MonitorType]:
    """Compute the path length from the source to the monitor."""
    return MonitorLtotal[RunType, MonitorType](
        monitor_data.transform_coords(
            'Ltotal',
            graph=coord_transform_graph,
            keep_intermediate=False,
            keep_aliases=False,
            rename_dims=False,
        ).coords['Ltotal']
    )


providers = (
    compute_monitor_ltotal,
    detector_time_of_flight_data,
    monitor_time_of_flight_data,
)
"""Providers for time-of-flight calculation on indirect geometry spectrometers.

The providers here override the default providers of
:class:`ess.reduce.time_of_flight.GenericTofWorkflow`
to customize the workflow for indirect geometry spectrometers.
"""
