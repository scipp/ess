# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers to compute neutron time-of-flight and wavelength.
"""

import sciline as sl
import scippneutron as scn

from ess.reduce import time_of_flight

from .types import (
    Choppers,
    CoordTransformGraph,
    DetectorData,
    DetectorLtotal,
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    LtotalRange,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    RunType,
    SampleRun,
    SimulationResults,
    TimeOfFlightLookupTable,
    TimeResolution,
    TofData,
    WavelengthData,
)


def make_coordinate_transform_graph() -> CoordTransformGraph:
    """
    Create a graph of coordinate transformations to compute the wavelength from the
    time-of-flight.
    """
    graph = {
        **scn.conversion.graph.beamline.beamline(scatter=False),
        **scn.conversion.graph.tof.elastic("tof"),
    }
    return CoordTransformGraph(graph)


def extract_detector_ltotal(
    detector: DetectorData[RunType], graph: CoordTransformGraph
) -> DetectorLtotal[RunType]:
    """
    Extract Ltotal from the detector data.
    TODO: This is a temporary implementation. We should instead read the positions
    separately from the event data, so we don't need to re-load the positions every time
    new events come in while streaming live data.

    Parameters
    ----------
    detector:
        Detector data.
    graph:
        Graph of coordinate transformations.
    """
    with_ltotal = detector.transform_coords("Ltotal", graph=graph)
    return DetectorLtotal[RunType](with_ltotal.coords["Ltotal"])


def simulate_chopper_cascade(choppers: Choppers[SampleRun]) -> SimulationResults:
    """
    Simulate neutrons traveling through the chopper cascade.

    Parameters
    ----------
    choppers:
        Chopper settings.
    """
    return time_of_flight.simulate_beamline(choppers=choppers, neutrons=200_000)


def build_tof_lookup_table(
    simulation: SimulationResults,
    ltotal_range: LtotalRange,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride,
    pulse_stride_offset: PulseStrideOffset,
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    error_threshold: LookupTableRelativeErrorThreshold,
) -> TimeOfFlightLookupTable:
    """
    Build a lookup table on-the-fly to compute time-of-flight as a function of Ltotal
    and event_time_offset.

    Parameters
    ----------
    simulation:
        Simulation data of neutrons traveling through the chopper cascade.
    ltotal_range:
        Range of Ltotal values to cover.
    pulse_period:
        Period of the source pulse.
    pulse_stride:
        Stride of the source pulse.
    pulse_stride_offset:
        Offset of the source pulse stride.
    distance_resolution:
        Resolution of the distance dimension of the table.
    time_resolution:
        Resolution of the time dimension of the table.
    error_threshold:
        Relative error threshold above which values are masked.
    """
    wf = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )
    wf[time_of_flight.SimulationResults] = simulation
    wf[time_of_flight.LtotalRange] = ltotal_range
    wf[time_of_flight.PulsePeriod] = pulse_period
    wf[time_of_flight.PulseStride] = pulse_stride
    wf[time_of_flight.PulseStrideOffset] = pulse_stride_offset
    wf[time_of_flight.DistanceResolution] = distance_resolution
    wf[time_of_flight.TimeResolution] = time_resolution
    wf[time_of_flight.LookupTableRelativeErrorThreshold] = error_threshold
    return wf.compute(time_of_flight.TimeOfFlightLookupTable)


def compute_detector_time_of_flight(
    detector_data: DetectorData[RunType],
    lookup: TimeOfFlightLookupTable,
    ltotal: DetectorLtotal[RunType],
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride,
    pulse_stride_offset: PulseStrideOffset,
) -> TofData[RunType]:
    """
    Compute the time-of-flight of neutrons detected by the detector.

    Parameters
    ----------
    detector_data:
        Detector data.
    lookup:
        Lookup table to compute time-of-flight.
    ltotal:
        Total path length of neutrons from source to detector (L1 + L2).
    pulse_period:
        Period of the source pulse.
    pulse_stride:
        Stride of the source pulse.
    pulse_stride_offset:
        Offset of the source pulse stride.
    """
    wf = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )
    wf[time_of_flight.RawData] = detector_data
    wf[time_of_flight.TimeOfFlightLookupTable] = lookup
    wf[time_of_flight.Ltotal] = ltotal
    wf[time_of_flight.PulsePeriod] = pulse_period
    wf[time_of_flight.PulseStride] = pulse_stride
    wf[time_of_flight.PulseStrideOffset] = pulse_stride_offset
    return TofData[RunType](wf.compute(time_of_flight.TofData))


def compute_detector_wavelength(
    tof_data: TofData[RunType],
    graph: CoordTransformGraph,
) -> WavelengthData[RunType]:
    """
    Compute the wavelength of neutrons detected by the detector.

    Parameters
    ----------
    tof_data:
        Data with a time-of-flight coordinate.
    graph:
        Graph of coordinate transformations.
    """
    return WavelengthData[RunType](tof_data.transform_coords("wavelength", graph=graph))


providers = (
    make_coordinate_transform_graph,
    extract_detector_ltotal,
    simulate_chopper_cascade,
    build_tof_lookup_table,
    compute_detector_time_of_flight,
    compute_detector_wavelength,
)
"""Providers to compute neutron time-of-flight and wavelength."""
