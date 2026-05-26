# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Utilities for computing wavelength lookup tables.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NewType

import numpy as np
import sciline as sl
import scipp as sc
import scippnexus as snx
from scippneutron.chopper import DiskChopper
from scippneutron.tof import chopper_cascade

from ..nexus.types import Component, DiskChoppers, MonitorType, Position, RunType
from .types import (
    DetectorLtotal,
    LookupTable,
    LookupTableFilename,
    Lut,
    MonitorLtotal,
    PulseStrideOffset,
)


@dataclass
class BeamlineComponentReading:
    """
    Reading at a given position along the beamline from a time-of-flight simulation.
    The data (apart from ``distance``) should be flat lists (1d arrays) of length N
    where N is the number of neutrons, containing the properties of the neutrons in the
    simulation.

    Parameters
    ----------
    time_of_arrival:
        Time of arrival of the neutrons at the position where the events were recorded
        (1d array of size N).
    wavelength:
        Wavelength of the neutrons (1d array of size N).
    weight:
        Weight/probability of the neutrons (1d array of size N).
    distance:
        Distance from the source to the position where the events were recorded
        (single value; we assume all neutrons were recorded at the same position).
        For a ``tof`` simulation, this is just the position of the detector where the
        events are recorded. For a ``McStas`` simulation, this is the distance between
        the source and the event monitor.
    """

    time_of_arrival: sc.Variable
    wavelength: sc.Variable
    weight: sc.Variable
    distance: sc.Variable

    def __post_init__(self):
        self.speed = (sc.constants.h / sc.constants.m_n) / self.wavelength


@dataclass
class SimulationResultsBaseClass:
    """
    Results of a time-of-flight simulation used to create a lookup table.
    It should contain readings at various positions along the beamline, e.g., at
    the source and after each chopper.
    It also contains the chopper parameters used in the simulation, so it can be
    determined if this simulation is compatible with a given experiment.

    Parameters
    ----------
    readings:
        A dict of :class:`BeamlineComponentReading` objects representing the readings at
        various positions along the beamline. The keys in the dict should correspond to
        the names of the components (e.g., 'source', 'chopper1', etc.).
    choppers:
        The chopper parameters used in the simulation (if any). These are used to verify
        that the simulation is compatible with a given experiment (comparing chopper
        openings, frequencies, phases, etc.).
    """

    readings: dict[str, BeamlineComponentReading]
    choppers: dict[str, DiskChopper] | None = None


class SimulationResults(
    sl.Scope[RunType, SimulationResultsBaseClass],
    SimulationResultsBaseClass,
):
    """
    Results of a time-of-flight simulation used to create a lookup table.
    It should contain readings at various positions along the beamline, e.g., at
    the source and after each chopper.
    It also contains the chopper parameters used in the simulation, so it can be
    determined if this simulation is compatible with a given experiment.

    Parameters
    ----------
    readings:
        A dict of :class:`BeamlineComponentReading` objects representing the readings at
        various positions along the beamline. The keys in the dict should correspond to
        the names of the components (e.g., 'source', 'chopper1', etc.).
    choppers:
        The chopper parameters used in the simulation (if any). These are used to verify
        that the simulation is compatible with a given experiment (comparing chopper
        openings, frequencies, phases, etc.).
    """


NumberOfSimulatedNeutrons = NewType("NumberOfSimulatedNeutrons", int)
"""
Number of neutrons simulated in the simulation that is used to create the lookup table.
This is typically a large number, e.g., 1e6 or 1e7.
"""


class LtotalRange(
    sl.Scope[RunType, Component, tuple[sc.Variable, sc.Variable]],
    tuple[sc.Variable, sc.Variable],
):
    """
        Range (min, max) of the total length of the flight path from the source to the
        detector.
        This is used to create the lookup table to compute the neutron time-of-flight.
        Note that the resulting table will extend slightly beyond this range, as the
        supplied
    range is not necessarily a multiple of the distance resolution.

        Note also that the range of total flight paths is supplied manually to the
        workflow instead of being read from the input data, as it allows us to compute
        the expensive part of the workflow in advance (the lookup table) and does not
        need to be repeated for each run, or for new data coming in in the case of live
        data collection.
    """


DistanceResolution = NewType("DistanceResolution", sc.Variable)
"""
Step size of the distance axis in the lookup table.
Should be a single scalar value with a unit of length.
This is typically of the order of 1-10 cm.
"""

TimeResolution = NewType("TimeResolution", sc.Variable)
"""
Step size of the event_time_offset axis in the lookup table.
This is basically the 'time-of-flight' resolution of the detector.
Should be a single scalar value with a unit of time.
This is typically of the order of 0.1-0.5 ms.

Since the event_time_offset range needs to span exactly one pulse period, the final
resolution in the lookup table will be at least the supplied value here, but may be
smaller if the pulse period is not an integer multiple of the time resolution.
"""

PulsePeriod = NewType("PulsePeriod", sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""


class PulseStride(sl.Scope[RunType, int], int):
    """
    Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
    """


SimulationSeed = NewType("SimulationSeed", int | None)
"""Seed for the random number generator used in the simulation.
"""

SimulationFacility = NewType("SimulationFacility", str)
"""
Facility where the experiment is performed, e.g., 'ess'.
"""


@dataclass
class SourceBounds:
    """Time and wavelength bounds of the neutrons in the source pulse that encompass
    all possible neutrons that can be generated by the source.
    """

    time: tuple[sc.Variable, sc.Variable]
    """Time range (start, end) of the source pulse."""
    wavelength: tuple[sc.Variable, sc.Variable]
    """Wavelength range (min, max) of the neutrons in the source pulse."""


class ChopperFrameSequence(
    sl.Scope[RunType, chopper_cascade.FrameSequence], chopper_cascade.FrameSequence
):
    """
    Sequence of chopper frames used to compute the wavelength as a function of distance
    and event_time_offset in the lookup table.
    """


def _compute_mean_wavelength(
    simulation: BeamlineComponentReading,
    distance: sc.Variable,
    time_bins: sc.Variable,
    time_unit: str,
    frame_period: sc.Variable,
    time_bins_half_width: sc.Variable,
) -> sc.DataArray:
    """
    Compute the mean time-of-flight inside event_time_offset bins for a given range of
    distances.

    Parameters
    ----------
    simulation:
        Results of a time-of-flight simulation used to create a lookup table.
    distance:
        Distance where table is computed.
    time_bins:
        Bin edges for the event_time_offset axis in the lookup table.
    time_unit:
        Unit of the event_time_offset axis.
    frame_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    time_bins_half_width:
        Half width of the time bins in the event_time_offset axis.
    """
    travel_length = distance - simulation.distance.to(unit=distance.unit)
    # Compute arrival and flight times for all neutrons
    toas = simulation.time_of_arrival + (travel_length / simulation.speed).to(
        unit=time_unit, copy=False
    )

    data = sc.DataArray(
        data=simulation.weight,
        coords={"toa": toas, "wavelength": simulation.wavelength},
    )

    # Add the event_time_offset coordinate, wrapped to the frame_period
    data.coords['event_time_offset'] = data.coords['toa'] % frame_period

    # Because we staggered the mesh by half a bin width, we want the values above
    # the last bin edge to wrap around to the first bin.
    # Technically, those values should end up between -0.5*bin_width and 0, but
    # a simple modulo also works here because even if they end up between 0 and
    # 0.5*bin_width, we are (below) computing the mean between -0.5*bin_width and
    # 0.5*bin_width and it yields the same result.
    # data.coords['event_time_offset'] %= pulse_period - time_bins_half_width
    data.coords['event_time_offset'] %= frame_period - time_bins_half_width

    binned = data.bin(event_time_offset=time_bins)
    binned_sum = binned.bins.sum()
    # Weighted mean of wavelength inside each bin
    mean_wavelength = (
        binned.bins.data * binned.bins.coords["wavelength"]
    ).bins.sum() / binned_sum
    # Compute the variance of the wavelengths to track regions with large uncertainty
    variance = (
        binned.bins.data * (binned.bins.coords["wavelength"] - mean_wavelength) ** 2
    ).bins.sum() / binned_sum

    mean_wavelength.variances = variance.values
    return mean_wavelength


def make_wavelength_lut_from_simulation(
    simulation: SimulationResults[RunType],
    ltotal_range: LtotalRange[RunType, Component],
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride[RunType],
) -> LookupTable[RunType, Component]:
    """
    Compute a lookup table for wavelength as a function of distance and
    time-of-arrival.

    Parameters
    ----------
    simulation:
        Results of a time-of-flight simulation used to create a lookup table.
        The results should be a flat table with columns for time-of-arrival,
        wavelength, and weight.
    ltotal_range:
        Range of total flight path lengths from the source to the detector or monitor.
    distance_resolution:
        Resolution of the distance axis in the lookup table.
    time_resolution:
        Resolution of the time-of-arrival axis in the lookup table. Must be an integer.
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.

    Notes
    -----

    Below are some details about the binning and wrapping around frame period in the
    time dimension.

    We have some simulated ``toa`` (events) from a Tof/McStas simulation.
    Those are absolute ``toa``, unwrapped.
    First we compute the usual ``event_time_offset = toa % frame_period``.

    Now, we want to ensure periodic boundaries. If we make a bin centered around 0,
    and a bin centered around 71ms: the first bin will use events between 0 and
    ``0.5 * dt`` (where ``dt`` is the bin width).
    The last bin will use events between ``frame_period - 0.5*dt`` and
    ``frame_period + 0.5 * dt``. So when we compute the mean inside those two bins,
    they will not yield the same results.
    It is as if the first bin is missing the events it should have between
    ``-0.5 * dt`` and 0 (because of the modulo we computed above).

    To fix this, we do not make a last bin around 71ms (the bins stop at
    ``frame_period - 0.5*dt``). Instead, we compute modulo a second time,
    but this time using ``event_time_offset %= (frame_period - 0.5*dt)``.
    (we cannot directly do ``event_time_offset = toa % (frame_period - 0.5*dt)`` in a
    single step because it would introduce a gradual shift,
    as the pulse number increases).

    This second modulo effectively takes all the events that would have gone in the
    last bin (between ``frame_period - 0.5*dt`` and ``frame_period``) and puts them in
    the first bin. Instead of placing them between ``-0.5*dt`` and 0,
    it places them between 0 and ``0.5*dt``, but this does not really matter,
    because we then take the mean inside the first bin.
    Whether the events are on the left or right side of zero does not matter.

    Finally, we make a copy of the left edge, and append it to the right of the table,
    thus ensuring that the values on the right edge are strictly the same as on the
    left edge.
    """
    distance_unit = "m"
    time_unit = "us"
    res = distance_resolution.to(unit=distance_unit)
    pulse_period = pulse_period.to(unit=time_unit)
    frame_period = pulse_period * pulse_stride

    min_dist = ltotal_range[0].to(unit=distance_unit)
    max_dist = ltotal_range[1].to(unit=distance_unit)

    # We need to bin the data below, to compute the weighted mean of the wavelength.
    # This results in data with bin edges.
    # However, the 2d interpolator expects bin centers.
    # We want to give the 2d interpolator a table that covers the requested range,
    # hence we need to extend the range by at least half a resolution in each direction.
    # Then, we make the choice that the resolution in distance is the quantity that
    # should be preserved. Because the difference between min and max distance is
    # not necessarily an integer multiple of the resolution, we need to add a pad to
    # ensure that the last bin is not cut off. We want the upper edge to be higher than
    # the maximum distance, hence we pad with an additional 1.5 x resolution.
    pad = 2.0 * res
    distances = sc.arange('distance', min_dist - pad, max_dist + pad, res)

    # Create some time bins for event_time_offset.
    # We want our final table to strictly cover the range [0, frame_period].
    # However, binning the data associates mean values inside the bins to the bin
    # centers. Instead, we stagger the mesh by half a bin width so we are computing
    # values for the final mesh edges (the bilinear interpolation needs values on the
    # edges/corners).
    nbins = int(frame_period / time_resolution.to(unit=time_unit)) + 1
    time_bins = sc.linspace(
        'event_time_offset', 0.0, frame_period.value, nbins + 1, unit=pulse_period.unit
    )
    time_bins_half_width = 0.5 * (time_bins[1] - time_bins[0])
    time_bins -= time_bins_half_width

    # Sort simulation readings by reverse distance
    sorted_simulation_results = sorted(
        simulation.readings.values(), key=lambda x: x.distance.value, reverse=True
    )

    pieces = []
    # To avoid large RAM usage, and having to split the distances into chunks
    # according to which component reading to use, we simply loop over distances one
    # by one here.
    for dist in distances:
        # Find the correct simulation reading
        simulation_reading = None
        for reading in sorted_simulation_results:
            if dist.value >= reading.distance.to(unit=dist.unit).value:
                simulation_reading = reading
                break
        if simulation_reading is None:
            closest = sorted_simulation_results[-1]
            raise ValueError(
                "Building the lookup table failed: the requested position "
                f"{dist.value} {dist.unit} is before the component with the lowest "
                "distance in the simulation. The first component in the beamline "
                f"has distance {closest.distance.value} {closest.distance.unit}."
            )

        pieces.append(
            _compute_mean_wavelength(
                simulation=simulation_reading,
                distance=dist,
                time_bins=time_bins,
                time_unit=time_unit,
                frame_period=frame_period,
                time_bins_half_width=time_bins_half_width,
            )
        )

    table = sc.concat(pieces, 'distance')
    table.coords["event_time_offset"] = sc.midpoints(table.coords["event_time_offset"])

    # Copy the left edge to the right to create periodic boundary conditions
    table = sc.DataArray(
        data=sc.concat(
            [table.data, table.data['event_time_offset', 0]], dim='event_time_offset'
        ),
        coords={
            "distance": distances,
            "event_time_offset": sc.concat(
                [table.coords["event_time_offset"], frame_period],
                dim='event_time_offset',
            ),
        },
    )

    return LookupTable[RunType, Component](
        Lut(
            array=table,
            pulse_period=pulse_period,
            pulse_stride=pulse_stride,
            distance_resolution=table.coords["distance"][1]
            - table.coords["distance"][0],
            time_resolution=table.coords["event_time_offset"][1]
            - table.coords["event_time_offset"][0],
            choppers=sc.DataGroup(
                {k: sc.DataGroup(ch.as_dict()) for k, ch in simulation.choppers.items()}
            )
            if simulation.choppers is not None
            else None,
        )
    )


def _to_component_reading(component) -> BeamlineComponentReading:
    events = component.data.squeeze().flatten(to='event')
    sel = sc.full(value=True, sizes=events.sizes)
    for key in {'blocked_by_others', 'blocked_by_me'} & set(events.masks.keys()):
        sel &= ~events.masks[key]
    events = events[sel]
    # If the component is a source, use 'birth_time' as 'toa'
    toa = (
        events.coords["toa"] if "toa" in events.coords else events.coords["birth_time"]
    )
    return BeamlineComponentReading(
        time_of_arrival=toa,
        wavelength=events.coords["wavelength"],
        weight=events.data,
        distance=component.distance,
    )


def simulate_chopper_cascade_using_tof(
    choppers: DiskChoppers[RunType],
    source_position: Position[snx.NXsource, RunType],
    neutrons: NumberOfSimulatedNeutrons,
    pulse_stride: PulseStride[RunType],
    seed: SimulationSeed,
    facility: SimulationFacility,
) -> SimulationResults[RunType]:
    """
    Simulate a pulse of neutrons propagating through a chopper cascade using the
    ``tof`` package (https://scipp.github.io/tof).

    Parameters
    ----------
    choppers:
        A dict of DiskChopper objects representing the choppers in the beamline. See
        https://scipp.github.io/scippneutron/user-guide/chopper/processing-nexus-choppers.html
        for more information.
    source_position:
        A scalar variable with ``dtype=vector3`` that defines the source position.
        Must be in the same coordinate system as the choppers' axle positions.
    neutrons:
        Number of neutrons to simulate.
    pulse_stride:
        The pulse strinde; we need to simulate at least enough pulses to cover the
        requested stride.
    seed:
        Seed for the random number generator used in the simulation.
    facility:
        Facility where the experiment is performed.
    """
    import tof

    tof_choppers = []
    for name, ch in choppers.items():
        chop = tof.Chopper.from_diskchopper(ch, name=name)
        chop.distance = sc.norm(
            ch.axle_position - source_position.to(unit=ch.axle_position.unit)
        )
        tof_choppers.append(chop)

    source = tof.Source(
        facility=facility, neutrons=neutrons, pulses=pulse_stride, seed=seed
    )
    sim_readings = {"source": _to_component_reading(source)}
    if not tof_choppers:
        return SimulationResults[RunType](
            SimulationResultsBaseClass(readings=sim_readings, choppers=None)
        )
    model = tof.Model(source=source, choppers=tof_choppers)
    results = model.run()
    for name, ch in results.choppers.items():
        sim_readings[name] = _to_component_reading(ch)
    return SimulationResults[RunType](
        SimulationResultsBaseClass(readings=sim_readings, choppers=choppers)
    )


def _polygon_intersections(polygons: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    # Decompose the polygons into two 1D lines: the upper and lower bounds
    bounds = []
    for polygon in polygons:
        left = polygon[:, 0].argmin()
        right = polygon[:, 0].argmax()
        k = (right - left) % len(polygon)
        p = np.roll(polygon, -left, axis=0)

        bound1 = p[: k + 1]
        bound2 = np.concatenate((p[k:], p[:1]))[::-1]

        # In the case of an exactly vertical left or right edge of a polygon, the argmin
        # and argmax would pick one of the points, and then one bound of the polygon
        # (say the upper) would contain one of the vertical points, while the lower
        # bound would contain both. Then the np.interp would give us the one vertical
        # point from the upper bound, but it's undefined what the lower bound would
        # give us, because you could get either of the two points.
        # To fix, if the two leftmost or rightmost points have the same x value, we set
        # the y value of the first point to be the same as the second point.
        for b in (bound1, bound2):
            if b[0, 0] == b[1, 0]:
                b[0, 1] = b[1, 1]
            if b[-1, 0] == b[-2, 0]:
                b[-1, 1] = b[-2, 1]

        bounds.extend((bound1, bound2))

    # Now find intersections of the vertical lines at x with the bounds.
    y = np.vstack(
        [np.interp(x, b[:, 0], b[:, 1], left=np.nan, right=np.nan) for b in bounds]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
        )
        y_min = np.nanmin(y, axis=0)
        y_max = np.nanmax(y, axis=0)

    # Median value and spread estimate
    return 0.5 * (y_min + y_max), 0.5 * (y_max - y_min)


def _estimate_wavelength_by_polygon_centers(
    subframes: list[chopper_cascade.Subframe],
    time_edges: sc.Variable,
    time_unit: str,
    frame_period: sc.Variable,
) -> sc.DataArray:
    """
    Compute the mean wavelength inside event_time_offset bins for a given range of
    distances.

    This is done by finding the intersection of the edges of the subframe polygons
    (generated by the ``chopper_cascade`` module) with vertical lines at the specified
    time edges.
    We then take the mean of the minimum and maximum intersection points as an estimate
    of the mean wavelength in each bin. This handles the case where there are multiple
    subframes overlapping in a single time bin.

    Parameters
    ----------
    subframes:
        List of subframes to consider. These should already be propagated to the
        correct distance.
    time_edges:
        Edges of the time bins for which to compute the mean wavelength. Should be a
        1D variable with a unit of time.
    time_unit:
        Unit to use for all time quantities.
    frame_period:
        Period of the source pulses, used to handle the periodicity of the subframes.
    """

    # Here, the frame could be offset by more than one frame period (if the neutron
    # flight path is very long). So we shift the frame back enough times so that
    # the minimum time is between 0 and the frame period.
    min_time = sc.reduce([f.time.min() for f in subframes]).min()
    noffset = int(min_time.to(unit=time_unit).value / frame_period.value)

    # To handle the periodicity of the subframes, we need to consider not only the
    # original subframes, but also copies of the subframes shifted by the frame period.
    # This is because neutrons that arrive after the frame period will wrap around and
    # appear in the next pulse, which is equivalent to the original pulse but shifted
    # by the frame period.
    polygons = [
        np.stack(
            [
                (f.time.to(unit=time_unit) - (noffset + i) * frame_period).values,
                f.wavelength.values,
            ],
            axis=1,
        )
        for f in subframes
        for i in (0, 1)
    ]

    wavs, stddevs = _polygon_intersections(polygons, time_edges.values)

    return sc.array(
        dims=time_edges.dims,
        values=wavs,
        variances=stddevs**2,
        unit=subframes[0].wavelength.unit,
    )


def compute_frame_sequence(
    pulse_period: PulsePeriod,
    disk_choppers: DiskChoppers[RunType],
    source_position: Position[snx.NXsource, RunType],
    source_bounds: SourceBounds,
    pulse_stride: PulseStride[RunType],
) -> ChopperFrameSequence[RunType]:
    """
    Compute the chopper frame sequence for a given set of disk choppers and source pulse
    parameters.

    Parameters
    ----------
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    disk_choppers:
        Disk chopper parameters.
    source_position:
        Position of the neutron source.
    source_bounds:
        Time and wavelength range of the source pulse.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    """

    # The `pulse_frequency` parameter in time_offset_open and time_offset_close below
    # decides how many rotations the chopper will perform when computing the open and
    # close times. Because we want to cover a number of pulses equal to `pulse_stride`,
    # we need to set the pulse frequency to be `pulse_stride` times smaller than the
    # actual pulse frequency.
    #
    # In addition, the time_offset_open and time_offset_close below require the
    # pulse_frequency to be an integer multiple of the pulse frequency or vice versa.
    # A simple trick is to make sure that the requested pulse frequency is divided by
    # an even number. We need to rotate the chopper for long enough to cover wrapping
    # around the frame period, so we cover two pulses strides.
    frequency_for_chopper_rotation = (1.0 / pulse_period.to(unit='s')) / (
        pulse_stride * 2
    )

    chops = {
        key: chopper_cascade.Chopper(
            distance=sc.norm(
                ch.axle_position - source_position.to(unit=ch.axle_position.unit)
            ),
            time_open=ch.time_offset_open(
                pulse_frequency=frequency_for_chopper_rotation
            ),
            time_close=ch.time_offset_close(
                pulse_frequency=frequency_for_chopper_rotation
            ),
        )
        for key, ch in disk_choppers.items()
    }

    frames = chopper_cascade.FrameSequence.from_source_pulse(
        time_min=source_bounds.time[0],
        time_max=source_bounds.time[1],
        wavelength_min=source_bounds.wavelength[0],
        wavelength_max=source_bounds.wavelength[1],
        pulse_period=pulse_period,
        npulses=pulse_stride,
    )
    frames = frames.chop(chops.values())
    return ChopperFrameSequence[RunType](frames)


def make_wavelength_lut_from_polygons(
    ltotal_range: LtotalRange[RunType, Component],
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride[RunType],
    frames: ChopperFrameSequence,
) -> LookupTable[RunType, Component]:
    """
    Compute a lookup table for wavelength as a function of distance and
    time-of-arrival.

    Parameters
    ----------
    ltotal_range:
        Range of total flight path lengths from the source to the detector.
    distance_resolution:
        Resolution of the distance axis in the lookup table.
    time_resolution:
        Resolution of the time-of-arrival axis in the lookup table. Must be an integer.
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    frames:
        Chopper frame sequence used to compute the wavelength as a function of distance
        and event_time_offset in the lookup table.
    """
    distance_unit = "m"
    time_unit = "us"
    res = distance_resolution.to(unit=distance_unit)
    pulse_period = pulse_period.to(unit=time_unit)
    frame_period = pulse_period * pulse_stride

    min_dist = ltotal_range[0].to(unit=distance_unit)
    max_dist = ltotal_range[1].to(unit=distance_unit)

    # We want to give the 2d interpolator a table that covers the requested range,
    # hence we need to extend the range by at least half a resolution in each direction.
    # Then, we make the choice that the resolution in distance is the quantity that
    # should be preserved. Because the difference between min and max distance is
    # not necessarily an integer multiple of the resolution, we need to add a pad to
    # ensure that the last bin is not cut off. We want the upper edge to be higher than
    # the maximum distance, hence we pad with an additional 1.5 x resolution.
    pad = 2.0 * res
    distances = sc.arange('distance', min_dist - pad, max_dist + pad, res)

    # Create some time bins for event_time_offset.
    # We want our final table to strictly cover the range [0, frame_period].
    nbins = int(frame_period / time_resolution.to(unit=time_unit)) + 1
    time_edges = sc.linspace(
        'event_time_offset', 0.0, frame_period.value, nbins + 1, unit=pulse_period.unit
    )

    # Sort frames by reverse distance
    sorted_frames = sorted(frames, key=lambda x: x.distance.value, reverse=True)

    pieces = []
    # To avoid large RAM usage, and having to split the distances into chunks
    # according to which frame to use, we simply loop over distances one
    # by one here.
    for dist in distances:
        # Find the correct simulation reading
        selected_frame = None
        for frame in sorted_frames:
            if dist.value >= frame.distance.to(unit=dist.unit).value:
                selected_frame = frame
                break
        if selected_frame is None:
            raise ValueError(
                "Building the lookup table failed: the requested position "
                f"{dist:c} is before the component with the lowest "
                "distance in the simulation. The first component in the beamline "
                f"has distance {sorted_frames[0].distance:c}."
            )

        subframes = selected_frame.propagate_to(dist).subframes

        pieces.append(
            _estimate_wavelength_by_polygon_centers(
                subframes=subframes,
                time_edges=time_edges,
                time_unit=time_unit,
                frame_period=frame_period,
            )
        )

    table = sc.DataArray(
        data=sc.concat(pieces, 'distance'),
        coords={"distance": distances, "event_time_offset": time_edges},
    )

    return LookupTable[RunType, Component](
        Lut(
            array=table,
            pulse_period=pulse_period,
            pulse_stride=pulse_stride,
            distance_resolution=table.coords["distance"][1]
            - table.coords["distance"][0],
            time_resolution=table.coords["event_time_offset"][1]
            - table.coords["event_time_offset"][0],
            # TODO: Do we still want to store the chopper info in the lookup table?
        )
    )


def _ltotal_range_from_ltotal(ltotal: sc.Variable) -> tuple[sc.Variable, sc.Variable]:
    return (ltotal.min(), ltotal.max())


def ltotal_range_from_ltotal_detector(
    ltotal: DetectorLtotal[RunType],
) -> LtotalRange[RunType, snx.NXdetector]:
    """
    Compute the range of total flight path lengths from the source to the detector from
    the ltotal variable in the input data for the detector workflow.
    """
    return LtotalRange[RunType, snx.NXdetector](_ltotal_range_from_ltotal(ltotal))


def ltotal_range_from_ltotal_monitor(
    ltotal: MonitorLtotal[RunType, MonitorType],
) -> LtotalRange[RunType, MonitorType]:
    """
    Compute the range of total flight path lengths from the source to the detector from
    the ltotal variable in the input data for the monitor workflow.
    """
    return LtotalRange[RunType, MonitorType](_ltotal_range_from_ltotal(ltotal))


def guess_pulse_stride_from_choppers(
    choppers: DiskChoppers[RunType], pulse_period: PulsePeriod
) -> PulseStride[RunType]:
    """
    If the pulse stride is not provided, we try to guess it from the chopper parameters.
    If there is a chopper rotating slower than the pulse_period, we use its rotation
    frequency to estimate the pulse stride.
    We omit choppers with a zero rotation frequency, as they are considered inactive.
    """
    stride = 1
    for chopper in choppers.values():
        f = sc.abs(chopper.frequency)
        if f.value == 0:
            continue
        stride = max(stride, round((1 / pulse_period / f).to(unit="").value))
    return PulseStride[RunType](stride)


def load_lookup_table_from_file(
    filename: LookupTableFilename[RunType, Component],
) -> LookupTable[RunType, Component]:
    """Load a wavelength lookup table from an HDF5 file."""
    table = sc.io.load_hdf5(filename)

    # Support old format where the metadata were stored as coordinates of the DataArray.
    # Note that no chopper info was saved in the old format.
    if isinstance(table, sc.DataArray):
        to_be_dropped = {
            "pulse_period",
            "pulse_stride",
            "distance_resolution",
            "time_resolution",
            "error_threshold",
        } & set(table.coords)
        table = {
            "array": table.drop_coords(list(to_be_dropped)),
            "pulse_period": table.coords["pulse_period"],
            "pulse_stride": table.coords["pulse_stride"].value,
            "distance_resolution": table.coords["distance_resolution"],
            "time_resolution": table.coords["time_resolution"],
        }

    # Some old tables have the error_threshold stored as an entry in the data group.
    # The masking based on uncertainty is now done later, as part of the tof workflow,
    # so we need to remove this entry if it exists.
    if "error_threshold" in table:
        del table["error_threshold"]

    return LookupTable[RunType, Component](Lut(**table))


def providers(
    mode: Literal["analytical", "simulation", "file"] = "analytical",
) -> tuple[Callable, ...]:
    if mode == "file":
        return (load_lookup_table_from_file,)

    common = (
        ltotal_range_from_ltotal_detector,
        ltotal_range_from_ltotal_monitor,
        guess_pulse_stride_from_choppers,
    )

    if mode == "analytical":
        extra = (
            make_wavelength_lut_from_polygons,
            compute_frame_sequence,
        )

    elif mode == "simulation":
        extra = (
            make_wavelength_lut_from_simulation,
            simulate_chopper_cascade_using_tof,
        )
    else:
        raise ValueError(f"Unknown lookup table provider mode: {mode}")

    return common + extra


def default_parameters(
    mode: Literal["analytical", "simulation", "file"] = "analytical",
) -> dict:
    params = {PulseStrideOffset: None}
    if mode == "file":
        return params

    params.update(
        {
            PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
            DistanceResolution: sc.scalar(0.1, unit="m"),
            TimeResolution: sc.scalar(50.0, unit='us'),
            SourceBounds: SourceBounds(
                time=(sc.scalar(0.0, unit='ms'), sc.scalar(5.0, unit='ms')),
                wavelength=(
                    sc.scalar(0.0, unit='angstrom'),
                    sc.scalar(15.0, unit='angstrom'),
                ),
            ),
        }
    )
    if mode == "simulation":
        params.update(
            {
                NumberOfSimulatedNeutrons: 1_000_000,
                SimulationSeed: None,
                SimulationFacility: 'ess',
            }
        )
    return params
