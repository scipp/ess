# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from scipp import DataArray

from ..types import (
    FrameTimeMonitor,
    IncidentSlowness,
    MonitorName,
    MonitorNormalisation,
    MonitorPosition,
    NeXusFileName,
    NormWavelengthEvents,
    PrimaryFocusDistance,
    PrimaryFocusTime,
    PrimarySpectrometerObject,
    SlothMonitor,
    SlownessMonitor,
    SourceFrequency,
    SourceMonitorFlightTime,
    SourceMonitorPathLength,
    SourcePosition,
    WallTimeMonitor,
    WavelengthBins,
    WavelengthEvents,
    WavelengthMonitor,
)


def incident_monitor_normalization(
    slowness: IncidentSlowness, monitor: SlownessMonitor
) -> MonitorNormalisation:
    """For each event, return the corresponding monitor intensity"""
    from scipp import lookup

    coords = list(monitor.coords)
    if len(coords) != 1:
        raise ValueError(f'Monitor expected to have exactly 1 coordinate, has {coords}')
    return lookup(monitor, dim=coords[0])[slowness]


def monitor_position(file: NeXusFileName, monitor: MonitorName) -> MonitorPosition:
    """Extract the position of the named monitor from a NeXus file"""
    from scippnexus import File, compute_positions

    with File(file) as data:
        return compute_positions(data['entry/instrument'][monitor][...])['position']


def source_monitor_path_length(
    file: NeXusFileName, source: SourcePosition, monitor: MonitorPosition
) -> SourceMonitorPathLength:
    """Compute the primary spectrometer path length from source to monitor positions

    Note:
        This *requires* that the instrument group *is sorted* along the beam path.
        HDF5 group entries are sorted alphabetically, so you should ensure that
        the NeXus file was constructed with this in mind.
    """
    from scipp import concat, dot, sqrt, sum
    from scippnexus import File, NXguide, compute_positions

    with File(file) as data:
        positions = [
            compute_positions(v[...])['position']
            for v in data['entry/instrument'][NXguide].values()
        ]

    def vector_length(vector):
        return sqrt(dot(vector, vector))

    # Find the closest guide to the monitor position, ignoring the possibility that
    # a guide could be _beyond_ the monitor _and_ closest :(
    closest = 0
    distance = vector_length(source - monitor)
    for i, position in enumerate(positions):
        d = vector_length(position - monitor)
        if d < distance:
            distance = d
            closest = i

    positions = concat((source, *positions[:closest], monitor), dim='path')
    diff = positions['path', 1:] - positions['path', :-1]
    return sum(vector_length(diff))


def monitor_pivot_time(
    primary: PrimarySpectrometerObject, length: SourceMonitorPathLength
) -> SourceMonitorFlightTime:
    """Find the pivot time between source-pulse arrival times at the monitor position"""
    from choppera.nexus import primary_pivot_time_at

    return primary_pivot_time_at(primary, length)


def monitor_wall_time(
    monitor: FrameTimeMonitor,
    frequency: SourceFrequency,
    least: SourceMonitorFlightTime,
) -> WallTimeMonitor:
    """Convert the independent 'frame_time' coordinate of a histogram DataArray to
    the equivalent unwrapped 'wall_time'

    Parameters
    ----------
    monitor:
        A histogram beam monitor which has with recorded 'frame time' relative to
        the most-recent source pulse
    frequency:
        The source repetition frequency
    least:
        The minimum wall time between source pulse and arrival at the monitor position

    Returns
    -------
    :
        The same intensities with independent axis converted to the likely time since
        neutron-producing proton pulse
    """
    from choppera.nexus import unwrap, unwrap_histogram

    frame = 'frame_time'
    if frame not in monitor.coords:
        raise RuntimeError(f'A FrameTimeMonitor must have coordinate "{frame}"')
    wall = 'wall_time'
    names = {frame: wall}
    if monitor.sizes[frame] + 1 == monitor.coords[frame].size:
        coord, values = unwrap_histogram(
            monitor.coords[frame], monitor.data, frequency, least
        )
    else:
        values = monitor.data
        coord = unwrap(monitor.coord[frame], frequency, least)

    return DataArray(values.rename(names), coords={wall: coord.rename(names)})


def monitor_slowness(
    monitor: WallTimeMonitor,
    length: SourceMonitorPathLength,
    distance: PrimaryFocusDistance,
    focus: PrimaryFocusTime,
) -> SlownessMonitor:
    """Convert the independent 'wall_time' coordinate of a histogram DataArray to
    the equivalent slowness

    Parameters
    ----------
    monitor:
        A histogram beam monitor which has been converted from recorded 'frame time' to
        time since producing proton pulse
    length:
        The path length from the source to the monitor
    distance:
        The distance from the source to the time-of-flight defining (chopper) position
    focus:
        The (mean) time from proton pulse to when all neutrons passed the tof
        defining point

    Returns
    -------
    :
        The same intensities with independent axis converted to the inverse velocity
        of the neutrons, which scales linearly with wall time
    """
    from ..utils import in_same_unit

    wall = 'wall_time'
    if wall not in monitor.coords:
        raise RuntimeError(f'A WallTimeMonitor must have coordinate "{wall}"')
    slow = 'slowness'
    names = {wall: slow}
    wall_time = monitor.coords[wall]
    duration = wall_time - in_same_unit(focus, to=wall_time)
    slowness = (duration / (length - distance)).rename(names).to(unit='s/m')
    return DataArray(monitor.data.rename(names), coords={slow: slowness})


def monitor_wavelength(monitor: SlownessMonitor) -> WavelengthMonitor:
    """Convert the independent 'slowness' coordinate of a histogram DataArray to the
    equivalent wavelength

    Parameters
    ----------
    monitor:
        A histogram beam monitor which has been converted from recorded 'frame time'
        to inverse neutron velocity

    Returns
    -------
    :
        The same intensities with independent axis converted to wavelength
    """
    from scipp.constants import Planck, neutron_mass

    c = Planck / neutron_mass
    slow = 'slowness'
    # wavelength = 'incident_wavelength'
    if slow not in monitor.coords:
        raise RuntimeError(f'A SlownessMonitor must have the coordinate "{slow}"')

    def converter(slowness):
        return (c * slowness).to(unit='angstrom')

    # names = {slow: wavelength}
    converted = monitor.transform_coords(incident_wavelength=converter)
    return converted


def monitor_sloth(
    primary: PrimarySpectrometerObject, monitor: SlownessMonitor
) -> SlothMonitor:
    """Convert the independent 'slowness' coordinate of a histogram DataArray to the
    equivalent sloth, which is -- equivalently -- normalised slowness, normalised
    inverse velocity, or normalised (incident) wavelength

    Parameters
    ----------
    primary:
        The primary spectrometer object describing the choppers and guide setting(s)
    monitor:
        A histogram beam monitor which has been converted from recorded 'frame time' to
        inverse neutron velocity

    Returns
    -------
    :
        The same intensities with independent axis converted to the sloth,
        which is the normalised slowness, inverse velocity, and incident wavelength
    """
    from choppera.nexus import primary_slowness
    from scipp import max, min

    from ..utils import in_same_unit

    slow = 'slowness'
    if slow not in monitor.coords:
        raise RuntimeError(f'A SlownessMonitor must have the coordinate "{slow}"')
    sloth = 'sloth'
    names = {slow: sloth}
    min_max = in_same_unit(primary_slowness(primary), to=monitor.coords[slow])
    normed = (
        (monitor.coords[slow] - min(min_max)) / (max(min_max) - min(min_max))
    ).rename(names)
    return DataArray(monitor.data.rename(names), coords={sloth: normed})


def normalise(
    events: WavelengthEvents,
    monitor: WavelengthMonitor,
    edges: WavelengthBins,
) -> NormWavelengthEvents:
    """Ensure the WavelengthEvents are binned according to the WavelengthBins, then use
    the WavelengthMonitor as a lookup table to record the per-bin normalization.

    Parameters
    ----------
    events
        Event data which includes a per-event sloth coordinate
    monitor
        (Probably) 1-D histogram data with a sloth coordinate
    edges
        1-D bin boundaries for the event data

    Returns
    -------
    :
        Event data binned by sloth, with a coordinate that is the per-bin normalization
    """
    from scipp import lookup

    dim = 'incident_wavelength'
    centres = (edges[:-1] + edges[1:]) / 2
    variances = None
    if monitor.variances is not None:
        monitor = monitor.copy()
        variances = monitor.copy()
        variances.values = monitor.variances
        monitor.variances = None
        variances.variances = None
    counts = lookup(monitor, dim)[centres]
    if variances is not None:
        # Bad form, maybe. But two events with the same sloth bin are normalized
        # by the same monitor counts -- which has a known uncertainty -- and scipp
        # refuses to allow lookup on data which has variances defined.
        counts.variances = (lookup(variances, dim)[centres]).values

    binned = events.bin(**{dim: edges})
    binned.coords['monitor'] = counts
    return binned


providers = (
    incident_monitor_normalization,
    monitor_pivot_time,
    monitor_wall_time,
    monitor_slowness,
    monitor_wavelength,
    monitor_sloth,
    monitor_position,
    source_monitor_path_length,
    normalise,
)
