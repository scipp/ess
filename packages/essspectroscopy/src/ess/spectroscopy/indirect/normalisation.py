from ess.spectroscopy.types import *


def incident_monitor_normalization(
    slowness: IncidentSlowness, monitor: SlownessMonitor
) -> MonitorNormalisation:
    """For the incident slowness of each event, return the corresponding monitor intensity"""
    from scipp import lookup

    coords = list(monitor.coords)
    if len(coords) != 1:
        raise ValueError(f'Monitor expected to have exactly 1 coordinate, has {coords}')
    return lookup(monitor, dim=coords[0])[slowness]


def monitor_pivot_time(
    primary: PrimarySpectrometerObject, length: SourceMonitorPathLength
) -> SourceMonitorFlightTime:
    from choppera.nexus import primary_pivot_time_at

    return primary_pivot_time_at(primary, length)


def monitor_wall_time(
    monitor: FrameTimeMonitor,
    frequency: SourceFrequency,
    least: SourceMonitorFlightTime,
) -> WallTimeMonitor:
    """Convert the independent 'frame_time' coordinate of a histogram DataArray to the equivalent unwrapped 'wall_time'

    Parameters
    ----------
    monitor: A histogram beam monitor which has with recorded 'frame time' relative to the most-recent source pulse
    frequency: The source repetition frequency
    least: The minimum wall time between source pulse and arrival at the monitor position

    Returns
    -------
    The same intensities with independent axis converted to the likely time since neutron-producing proton pulse
    """
    from choppera.nexus import unwrap

    coords = list(monitor.coords)
    if len(coords) != 1:
        raise ValueError(f'Monitor expected to have exactly 1 coordinate, has {coords}')
    frame = coords[0]
    wall = 'wall_time'
    names = {frame: wall}
    return DataArray(
        monitor.data.rename(names),
        coords={wall: unwrap(monitor.coords[frame], frequency, least).rename(names)},
    )


def monitor_slowness(
    monitor: WallTimeMonitor,
    length: SourceMonitorPathLength,
    distance: PrimaryFocusDistance,
    focus: PrimaryFocusTime,
) -> SlownessMonitor:
    """Convert the independent 'wall_time' coordinate of a histogram DataArray to the equivalent slowness

    Parameters
    ----------
    monitor: A histogram beam monitor which has been converted from recorded 'frame time' to time since producing proton pulse
    length: The path length from the source to the monitor
    distance: The distance from the source to the time-of-flight defining (chopper) position
    focus: The (mean) time from proton pulse to when all neutrons passed the tof defining point

    Returns
    -------
    The same intensities with independent axis converted to the inverse velocity of the neutrons, which scales
    linearly with wall time
    """
    coords = list(monitor.coords)
    if len(coords) != 1:
        raise ValueError(f'Monitor expected to have exactly 1 coordinate, has {coords}')
    wall = coords[0]
    slow = 'slowness'
    names = {wall: slow}
    slowness = (
        ((monitor.coords[wall] - focus) / (length - distance))
        .rename(names)
        .to(unit='s/m')
    )
    return DataArray(monitor.data.rename(names), coords={slow: slowness})


providers = [
    incident_monitor_normalization,
    monitor_pivot_time,
    monitor_wall_time,
    monitor_slowness,
]
