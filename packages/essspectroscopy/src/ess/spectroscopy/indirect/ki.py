"""Utilities for the primary spectrometer of an indirect geometry time-of-flight spectrometer"""

from __future__ import annotations

from ess.spectroscopy.types import (
    NeXusFileName, SourceName, SampleName, FocusComponentNames, FocusComponentName, SourcePosition, SamplePosition,
    PrimaryFocusDistance, PrimarySpectrometerObject, PrimaryFocusTime, SourceSamplePathLength, SourceFrequency,
    SourceDuration, SourceDelay, SourceVelocities, SourceSampleFlightTime, SampleFrameTime, SampleTime,
    IncidentSlowness, IncidentWavenumber, IncidentDirection, IncidentWavelength, IncidentWavevector, IncidentEnergy

)
from scippnexus import Group


def determine_name_with_type(
    instrument: Group, name: str | None, options: list, type_name: str
) -> str:
    """Investigate an open NeXus file group for objects with matching name or base type

    Parameter
    ---------
    instrument: scippnexus.Group
        The group to investigate, likely /entry/instrument
    name: str
        A preferred object name, will be returned if present in the scippnexus.Group
    options: list
        Should be scippnexus define NeXus object types, e.g., scippnexus.NXdetector
    type_name: str
        Candidate group members contain this name

    Returns
    -------
    :
        The matching group-member name

    Raises
    ------
    ValueError
        If no group member has the specified `name`,
        contains the specified `type_name` or is of the specified `option` type
    """
    if name is not None and name in instrument:
        return name
    found = {x for x in instrument if type_name in x.lower()}
    for option in options:
        found.update(set(instrument[option]))
    if len(found) != 1:
        raise ValueError(f"Could not determine {type_name} name: {found}")
    return list(found)[0]


def guess_source_name(file: NeXusFileName) -> SourceName:
    """Guess the name of the source in the NeXus instrument file"""
    from scippnexus import File, NXmoderator, NXsource

    with File(file) as data:
        instrument = data['entry/instrument']
        name = determine_name_with_type(
            instrument, None, [NXsource, NXmoderator], 'source'
        )
        return SourceName(name)


def guess_sample_name(file: NeXusFileName) -> SampleName:
    """Guess the name of the sample in the NeXus instrument file"""
    from scippnexus import File, NXsample

    with File(file) as data:
        instrument = data['entry/instrument']
        name = determine_name_with_type(instrument, None, [NXsample], 'sample')
        return SampleName(name)


def guess_focus_component_names(file: NeXusFileName) -> FocusComponentNames:
    """Guess the names of the components which define the focus distance of a Primary Spectrometer

    Note
    ----
    The order of components in the NeXus file must be consistent with the order of components along the beamline.
    This assumes that only NXdisk_chopper are used to define a focus distance, and that the first chopper
    or choppers along the beamline, within a fixed small distance, can define the focus distance.
    The component type, primacy, and allowed distance range could be user configurable inputs.

    Parameters
    ----------
    file: NeXusFileName
        The (HDF5) NeXus file name that contains an 'entry/instrument' group with one or more
        `scippnexus.NXdisk_chopper` groups inside

    Returns
    -------
    :
        The name or names of the time-focus-defining choppers, given the restrictions noted above
    """
    from scipp import scalar
    from scippnexus import File, NXdisk_chopper, compute_positions

    from ..utils import norm

    allowance = scalar(0.5, unit='m')

    with File(file) as data:
        instrument = data['entry/instrument']
        choppers = {
            k: compute_positions(v[...])['position']
            for k, v in instrument[NXdisk_chopper].items()
        }

    names = list(choppers.keys())
    focus_names = [FocusComponentName(names[0])]
    last = choppers[names[0]]['position']
    distance = 0 * allowance
    for name in names[1:]:
        x = choppers[name]['position']
        distance += norm(x - last)
        last = x
        if distance <= allowance:
            focus_names.append(FocusComponentName(name))
        else:
            break
    return FocusComponentNames(focus_names)


def source_position(file: NeXusFileName, source: SourceName) -> SourcePosition:
    """Extract the position of the named source from a NeXus file"""
    from scippnexus import File, compute_positions

    with File(file) as data:
        return compute_positions(data['entry/instrument'][source][...])['position']


def sample_position(file: NeXusFileName, sample: SampleName) -> SamplePosition:
    """Extract the position of the named sample from a NeXus file"""
    from scippnexus import File, compute_positions

    with File(file) as data:
        return compute_positions(data['entry/instrument'][sample][...])['position']


def focus_distance(
    file: NeXusFileName, origin: SourcePosition, names: FocusComponentNames
) -> PrimaryFocusDistance:
    """Extract the average distance from the provided source position to the named components

    Warnings
    --------
    This distance is straight-line distance which may not precisely match the path-length that the neutrons take
    due to curved guides or other reflecting components. Care should be taken if the primary spectrometer includes
    any such components to ensure that the difference in flight path and straight-line path is not important

    Parameters
    ----------
    file: NeXusFileName
        The name of the HDF5 NeXus file containing the 'entry/instrument' group with the named components
    origin:
        The position of the source, likely obtained from the same NeXus file
    names: list
        The name(s) of the components whose average distance is calculated

    Returns
    -------
    :
        The average straight-line distance from the source position to the named component(s)
    """
    from scippnexus import File, compute_positions

    from ..utils import norm

    pos = 0 * origin
    with File(file) as data:
        for name in names:
            pos += compute_positions(data['entry/instrument'][name][...])['position']
    pos /= len(names)
    return norm(pos - origin)


def focus_time(
    primary: PrimarySpectrometerObject, distance: PrimaryFocusDistance
) -> PrimaryFocusTime:
    """Return the time, relative to the pulse time, that neutrons pass the focus position in a primary spectrometer"""
    from choppera.nexus import primary_focus_time

    return primary_focus_time(primary, distance)


def primary_path_length(
    file: NeXusFileName, source: SourcePosition, sample: SamplePosition
) -> SourceSamplePathLength:
    """Compute the primary spectrometer path length from source to sample positions

    Note:
        This *requires* that the instrument group *is sorted* along the beam path. HDF5 group entries are sorted
        alphabetically, so you should ensure that the NeXus file was constructed with this in mind.
    """
    from scipp import concat, dot, sqrt, sum
    from scippnexus import File, NXguide, compute_positions

    with File(file) as data:
        positions = [
            compute_positions(v[...])['position']
            for v in data['entry/instrument'][NXguide].values()
        ]

    positions = concat((source, *positions, sample), dim='path')
    diff = positions['path', 1:] - positions['path', :-1]
    return sum(sqrt(dot(diff, diff)))


def primary_spectrometer(
    file: NeXusFileName,
    source: SourceName,
    sample: SampleName,
    frequency: SourceFrequency,
    duration: SourceDuration,
    delay: SourceDelay,
    velocities: SourceVelocities,
) -> PrimarySpectrometerObject:
    """Construct and return a choppera.PrimarySpectrometer object

    Parameters
    ----------
    file:
        The HDF5 NeXus file with information about the primary spectrometer under 'entry/instrument'
    source:
        The name of the source component in the NeXus instrument group
    sample:
        The name of the sample component in the NeXus instrument group
    frequency:
        The frequency of the source, e.g., scipp.scalar(14.0, unit='Hz') for ESS
    duration:
        The source pulse duration, e.g., approximately scipp.scalar(3.0, unit='msec') for ESS
    delay:
        The velocity dependent time delay between source pulse time and neutrons exiting the moderator,
        should be 1-D (with dim='wavelength') and have values corresponding to the provided `velocities`
    velocities:
        The velocities of neutrons produced by the source; must be 1-D (with dim='wavelength') with at least two
        values. The velocities and delay times should be the same length, with a 1:1 correspondence.
        These values are then used to define the time-slowness phase space polygon produced by each source pulse.

    Returns
    -------
    :
        A choppera.PrimarySpectrometer object representing the NeXus file contents
    """
    from choppera.nexus import primary_spectrometer
    from scippnexus import File

    with File(file) as data:
        instrument = data['entry/instrument']
        assert (
            source in instrument
        ), f"The source '{source}' is not in the instrument group"
        assert (
            sample in instrument
        ), f"The sample '{sample}' is not in the instrument group"
        return primary_spectrometer(
            instrument, source, sample, frequency, duration, delay, velocities
        )


def primary_pivot_time(primary: PrimarySpectrometerObject) -> SourceSampleFlightTime:
    """Return a time, relative to the source pulse time, between neutron-arrival-time periods

    Notes
    -----
    The determined pivot time is be before the earliest arrival time of neutrons produced in a source pulse
    that then are transmitted through the primary spectrometer, while also being after the latest arrival time
    from the preceding pulse.

    Parameters
    ----------
    primary:
        The choppera.PrimarySpectrometer object

    Returns
    -------
    :
        The determined pivot time
    """
    from choppera.nexus import primary_pivot_time as primary_time

    return primary_time(primary)


def unwrap_sample_time(
    times: SampleFrameTime, frequency: SourceFrequency, least: SourceSampleFlightTime
) -> SampleTime:
    """Use the pivot time to shift neutron event time offsets, recovering 'real' time after source pulse per event"""
    from choppera.nexus import unwrap as choppera_unwrap

    return choppera_unwrap(times, frequency, least)


def incident_slowness(
    length: SourceSamplePathLength,
    time: SampleTime,
    distance: PrimaryFocusDistance,
    focus: PrimaryFocusTime,
) -> IncidentSlowness:
    """Find the slowness of neutrons which pass through the primary spectrometer

    Parameters
    ----------
    length:
        The path length travelled by neutrons which go through the primary spectrometer
    time:
        The unwrapped 'wall' time for each event, relative to their producing source pulse
    distance:
        How far along the primary spectrometer the neutrons travelled before reaching the time-focus point
    focus:
        The 'wall' time relative to a source pulse time at which all neutrons must have passed the time-focus point

    Returns
    -------
    :
        The inverse of the velocity for each neutron, that is its 'slowness', which is proportional to wavelength
    """
    from ..utils import in_same_unit

    tof = time - in_same_unit(focus, to=time)
    slowness = tof / (length - distance)  # slowness _is_ inverse velocity
    return slowness


def incident_wavelength(slowness: IncidentSlowness) -> IncidentWavelength:
    """Calculate the incident wavelength from the incident slowness for each neutron"""
    from scipp.constants import Planck, neutron_mass

    return (slowness * Planck / neutron_mass).to(unit='angstrom')


def incident_wavenumber(slowness: IncidentSlowness) -> IncidentWavenumber:
    """Calculate the incident wave number from the incident slowness for each neutron"""
    from scipp.constants import hbar, neutron_mass

    return (neutron_mass / hbar / slowness).to(unit='1/angstrom')


def incident_direction() -> IncidentDirection:
    """Return the incident neutron direction in the laboratory frame, which is defined to be [001]"""
    from scipp import vector

    return vector([0, 0, 1.0])


def incident_wavevector(
    ki_magnitude: IncidentWavenumber, direction: IncidentDirection
) -> IncidentWavevector:
    """Find the incident wavevector from its magnitude and direction"""
    return ki_magnitude * direction


def incident_energy(ki: IncidentWavenumber) -> IncidentEnergy:
    """Convert the incident wavenumber to incident energy in meV"""
    from scipp.constants import hbar, neutron_mass

    return ((hbar * hbar / 2 / neutron_mass) * ki * ki).to(unit='meV')


providers = (
    sample_position,
    source_position,
    guess_sample_name,
    guess_source_name,
    guess_focus_component_names,
    focus_distance,
    focus_time,
    primary_path_length,
    primary_spectrometer,
    primary_pivot_time,
    unwrap_sample_time,
    incident_direction,
    incident_slowness,
    incident_wavelength,
    incident_wavenumber,
    incident_wavevector,
    incident_energy,
)
