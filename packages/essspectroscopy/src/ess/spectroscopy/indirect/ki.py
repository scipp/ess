"""Utilities for the primary spectrometer of an indirect geometry time-of-flight spectrometer"""
#TODO move elsewhere since possibly (probably) too specific to BIFROST.

from ess.spectroscopy.types import *
from scippnexus import Group

def determine_name_with_type(instrument: Group, name: str | None, options: list, type_name: str) -> str:
    if name is not None and name in instrument:
        return name
    foud = {x for x in instrument if type_name in x.lower()}
    for option in options:
        found.update(set(instrument[option]))
    if len(found) != 1:
        raise RuntimeError(f"Could not determine {type_name} name: {found}")
    return list(found)[0]


def guess_source_name(file: NeXusFileName) -> SourceName:
    from scippnexus import NXsource, NXmoderator, File
    with File(file) as data:
        instrument = data['entry/instrument']
        return determine_name_with_type(instrument, name, [NXsource, NXmoderator], 'source')

def guess_sample_name(file: NeXusFileName) -> SampleName:
    from scippnexus import NXsample, File
    with File(file) as data:
        instrument = data['entry/instrument']
        return determine_name_with_type(instrument, name, [NXsample], 'sample')


def source_position(file: NeXusFileName, source: SourceName) -> SourcePosition:
    from scippnexus import compute_positions, File
    with File(file) as data:
        source = data['entry/instrument'][source][...]
    return compute_positions(source)['position']


def sample_position(file: NeXusFileName, sample: SampleName) -> SamplePosition:
    from scippnexus import compute_positions, File
    with File(file) as data:
        sample = data['entry/instrument'][sample][...]
    return compute_positions(sample)['position']


def primary_path_length(file: NeXusFileName, source: SourcePosition, sample: SamplePosition) -> SourceSamplePathLength:
    """Compute the primary spectrometer path length

    Note:
        This *requires* that the instrument group *is sorted* along the beam path. HDF5 group entries are sorted
        alphabetically, so you should ensure that the NeXus file was constructed with this in mind.
    """
    from scippnexus import compute_positions, NXguide, File
    from scipp import dot, sqrt, sum, concat
    with File(file) as data:
        positions = [compute_positions(v[...])['position'] for v in data['entry/instrument'][NXguide].values()]

    positions = concat((source, *positions, sample), dim='path')
    diff = positions['path', 1:] - positions['path', :-1]
    return sum(sqrt(dot(diff, diff)))


def primary_spectrometer(file: NeXusFileName,
                          source: SourceName,
                          sample: SampleName,
                          frequency: SourceFrequency,
                          duration: SourceDuration,
                          delay: SourceDelay,
                          velocities: SourceVelocities) -> PrimarySpectrometerObject:
    from scippnexus import File
    from choppera.nexus import primary_spectrometer
    with File(file) as data:
        instrument = data['entry/instrument']
        assert source in instrument, f"The source '{source}' is not in the instrument group"
        assert sample in instrument, f"The sample '{sample}' is not in the instrument group"
        return primary_spectrometer(instrument, source, sample, frequency, duration, delay, velocities)


def primary_pivot_time(primary: PrimarySpectrometerObject) -> SourceSampleFlightTime:
    from choppera.nexus import primary_pivot_time as primary_time
    return primary_time(primary)


def unwrap_sample_time(times: SampleFrameTime, frequency: SourceFrequency, least: SourceSampleFlightTime) -> SampleTime:
    from choppera.nexus import unwrap as choppera_unwrap
    return choppera_unwrap(times, frequency, least)


def ki_wavenumber(length: SourceSamplePathLength, time: SampleTime) -> IncidentWavenumber:
    from scipp.constants import neutron_mass, hbar
    velocity = length / time
    k = neutron_mass * velocity / hbar
    return k.to(unit='1/angstrom')


def ki_wavevector(ki_magnitude: IncidentWavenumber) -> IncidentWavevector:
    from scipp import vector
    z = vector([0, 0, 1.])
    return ki_magnitude * z


def ei(ki: IncidentWavenumber) -> IncidentEnergy:
    from scipp.constants import hbar, neutron_mass
    return ((hbar * hbar / 2 / neutron_mass) * ki * ki).to(unit='meV')


providers = [
    primary_pivot_time,
    primary_path_length,
    sample_position,
    source_position,
    guess_sample_name,
    guess_source_name,
    unwrap_sample_time,
    primary_spectrometer,
]