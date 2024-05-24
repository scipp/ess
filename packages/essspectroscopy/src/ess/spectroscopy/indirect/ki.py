"""Utilities for the primary spectrometer of an indirect geometry time-of-flight spectrometer"""
#TODO move elsewhere since possibly (probably) too specific to BIFROST.

from ess.spectroscopy.types import *


def source_position(file: NeXusFileName) -> SourcePosition:
    from scippnexus import compute_positions, NXsource, NXmoderator, File
    options = {}
    with File(file) as data:
        instrument = data['entry/instrument']
        for nx_type in (NXsource, NXmoderator):
            options.update(instrument[nx_type])
    if len(options) == 0:
        raise RuntimeError('No source position')
    return compute_positions(list(options.values())[-1][...])['position']


def sample_position(file: NeXusFileName) -> SamplePosition:
    from scippnexus import compute_positions, NXsample, File
    from scipp import vector
    options = {}
    with File(file) as data:
        instrument = data['entry/instrument']
        options.update(instrument[NXsample])
    if len(options) == 0:
        return vector(value=[0., 0., 0.], unit='m')
    return compute_positions(list(options.values())[-1][...])['position']


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
