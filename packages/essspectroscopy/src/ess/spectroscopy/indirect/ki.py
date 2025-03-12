# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for the primary spectrometer of an
indirect geometry time-of-flight spectrometer
"""

from __future__ import annotations

import sciline
from scippnexus import Group

from ess.reduce import time_of_flight
from ess.spectroscopy.types import (
    Filename,
    IncidentDirection,
    SampleEvents,
    SampleName,
    SamplePosition,
    SourceName,
    SourcePosition,
    SourceSamplePathLength,
    TimeOfFlightLookupTable,
    TofSampleEvents,
)


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
    return next(iter(found))


def guess_source_name(file: Filename) -> SourceName:
    """Guess the name of the source in the NeXus instrument file"""
    from scippnexus import File, NXmoderator, NXsource

    with File(file) as data:
        instrument = data['entry/instrument']
        name = determine_name_with_type(
            instrument, None, [NXsource, NXmoderator], 'source'
        )
        return SourceName(name)


def guess_sample_name(file: Filename) -> SampleName:
    """Guess the name of the sample in the NeXus instrument file"""
    from scippnexus import File, NXsample

    with File(file) as data:
        instrument = data['entry/instrument']
        name = determine_name_with_type(instrument, None, [NXsample], 'sample')
        return SampleName(name)


def source_position(file: Filename, source: SourceName) -> SourcePosition:
    """Extract the position of the named source from a NeXus file"""
    from scippnexus import File, compute_positions

    with File(file) as data:
        return compute_positions(data['entry/instrument'][source][...])['position']


def sample_position(file: Filename, sample: SampleName) -> SamplePosition:
    """Extract the position of the named sample from a NeXus file"""
    from scippnexus import File, compute_positions

    with File(file) as data:
        return compute_positions(data['entry/instrument'][sample][...])['position']


# TODO insert this as l1 into the events to make sure to use the correct length
def primary_path_length(
    file: Filename, source: SourcePosition, sample: SamplePosition
) -> SourceSamplePathLength:
    """Compute the primary spectrometer path length from source to sample positions

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

    positions = concat((source, *positions, sample), dim='path')
    diff = positions['path', 1:] - positions['path', :-1]
    return sum(sqrt(dot(diff, diff)))


def unwrap_sample_time(
    sample_events: SampleEvents,
    table: TimeOfFlightLookupTable,
    l1: SourceSamplePathLength,
) -> TofSampleEvents:
    """Use the pivot time to shift neutron event time offsets, recovering 'real'
    time after source pulse per event"""

    pipeline = sciline.Pipeline(
        time_of_flight.providers(),
        params={
            **time_of_flight.default_parameters(),
            time_of_flight.TimeOfFlightLookupTable: table,
            time_of_flight.Ltotal: l1,
            time_of_flight.RawData: sample_events,
        },
    )
    result = pipeline.compute(time_of_flight.TofData)
    # This is time-of-flight at the sample.
    result.bins.coords['sample_tof'] = result.bins.coords.pop('tof')
    return TofSampleEvents(result)


def incident_direction() -> IncidentDirection:
    """Return the incident neutron direction in the laboratory frame"""
    from scipp import vector

    return vector([0, 0, 1.0])


providers = (
    sample_position,
    source_position,
    guess_sample_name,
    guess_source_name,
    primary_path_length,
    unwrap_sample_time,
    incident_direction,
)
