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
    DataAtSample,
    Filename,
    IncidentDirection,
    PrimarySpecCoordTransformGraph,
    RunType,
    SampleName,
    SamplePosition,
    SourceName,
    SourcePosition,
    TimeOfFlightLookupTable,
    TofData,
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


def primary_spectrometer_coordinate_transformation_graph() -> (
    PrimarySpecCoordTransformGraph[RunType]
):
    """Return a coordinate transformation graph for the primary spectrometer.

    Returns
    -------
    :
        Coordinate transformation graph for the primary spectrometer.
    """
    # For the incident beam, the original implementation here used the guides to
    # determine a more accurate estimate of the path length.
    # See function `primary_path_length` in
    # commit 929ef7f97e00a1e26c254fd5f08c8a3346255970
    # The result differs by <1mm from the straight line distance.
    # This should be well below the measurement accuracy.
    # So we use the simpler straight line distance here.

    from scippneutron.conversion.beamline import L1, straight_incident_beam

    return PrimarySpecCoordTransformGraph[RunType](
        {
            "incident_beam": straight_incident_beam,
            "L1": L1,
        }
    )


def unwrap_sample_time(
    sample_data: DataAtSample[RunType],
    table: TimeOfFlightLookupTable,
) -> TofData[RunType]:
    """Compute time-of-flight at the sample using a lookup table.

    Parameters
    ----------
    sample_data:
        Data with 'event_time_offset' and 'event_time_zero' coordinates
        describing the time-of-arrival at the sample.
    table:
        A time-of-flight lookup table.

    Returns
    -------
    :
        A copy of ``sample_data`` with a "sample_tof" coordinate containing
        the time-of-flight at the sample.
    """

    pipeline = sciline.Pipeline(
        time_of_flight.providers(),
        params={
            **time_of_flight.default_parameters(),
            time_of_flight.TimeOfFlightLookupTable: table,
            time_of_flight.Ltotal: sample_data.coords['L1'],
            time_of_flight.RawData: sample_data,
        },
    )
    result = pipeline.compute(time_of_flight.TofData)
    # This is time-of-flight at the sample.
    result.bins.coords['sample_tof'] = result.bins.coords.pop('tof')
    return TofData(result)


def incident_direction() -> IncidentDirection:
    """Return the incident neutron direction in the laboratory frame"""
    from scipp import vector

    return vector([0, 0, 1.0])


providers = (
    sample_position,
    source_position,
    guess_sample_name,
    guess_source_name,
    unwrap_sample_time,
    incident_direction,
)
