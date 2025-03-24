# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for the primary spectrometer of an
indirect geometry time-of-flight spectrometer
"""

from __future__ import annotations

import sciline

from ess.reduce import time_of_flight
from ess.spectroscopy.types import (
    DataAtSample,
    PrimarySpecCoordTransformGraph,
    RunType,
    TimeOfFlightLookupTable,
    TofData,
)


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
    del result.bins.coords['event_time_offset']
    del result.bins.coords['event_time_zero']
    return TofData(result)


providers = (
    primary_spectrometer_coordinate_transformation_graph,
    unwrap_sample_time,
)
