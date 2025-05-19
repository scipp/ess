# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Handling of initial parameters / primary spectrometer."""

from __future__ import annotations

from ess.spectroscopy.types import (
    PrimarySpecCoordTransformGraph,
    RunType,
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


providers = (primary_spectrometer_coordinate_transformation_graph,)
