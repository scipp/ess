# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers to compute neutron time-of-flight and wavelength.
"""

import scippneutron as scn

from .types import (
    CoordTransformGraph,
    CountsWavelength,
    DetectorTofData,
    RunType,
)


def make_coordinate_transform_graph() -> CoordTransformGraph:
    """
    Create a graph of coordinate transformations to compute the wavelength from the
    time-of-flight.
    """
    graph = {
        **scn.conversion.graph.beamline.beamline(scatter=False),
        **scn.conversion.graph.tof.elastic("tof"),
    }
    return CoordTransformGraph(graph)


def compute_detector_wavelength(
    tof_data: DetectorTofData[RunType],
    graph: CoordTransformGraph,
) -> CountsWavelength[RunType]:
    """
    Compute the wavelength of neutrons detected by the detector.

    Parameters
    ----------
    tof_data:
        Data with a time-of-flight coordinate.
    graph:
        Graph of coordinate transformations.
    """
    return CountsWavelength[RunType](
        tof_data.transform_coords("wavelength", graph=graph)
    )


providers = (
    make_coordinate_transform_graph,
    compute_detector_wavelength,
)
"""Providers to compute neutron time-of-flight and wavelength."""
