# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers to compute neutron time-of-flight and wavelength.
"""

import scippneutron as scn
import scippnexus as snx

from .types import (
    CoordTransformGraph,
    GravityVector,
    Position,
    RunType,
    TofDetector,
    WavelengthDetector,
)


def make_coordinate_transform_graph(
    sample_position: Position[snx.NXsample, RunType],
    source_position: Position[snx.NXsource, RunType],
    gravity: GravityVector,
) -> CoordTransformGraph[RunType]:
    """
    Create a graph of coordinate transformations to compute the wavelength from the
    time-of-flight.
    """
    graph = {
        **scn.conversion.graph.beamline.beamline(scatter=False),
        **scn.conversion.graph.tof.elastic("tof"),
        'sample_position': lambda: sample_position,
        'source_position': lambda: source_position,
        'gravity': lambda: gravity,
    }
    return CoordTransformGraph(graph)


def compute_detector_wavelength(
    tof_data: TofDetector[RunType],
    graph: CoordTransformGraph[RunType],
) -> WavelengthDetector[RunType]:
    """
    Compute the wavelength of neutrons detected by the detector.

    Parameters
    ----------
    tof_data:
        Data with a time-of-flight coordinate.
    graph:
        Graph of coordinate transformations.
    """
    return WavelengthDetector[RunType](
        tof_data.transform_coords("wavelength", graph=graph)
    )


providers = (
    make_coordinate_transform_graph,
    compute_detector_wavelength,
)
"""Providers to compute neutron time-of-flight and wavelength."""
