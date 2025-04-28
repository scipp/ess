# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers to compute neutron time-of-flight and wavelength.
"""

import scipp as sc
import scippneutron as scn

from ess.reduce import time_of_flight

from .types import (
    Choppers,
    CoordTransformGraph,
    DetectorTofData,
    DetectorWavelengthData,
    RunType,
    SampleRun,
    SimulationResults,
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


def simulate_chopper_cascade(choppers: Choppers[SampleRun]) -> SimulationResults:
    """
    Simulate neutrons traveling through the chopper cascade.

    Parameters
    ----------
    choppers:
        Chopper settings.
    """
    return time_of_flight.simulate_beamline(
        choppers=choppers,
        neutrons=200_000,
        source_position=sc.vector([0, 0, 0], unit="m"),
    )


def compute_detector_wavelength(
    tof_data: DetectorTofData[RunType],
    graph: CoordTransformGraph,
) -> DetectorWavelengthData[RunType]:
    """
    Compute the wavelength of neutrons detected by the detector.

    Parameters
    ----------
    tof_data:
        Data with a time-of-flight coordinate.
    graph:
        Graph of coordinate transformations.
    """
    return DetectorWavelengthData[RunType](
        tof_data.transform_coords("wavelength", graph=graph)
    )


providers = (
    make_coordinate_transform_graph,
    simulate_chopper_cascade,
    compute_detector_wavelength,
)
"""Providers to compute neutron time-of-flight and wavelength."""
