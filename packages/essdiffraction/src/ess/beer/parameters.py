# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""BEER beamline parameters."""

import scipp as sc
from ess.powder.types import EmptyDetector, RunType, WavelengthRange

from ess.reduce.unwrap.lut import ActiveDiskChoppers

from .types import BeamlineGeometryGraph, BeamlineParameters


def beamline_parameters(
    wavelength_range: WavelengthRange[RunType],
    detector: EmptyDetector[RunType],
    graph: BeamlineGeometryGraph[RunType],
    choppers: ActiveDiskChoppers[RunType],
) -> BeamlineParameters[RunType]:
    """Collect BEER beamline parameters.

    ``lambda_min`` and ``lambda_max`` bound the wavelength envelope transmitted by
    the chopper cascade. ``lambda_0`` is the midpoint of that envelope, rather than
    an intensity-weighted wavelength.

    ``mean_L2``, ``mean_Ltotal``, and ``mean_two_theta`` are unweighted means over
    the selected detector bank(s), ignoring NaNs. These scalar approximations can
    be used for output calibration.

    ``chopper_frequency`` is a DataGroup mapping active chopper names to signed
    rotation frequencies, including modulation choppers where present.
    """
    geometry = detector.transform_coords(
        ['L1', 'L2', 'Ltotal', 'two_theta'], graph=graph, keep_intermediate=False
    ).coords
    l2 = geometry['L2']
    ltotal = geometry['Ltotal']
    two_theta = geometry['two_theta']
    lambda_min = wavelength_range.nanmin()
    lambda_max = wavelength_range.nanmax()
    return BeamlineParameters[RunType](
        sc.DataGroup(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            lambda_0=(lambda_min + lambda_max) / 2,
            L1=geometry['L1'],
            L2=l2,
            Ltotal=ltotal,
            two_theta=two_theta,
            mean_L2=l2.nanmean(),
            mean_Ltotal=ltotal.nanmean(),
            mean_two_theta=two_theta.nanmean(),
            chopper_frequency=sc.DataGroup(
                {name: chopper.frequency for name, chopper in choppers.items()}
            ),
        )
    )
