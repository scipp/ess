# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""BEER beamline parameters."""

from collections.abc import Mapping

import scipp as sc
import scippnexus as snx
from ess.powder.types import DetectorTwoTheta, RunType, WavelengthRange
from scippneutron.chopper import DiskChopper

from ess.reduce.nexus.types import Position
from ess.reduce.unwrap.lut import ActiveDiskChoppers, chopper_distance_along_beam
from ess.reduce.unwrap.types import DetectorLtotal

from .types import BeamlineParameters


def effective_chopper_position(choppers: Mapping[str, DiskChopper]) -> sc.Variable:
    """Midpoint of pulse-shaping or modulation choppers, excluding frame choppers."""
    return sc.concat(
        [
            chopper.axle_position
            for name, chopper in choppers.items()
            if not name.startswith('FC')
        ],
        dim='chopper',
    ).mean()


def beamline_parameters(
    wavelength_range: WavelengthRange[RunType],
    detector_ltotal: DetectorLtotal[RunType],
    two_theta: DetectorTwoTheta[RunType],
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    choppers: ActiveDiskChoppers[RunType],
) -> BeamlineParameters[RunType]:
    """Collect BEER beamline parameters.

    ``lambda_min`` and ``lambda_max`` bound the wavelength envelope transmitted by
    the chopper cascade. ``lambda_0`` is the midpoint of that envelope, rather than
    an intensity-weighted wavelength.

    ``L1`` is the distance from the effective chopper position (the midpoint of the
    active pulse-shaping or modulation choppers) to the sample. ``L2`` and ``Ltotal``
    are per-pixel sample-to-detector and chopper-to-detector flight path lengths.
    ``two_theta`` is the per-pixel scattering angle. ``mean_L2``, ``mean_Ltotal``,
    and ``mean_two_theta`` are unweighted means over the selected detector bank(s),
    ignoring NaNs. These scalar approximations can be used for output calibration.

    ``chopper_frequency`` is a DataGroup mapping active chopper names to signed
    rotation frequencies, including modulation choppers where present.
    """
    source_to_sample = sc.norm(sample_position - source_position)
    source_to_chopper = chopper_distance_along_beam(
        effective_chopper_position(choppers), source_position
    )
    l2 = detector_ltotal - source_to_sample
    ltotal = detector_ltotal - source_to_chopper
    lambda_min = wavelength_range.nanmin()
    lambda_max = wavelength_range.nanmax()
    return BeamlineParameters[RunType](
        sc.DataGroup(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            lambda_0=(lambda_min + lambda_max) / 2,
            L1=source_to_sample - source_to_chopper,
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
