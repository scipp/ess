# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Automatic bin-edge generation for powder diffraction."""

import scipp as sc
import scippneutron as scn

from ess.reduce.unwrap import ChopperFrameSequence

from .types import (
    DetectorTwoTheta,
    DspacingBins,
    DspacingNBins,
    RunType,
    SampleRun,
    WavelengthRange,
)


def wavelength_range_from_chopper_frames(
    frames: ChopperFrameSequence[RunType],
) -> WavelengthRange[RunType]:
    """Return the wavelength envelope transmitted by the chopper cascade."""
    return WavelengthRange[RunType](frames[-1].bounds()['wavelength'])


def dspacing_bins_from_wavelength_and_two_theta(
    wavelength_range: WavelengthRange[SampleRun],
    two_theta: DetectorTwoTheta[SampleRun],
    nbins: DspacingNBins,
) -> DspacingBins:
    """Make d-spacing bin edges from wavelength and detector geometry.

    Wavelength and two-theta are independent geometry and beamline quantities.
    Pairing their opposite extrema and transforming them to d-spacing gives an
    envelope that covers every detector pixel.
    """
    if nbins < 1:
        raise ValueError(f'DspacingNBins must be positive, got {nbins}.')
    dspacing_range = scn.conversion.tof.dspacing_from_wavelength(
        wavelength=sc.concat(
            [wavelength_range.nanmin(), wavelength_range.nanmax()], dim='bound'
        ),
        two_theta=sc.concat([two_theta.nanmax(), two_theta.nanmin()], dim='bound'),
    )
    return DspacingBins(
        sc.linspace(
            dim='dspacing',
            start=dspacing_range[0].value,
            stop=dspacing_range[-1].value,
            num=nbins + 1,
            unit=dspacing_range.unit,
        )
    )


providers = (
    wavelength_range_from_chopper_frames,
    dspacing_bins_from_wavelength_and_two_theta,
)
"""Sciline providers for automatic powder-diffraction binning."""
