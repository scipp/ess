# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    BeamDivergenceLimits,
    DetectorRotation,
    QBins,
    RunType,
    SampleRotation,
    SampleRun,
    ThetaBins,
    WavelengthBins,
)
from .geometry import Detector


def theta_grid(
    nu: DetectorRotation[RunType], mu: SampleRotation[RunType]
) -> ThetaBins[RunType]:
    """Special grid used to create intensity maps over
    (theta, wavelength).
    The grid avoids aliasing artifacts that occur if the
    theta bins overlap the blade edges."""
    # angular offset of two blades:
    bladeAngle = 2.0 * sc.asin(0.5 * Detector.bladeZ / Detector.distance)
    # associate an angle with each z-coordinate on one blade
    blade_grid = sc.atan(
        sc.arange("theta", 0, 33)
        * Detector.dZ
        / (Detector.distance + sc.arange("theta", 0, 33) * Detector.dX)
    )
    # approximate angular step width on one blade
    stepWidth = blade_grid[1] - blade_grid[0]
    # shift "downwards" of the grid in order to define boundaries rather than centers
    blade_grid = blade_grid - 0.2 * stepWidth

    delta_grid = sc.array(
        dims=["theta"],
        values=[],
        unit=blade_grid.unit,
        dtype=blade_grid.dtype,
    )
    # loop over all blades but one:
    for _ in range(Detector.nBlades.value - 1):
        # append the actual blade's grid to the array of detector-local angles
        delta_grid = sc.concat((delta_grid, blade_grid), "theta")
        # shift the blade grid by the angular offset
        blade_grid = blade_grid + bladeAngle
        # remove all entries in the detector local grid which are above the
        #  expected next value (plus some space to avoid very thin bins)
        delta_grid = delta_grid[delta_grid < blade_grid[0] - 0.5 * stepWidth]
    # append the grid of the last blade.
    delta_grid = sc.concat((delta_grid, blade_grid), "theta")

    # add angular position of the detector
    grid = (
        nu.to(unit="rad")
        - mu.to(unit="rad")
        - sc.array(
            dims=delta_grid.dims, values=delta_grid.values[::-1], unit=delta_grid.unit
        ).to(unit="rad")
        + 0.5 * Detector.nBlades * bladeAngle.to(unit="rad")
    )
    return grid


def qgrid(
    detector_rotation: DetectorRotation[SampleRun],
    sample_rotation: SampleRotation[SampleRun],
    wbins: WavelengthBins,
    bdlims: BeamDivergenceLimits,
) -> QBins:
    '''Generates a suitable Q-binnning from
    the limits on wavelength and divergence angle.

    The binning is a geometric grid starting from
    the minimum achievable Q value or ``1e-3 A``, whichever is larger.
    '''
    theta_min = (
        bdlims[0].to(unit='rad', copy=False)
        + detector_rotation.to(unit='rad', dtype='float64')
        - sample_rotation.to(unit='rad', dtype='float64')
    )
    theta_max = (
        bdlims[-1].to(unit='rad', copy=False)
        + detector_rotation.to(unit='rad', dtype='float64')
        - sample_rotation.to(unit='rad', dtype='float64')
    )
    wmin, wmax = wbins[0], wbins[-1]
    qmin = reflectometry_q(wavelength=wmax, theta=theta_min)
    qmax = reflectometry_q(wavelength=wmin, theta=theta_max)
    qmin = max(qmin, sc.scalar(1e-3, unit='1/angstrom'))
    return QBins(sc.geomspace('Q', qmin, qmax, 501))


providers = (theta_grid, qgrid)
