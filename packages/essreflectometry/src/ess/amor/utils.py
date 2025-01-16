import scipp as sc

from ess.reflectometry.types import (
    DetectorRotation,
    RunType,
    SampleRotation,
)

from .geometry import Detector
from .types import ThetaBins


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


providers = (theta_grid,)
