import sciline
import scipp as sc

from ess.reflectometry.types import DetectorRotation, Run, SampleRotation

from .geometry import Detector


class ThetaBins(sciline.Scope[Run, sc.Variable], sc.Variable):
    '''Binning in theta that takes into consideration that some
    detector pixels have the same theta value'''


def theta_grid(nu: DetectorRotation[Run], mu: SampleRotation[Run]) -> ThetaBins[Run]:
    # angular offset of two blades:
    bladeAngle = 2.0 * sc.asin(0.5 * Detector.bladeZ / Detector.distance)
    # associate an angle with each z-coordinate on one blade
    blade_grid = sc.atan(
        sc.arange('theta', 0, 33)
        * Detector.dZ
        / (Detector.distance + sc.arange('theta', 0, 33) * Detector.dX)
    )
    # approximate angular step width on one blade
    stepWidth = blade_grid[1] - blade_grid[0]
    # shift "downwards" of the grid in order to define boundaries rather than centers
    blade_grid = blade_grid - 0.2 * stepWidth

    delta_grid = sc.array(
        dims=['theta'], values=[], unit=blade_grid.unit, dtype=blade_grid.dtype
    )
    # loop over all blades but one:
    for _ in range(Detector.nBlades.value - 1):
        # append the actual blade's grid to the array of detector-local angles
        delta_grid = sc.concat((delta_grid, blade_grid), 'theta')
        # shift the blade grid by the angular offset
        blade_grid = blade_grid + bladeAngle
        # remove all entries in the detector local grid which are above the
        #  expected next value (plus some space to avoid very thin bins)
        delta_grid = delta_grid[delta_grid < blade_grid[0] - 0.5 * stepWidth]
    # append the grid of the last blade.
    delta_grid = sc.concat((delta_grid, blade_grid), 'theta')

    # add angular position of the detector
    theta_grid = (
        nu.to(unit='rad')
        - mu.to(unit='rad')
        - sc.array(
            dims=delta_grid.dims, values=delta_grid.values[::-1], unit=delta_grid.unit
        ).to(unit='rad')
        + 0.5 * Detector.nBlades * bladeAngle.to(unit='rad')
    )
    # TODO: If theta filtering is added, use it here
    # some filtering
    # theta_grid = theta_grid[theta_grid>=thetaMin]
    # theta_grid = theta_grid[theta_grid<=thetaMax]
    return theta_grid
