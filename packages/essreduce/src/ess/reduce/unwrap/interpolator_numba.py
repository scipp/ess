# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
from numba import njit, prange


@njit(boundscheck=False, cache=True)
def _locate(grid: np.ndarray, value: float, one_over_step: float, uniform: bool) -> int:
    """Index of the cell of ``grid`` containing ``value``.

    ``value`` is assumed to lie within the grid. A value on the upper edge is
    assigned to the last cell, so that the caller can index ``grid[i + 1]``.
    """
    n = len(grid)
    if value == grid[n - 1]:
        return n - 2
    if uniform:
        return int((value - grid[0]) * one_over_step)
    return np.searchsorted(grid, value, side='right') - 1


@njit(boundscheck=False, cache=True, fastmath=False, parallel=True)
def interpolate(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    xoffset: np.ndarray | None,
    deltax: float,
    fill_value: float,
    x_uniform: bool,
    y_uniform: bool,
    out: np.ndarray,
):
    """
    Linear interpolation of data on a 2D rectilinear grid.

    Parameters
    ----------
    x:
        1D array of grid points along the x-axis (size nx), strictly increasing.
    y:
        1D array of grid points along the y-axis (size ny), strictly increasing.
    values:
        2D array of values on the grid. The shape must be (ny, nx).
    xp:
        1D array of x-coordinates where to interpolate (size N).
    yp:
        1D array of y-coordinates where to interpolate (size N).
    xoffset:
        1D array of integer offsets to apply to the x-coordinates (size N).
    deltax:
        Multiplier to apply to the integer offsets (i.e. the step size).
    fill_value:
        Value to use for points outside of the grid.
    x_uniform:
        Whether ``x`` is equally spaced, which allows the containing cell to be
        computed by division instead of searched for. See :class:`Interpolator`.
    y_uniform:
        Whether ``y`` is equally spaced.
    out:
        1D array where the interpolated values will be stored (size N).
    """
    if not (len(xp) == len(yp) == len(out)):
        raise ValueError("Interpolator: all input arrays must have the same size.")

    nx = len(x)
    ny = len(y)
    npoints = len(xp)
    xmin = x[0]
    xmax = x[nx - 1]
    ymin = y[0]
    ymax = y[ny - 1]

    one_over_dx = 1.0 / (x[1] - xmin)
    one_over_dy = 1.0 / (y[1] - ymin)
    both_uniform = x_uniform and y_uniform
    norm = one_over_dx * one_over_dy

    for i in prange(npoints):
        xx = xp[i] + (xoffset[i] * deltax if xoffset is not None else 0.0)
        yy = yp[i]

        if (xx < xmin) or (xx > xmax) or (yy < ymin) or (yy > ymax):
            out[i] = fill_value

        else:
            ix = _locate(x, xx, one_over_dx, x_uniform)
            iy = _locate(y, yy, one_over_dy, y_uniform)

            x1 = x[ix]
            x2 = x[ix + 1]
            y1 = y[iy]
            y2 = y[iy + 1]

            a11 = values[iy, ix]
            a21 = values[iy, ix + 1]
            a12 = values[iy + 1, ix]
            a22 = values[iy + 1, ix + 1]

            x2mxx = x2 - xx
            xxmx1 = xx - x1

            # A uniform grid normalizes by the same cell area everywhere, which
            # is worth hoisting out of the loop; a rectilinear one does not.
            cell = norm if both_uniform else 1.0 / ((x2 - x1) * (y2 - y1))

            out[i] = (
                (y2 - yy) * (x2mxx * a11 + xxmx1 * a21)
                + (yy - y1) * (x2mxx * a12 + xxmx1 * a22)
            ) * cell


def _is_uniform(grid: np.ndarray) -> bool:
    """Whether ``grid`` is equally spaced, to within floating-point noise."""
    if len(grid) < 3:
        return True
    steps = np.diff(grid)
    return bool(np.allclose(steps, steps[0], rtol=1.0e-9, atol=0.0))


class Interpolator:
    def __init__(
        self,
        time_edges: np.ndarray,
        distance_edges: np.ndarray,
        values: np.ndarray,
        fill_value: float = np.nan,
    ):
        """
        Interpolator for 2D rectilinear grid data (Numba implementation).

        The axes need not be equally spaced: a lookup table may sample distance
        densely where components sit and not at all in between. Uniformity is
        detected here, once, because it decides how the containing cell is
        found — by division for a uniform axis, by binary search otherwise —
        and that is a per-point cost in the interpolation loop.

        Parameters
        ----------
        time_edges:
            1D array of time grid points, strictly increasing.
        distance_edges:
            1D array of distance grid points, strictly increasing.
        values:
            2D array of values on the grid. The shape must be (ny, nx).
        fill_value:
            Value to use for points outside of the grid.
        """
        for name, grid in (
            ('time_edges', time_edges),
            ('distance_edges', distance_edges),
        ):
            if len(grid) < 2:
                raise ValueError(
                    f"Interpolator: {name} has {len(grid)} point(s); at least two "
                    "are needed to bracket a value."
                )
        self.time_edges = time_edges
        self.distance_edges = distance_edges
        self.values = values
        self.fill_value = fill_value
        self.time_uniform = _is_uniform(time_edges)
        self.distance_uniform = _is_uniform(distance_edges)

    def __call__(
        self,
        times: np.ndarray,
        distances: np.ndarray,
        pulse_period: float = 0.0,
        pulse_index: np.ndarray | None = None,
    ) -> np.ndarray:
        out = np.empty_like(times)
        interpolate(
            x=self.time_edges,
            y=self.distance_edges,
            values=self.values,
            xp=times,
            yp=distances,
            xoffset=pulse_index,
            deltax=pulse_period,
            fill_value=self.fill_value,
            x_uniform=self.time_uniform,
            y_uniform=self.distance_uniform,
            out=out,
        )
        return out
