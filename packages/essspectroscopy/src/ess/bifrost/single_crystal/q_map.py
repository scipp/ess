# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Build a Q-map for single crystal diffraction."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import plopp as pp
import scipp as sc
from matplotlib.axes import Axes

from ess.spectroscopy.types import QDetector, RunType

from .types import (
    CountsWithQMapCoords,
    IntensityQparQperp,
    IntensitySampleRotation,
    QParallelBins,
    QPerpendicularBins,
    QProjection,
    QRange,
    SampleRotationBins,
)


def project_momentum_transfer(
    counts: QDetector[RunType],
    *,
    q_projection: QProjection,
) -> CountsWithQMapCoords[RunType]:
    """Project Q onto given axes and also compute the norm of Q."""
    transformed = counts.transform_coords(
        ['Q_parallel', 'Q_perpendicular', 'Q'],
        graph={
            'Q_parallel': _make_projection_kernel(q_projection.parallel),
            'Q_perpendicular': _make_projection_kernel(q_projection.perpendicular),
            'Q': lambda sample_table_momentum_transfer: sc.norm(
                sample_table_momentum_transfer
            ),
        },
        keep_inputs=False,
    )
    if transformed.bins is not None:
        transformed.bins.coords['a3'] = sc.bins_like(
            transformed, transformed.coords['a3']
        )
        transformed = transformed.bins.concat()

    return CountsWithQMapCoords[RunType](transformed)


def _make_projection_kernel(vector: sc.Variable) -> Callable[..., sc.Variable]:
    normal = vector / sc.norm(vector)

    def projection(sample_table_momentum_transfer: sc.Variable) -> sc.Variable:
        return sc.dot(sample_table_momentum_transfer, normal)

    return projection


def default_q_projection() -> QProjection:
    return QProjection(
        parallel=sc.vector(value=[0, 0, 1]), perpendicular=sc.vector(value=[1, 0, 0])
    )


def histogram_qparallel_qperpendicular(
    events: CountsWithQMapCoords[RunType],
    q_parallel_bins: QParallelBins,
    q_perpendicular_bins: QPerpendicularBins,
) -> IntensityQparQperp[RunType]:
    """Histogram the data in Q_parallel and Q_perpendicular"""
    return IntensityQparQperp[RunType](
        events.hist(Q_perpendicular=q_perpendicular_bins, Q_parallel=q_parallel_bins)
    )


def integrate_q(
    events: CountsWithQMapCoords[RunType],
    q_range: QRange,
    sample_rotation_bins: SampleRotationBins,
) -> IntensitySampleRotation[RunType]:
    """Integrate the data over |Q| and histogram in the sample rotation."""
    unit = events.bins.coords['Q'].unit
    lo = q_range[0].to(unit=unit)
    hi = q_range[1].to(unit=unit)
    sliced = events.bins['Q', lo:hi]
    try:
        return IntensitySampleRotation[RunType](sliced.hist(a3=sample_rotation_bins))
    except ValueError as err:
        if "empty data range" in err.args[0].lower():
            # This happens when the user selects an empty range Q range,
            # which is mainly a problem in the widget.
            return IntensitySampleRotation[RunType](_empty_angle_array_like(sliced))
        raise


def make_q_map(
    events: sc.DataArray,
    q_parallel_bins: int | sc.Variable,
    q_perpendicular_bins: int | sc.Variable,
    sample_rotation_bins: int | sc.Variable,
    *,
    circle_n_points: int = 100,
    roi_fill_alpha: float = 0.2,
    roi_fill_color: Any = 'C1',
    roi_line_color: Any = 'C1',
) -> Any:
    """Build a figure with a 2D Q-map and a 1D slice for a range of Q values."""
    import ipywidgets as ipw
    from plopp.widgets import Box

    def make_q_hist(da: sc.DataArray) -> sc.DataArray:
        return histogram_qparallel_qperpendicular(
            da, q_parallel_bins, q_perpendicular_bins
        ).rename(Q_perpendicular=r'$Q_{\perp}$', Q_parallel='$Q_{||}$')

    def make_q_slice(da: sc.DataArray, q_range: tuple[float, float]) -> sc.DataArray:
        unit = da.bins.coords['Q'].unit
        lo = q_range[0] * unit
        hi = q_range[1] * unit
        return integrate_q(da, (lo, hi), sample_rotation_bins)

    q_lo = events.bins.coords['Q'].min().value
    q_hi = events.bins.coords['Q'].max().value
    slider = ipw.FloatRangeSlider(
        value=(q_lo, q_hi), min=q_lo, max=q_hi, step=(q_hi - q_lo) / 100
    )

    slider_node = pp.widget_node(slider)
    input_node = pp.Node(events)
    q_map_node = pp.Node(make_q_hist, input_node)
    q_slice_node = pp.Node(make_q_slice, input_node, slider_node)
    q_map_fig = pp.imagefigure(q_map_node, aspect='equal', norm='log')
    q_slice_fig = pp.linefigure(q_slice_node)

    roi_path = _ROICirclePath(q_lo, q_hi, circle_n_points)
    roi_circle = _ROICircle(
        roi_path,
        q_map_fig.ax,
        fill_alpha=roi_fill_alpha,
        fill_color=roi_fill_color,
        line_color=roi_line_color,
    )

    def update_roi(q_range: tuple[float, float]) -> None:
        roi_circle.set(*q_range)
        q_map_fig.canvas.draw()

    # Create a view without any output so that `update_roi` gets called
    # when the slider is moved. See https://github.com/scipp/plopp/pull/496
    pp.View(pp.Node(update_roi, slider_node))
    return Box([[q_map_fig], [q_slice_fig], [slider]])


class _ROICirclePath:
    # The path is encoded as an array xy with shape (n_points * 2 + 1, 2).
    # Where xy[:, 0] is the x coordinate and xy[:, 1] is the y coordinate.
    # xy[:n_points] is the outer circle, xy[n_points:-1] is the inner circle.
    # xy[-1] is the same as xy[0]; we close the path to be compatible with MPL.

    def __init__(self, r_inner: float, r_outer: float, n_points: int) -> None:
        self._r_inner = r_inner
        self._r_outer = r_outer
        self._n_points = n_points

        self._angles = np.linspace(0, 2 * np.pi, n_points, endpoint=True)
        # 2*n_points for the inner and outer circle,
        # +1 to repeat the first point at the end.
        self._xy = np.zeros((n_points * 2 + 1, 2), dtype=float)

        self.set_inner(r_inner)
        self.set_outer(r_outer)

    @property
    def r_inner(self) -> float:
        return self._r_inner

    @property
    def r_outer(self) -> float:
        return self._r_outer

    @property
    def closed_xy(self) -> npt.NDArray[float]:
        # includes the last point
        return self._xy

    @property
    def open_xy(self) -> npt.NDArray[float]:
        # does not include the last point
        return self._xy[:-1]

    @property
    def inner(self) -> npt.NDArray[float]:
        # does not include the last point
        return self._xy[self._n_points : -1]

    @property
    def outer(self) -> npt.NDArray[float]:
        # does not include the last point
        return self._xy[: self._n_points]

    def set_inner(self, r: float) -> None:
        # reverse=True makes MPL render this circle as a cutout.
        self._r_inner = r
        x, y = self.inner.T
        self._set_circle(x, y, r, reverse=True)

    def set_outer(self, r: float) -> None:
        self._r_outer = r
        x, y = self.outer.T
        self._set_circle(x, y, r, reverse=False)
        self._xy[-1] = self._xy[0]

    def _set_circle(
        self, x: npt.NDArray[float], y: npt.NDArray[float], r: float, reverse: bool
    ) -> None:
        if reverse:
            angles = self._angles[::-1]
        else:
            angles = self._angles

        np.cos(angles, out=x)
        x *= r

        np.sin(angles, out=y)
        y *= r


class _ROICircle:
    def __init__(
        self,
        path: _ROICirclePath,
        ax: Axes,
        *,
        fill_alpha: float = 0.2,
        fill_color: Any = 'C1',
        line_color: Any = 'C1',
    ) -> None:
        self._path = path
        self._fill = ax.fill(
            *self._path.open_xy.T, fill_color, alpha=fill_alpha, zorder=-1
        )[0]
        self._inner = ax.plot(*self._path.inner.T, c=line_color, zorder=2)[0]
        self._outer = ax.plot(*self._path.outer.T, c=line_color, zorder=2)[0]

    def set(self, r_inner: float, r_outer: float) -> None:
        if r_inner != self._path.r_inner:
            self._path.set_inner(r_inner)
            self._inner.set_data(*self._path.inner.T)
        if r_outer != self._path.r_outer:
            self._path.set_outer(r_outer)
            self._outer.set_data(*self._path.outer.T)
        self._fill.set_xy(self._path.closed_xy)


def _empty_angle_array_like(
    reference: sc.DataArray, *, dim: str = 'sample_rotation'
) -> sc.DataArray:
    a3_lo = reference.bins.coords[dim].min()
    a3_hi = reference.bins.coords[dim].max()
    return sc.DataArray(
        sc.array(
            dims=[dim],
            values=[0, 0],
            variances=[0, 0],
            unit=reference.unit,
            dtype=reference.dtype,
        ),
        coords={
            dim: sc.array(
                dims=[dim], values=[a3_lo.value, a3_hi.value], unit=a3_lo.unit
            )
        },
    )


providers = (
    default_q_projection,
    integrate_q,
    histogram_qparallel_qperpendicular,
    project_momentum_transfer,
)
