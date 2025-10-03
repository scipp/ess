# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Build a Q-map for single crystal diffraction."""

from collections.abc import Callable

import plopp as pp
import scipp as sc
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from plopp.widgets import Box

from ess.spectroscopy.types import DetectorCountsWithQ, RunType

from .types import CountsWithQMapCoords, QProjection


def project_momentum_transfer(
    counts: DetectorCountsWithQ[RunType],
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
    def projection(sample_table_momentum_transfer: sc.Variable) -> sc.Variable:
        return sc.dot(sample_table_momentum_transfer, vector / sc.norm(vector))

    return projection


def default_q_projection() -> QProjection:
    return QProjection(
        parallel=sc.vector(value=[0, 0, 1]), perpendicular=sc.vector(value=[1, 0, 0])
    )


def make_q_map(
    events: sc.DataArray,
    q_parallel_bins: int | sc.Variable,
    q_perpendicular_bins: int | sc.Variable,
    sample_rotation_bins: int | sc.Variable,
) -> Box:
    """Build a figure with a 2D Q-map and a 1D slice for a range of Q values."""
    import ipywidgets as ipw

    def make_q_hist(da: sc.DataArray) -> sc.DataArray:
        return da.hist(
            Q_perpendicular=q_perpendicular_bins, Q_parallel=q_parallel_bins
        ).rename(Q_perpendicular=r'$Q_{\perp}$', Q_parallel='$Q_{||}$')

    def make_q_slice(da: sc.DataArray, q_range: tuple[float, float]) -> sc.DataArray:
        unit = da.bins.coords['Q'].unit
        lo = q_range[0] * unit
        hi = q_range[1] * unit
        sliced = da.bins['Q', lo:hi]
        try:
            return sliced.hist(a3=sample_rotation_bins).rename(a3='sample_rotation')
        except ValueError as err:
            if "empty data range" in err.args[0].lower():
                return _empty_angle_array_like(sliced)
            raise

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

    lo_circle = _draw_circle(q_map_fig.ax, q_lo)
    hi_circle = _draw_circle(q_map_fig.ax, q_hi)

    def update_circle(q_range: tuple[float, float]) -> None:
        lo_circle.radius, hi_circle.radius = q_range
        q_map_fig.fig.canvas.draw_idle()

    pp.View(pp.Node(update_circle, slider_node))

    return Box([[q_map_fig], [q_slice_fig], [slider]])


def _draw_circle(ax: Axes, radius: float) -> Circle:
    circle = Circle((0, 0), radius, edgecolor='C1', facecolor='none', linewidth=1.5)
    ax.add_artist(circle)
    return circle


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


providers = (default_q_projection, project_momentum_transfer)
