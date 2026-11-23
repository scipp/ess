# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Viewer for high-dimensional data."""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from typing import Any

import ipywidgets as ipw
import numpy as np
import plopp as pp
import scipp as sc
from plopp.core.utils import coord_as_bin_edges, scalar_to_string


class FlatDataViewer(ipw.VBox):
    """A viewer that flattens and slices all dimensions to get a 2D plot."""

    def __init__(
        self,
        data: Mapping[str, sc.DataArray] | sc.DataArray,
        horizontal: Iterable[str] | dict[str, Iterable[str]],
        vertical: Iterable[str] | dict[str, Iterable[str]],
        *,
        cbar: bool = True,
        rasterized: bool = True,
        separators: str | None = None,
        **fig_kwargs: Any,
    ) -> None:
        """Create flat data view.

        Dims that are not in a bank are ignored for that bank.
        Dims on neither axis are sliced.

        Parameters
        ----------
        data:
            Data to show. A mapping (e.g., a data group) adds buttons for selecting
            a bank. The banks may have different dims
            but must all have the same unit and no empty dimensions.
        horizontal:
            Dims to display on the horizontal axis.
            Order is outermost to innermost (last dim varies fastest).
            If a dict, the keys are bank names and the values are dims to display
            on the horizontal axis for that bank.
        vertical:
            Dims to display on the vertical axis.
            Order is outermost to innermost (last dim varies fastest).
            If a dict, the keys are bank names and the values are dims to display
            on the vertical axis for that bank.
        cbar:
            Whether to show a colorbar.
        rasterized:
            If ``True``, the figure is rasterized which improves rendering
            speed but reduces resolution.
        separators:
            If not ``None``, draw lines along the bin edges of the slowest
            (first-listed) dim on each axis. The value of this argument is the
            line color.
        **fig_kwargs:
            Additional arguments passed to :func:`plopp.imagefigure`.
        """
        data_dict, horizontal_dims, vertical_dims = _build_dim_layout(
            data, horizontal, vertical
        )
        box = _build_widget(
            data_dict,
            horizontal_dims,
            vertical_dims,
            separators=separators,
            fig_kwargs={"cbar": cbar, "rasterized": rasterized} | fig_kwargs,
        )
        super().__init__([box])


def _build_widget(
    data: dict[str, sc.DataArray],
    horizontal: dict[str, tuple[str, ...]],
    vertical: dict[str, tuple[str, ...]],
    separators: str | None,
    fig_kwargs: dict[str, Any],
) -> ipw.VBox:
    first_name = next(iter(data))
    first_data = next(iter(data.values()))
    slicers = {
        first_name: _make_slicer(
            first_data,
            horizontal[first_name],
            vertical[first_name],
            separators,
            fig_kwargs,
        )
    }

    figure_box = ipw.VBox([slicers[first_name]])
    if len(data) == 1 and next(iter(data)) == "":
        return figure_box

    def observer(change: dict[str, object]) -> None:
        name = change["new"]
        if (slicer := slicers.get(name)) is None:
            slicer = slicers[name] = _make_slicer(
                data[name], horizontal[name], vertical[name], separators, fig_kwargs
            )
        figure_box.children = [slicer]

    selector = ipw.ToggleButtons(options=list(data))
    selector.observe(observer, names="value")
    return ipw.VBox([selector, figure_box])


def _make_slicer(
    data: sc.DataArray,
    horizontal: tuple[str, ...],
    vertical: tuple[str, ...],
    separators: str | None,
    fig_kwargs: dict[str, Any],
):
    flattened = _flatten(data, horizontal, vertical)
    [*_, v, h] = flattened.dims
    slicer = pp.slicer(flattened, keep=[v, h], **fig_kwargs)
    [plotted_node] = slicer.graph_nodes.values()
    horizontal_sizes = {dim: data.sizes[dim] for dim in horizontal}
    vertical_sizes = {dim: data.sizes[dim] for dim in vertical}
    slicer.ax.format_coord = _coord_formatter(
        plotted_node, horizontal_sizes, vertical_sizes, h, v, data.coords
    )
    if separators is not None:
        plotted = plotted_node()
        for x in _separator_positions(plotted, h, horizontal_sizes):
            slicer.ax.axvline(x, color=separators, linewidth=1)
        for y in _separator_positions(plotted, v, vertical_sizes):
            slicer.ax.axhline(y, color=separators, linewidth=1)
    return slicer


def _separator_positions(
    data: sc.DataArray, flat_dim: str, sizes: dict[str, int]
) -> np.ndarray:
    """Interior bin edges of the slowest dim in a flattened dim."""
    edges = coord_as_bin_edges(data, flat_dim).values
    n_slowest = next(iter(sizes.values()))
    block = (len(edges) - 1) // n_slowest  # product of the faster dims' sizes
    return edges[block:-1:block]


def _build_dim_layout(
    data: Mapping[str, sc.DataArray] | sc.DataArray,
    horizontal: Iterable[str] | dict[str, Iterable[str]],
    vertical: Iterable[str] | dict[str, Iterable[str]],
) -> tuple[
    dict[str, sc.DataArray],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    data_dict, h_dict, v_dict = _normalize_dim_args(data, horizontal, vertical)
    horizontal_dims = {}
    vertical_dims = {}
    for name, da in data_dict.items():
        h = h_dict[name]
        v = v_dict[name]
        if duplicates := sorted(
            dim for dim, count in Counter([*h, *v]).items() if count > 1
        ):
            raise ValueError(
                f"Dims {duplicates} are listed more than once in "
                "`horizontal` and `vertical`."
            )
        horizontal_dims[name] = tuple(dim for dim in h if dim in da.dims)
        vertical_dims[name] = tuple(dim for dim in v if dim in da.dims)
    return data_dict, horizontal_dims, vertical_dims


def _normalize_dim_args(
    data: Mapping[str, sc.DataArray] | sc.DataArray,
    horizontal: Iterable[str] | dict[str, Iterable[str]],
    vertical: Iterable[str] | dict[str, Iterable[str]],
) -> tuple[
    dict[str, sc.DataArray],
    Mapping[str, tuple[str, ...]],
    Mapping[str, tuple[str, ...]],
]:
    if isinstance(data, sc.DataArray):
        for name, x in (("horizontal", horizontal), ("vertical", vertical)):
            if isinstance(x, Mapping):
                raise TypeError(
                    f"The data is a single data array but `{name}` is a mapping."
                    "Pass a single list of strings instead."
                )
        return {"": data}, {"": tuple(horizontal)}, {"": tuple(vertical)}

    return data, _as_default_mapping(horizontal), _as_default_mapping(vertical)


def _as_default_mapping(
    x: Iterable[str] | dict[str, Iterable[str]],
) -> Mapping[str, tuple[str, ...]]:
    if isinstance(x, Mapping):
        return x
    values = tuple(x)  # build tuple early so allow multiple iterations later
    return defaultdict(lambda: values)


def _flatten(
    da: sc.DataArray, horizontal: tuple[str, ...], vertical: tuple[str, ...]
) -> sc.DataArray:
    sliced = [dim for dim in da.dims if dim not in [*horizontal, *vertical]]
    return (
        da.transpose([*sliced, *vertical, *horizontal])
        .flatten(horizontal, to=" * ".join(horizontal))
        .flatten(vertical, to=" * ".join(vertical))
    )


def _coord_formatter(
    plotted_node: Callable[[], sc.DataArray],
    horizontal: tuple[str, ...],
    vertical: tuple[str, ...],
    horizontal_dim: str,
    vertical_dim: str,
    coords: Mapping[str, sc.Variable],
):
    n_vertical, n_horizontal = (
        reduce(lambda a, b: a * b, dims)
        for dims in [vertical.values(), horizontal.values()]
    )

    def unwrap(x: int, y: int) -> dict[str, int]:
        return {**_unwrap_flat_index(x, horizontal), **_unwrap_flat_index(y, vertical)}

    def format_coord(x: float, y: float) -> str:
        """Show the indices of all flattened dims and the value under the cursor."""
        data = plotted_node()
        i = _bin_index(data.coords[horizontal_dim], n_horizontal, x)
        j = _bin_index(data.coords[vertical_dim], n_vertical, y)
        if i is None or j is None:
            return ""

        coord_values = (
            (
                dim,
                coord[index]
                if (coord := coords.get(dim, None)) is not None
                else sc.index(index),
            )
            for dim, index in unwrap(i, j).items()
        )
        formatted_coords = ", ".join(
            f"{dim}: {scalar_to_string(value)}" for dim, value in coord_values
        )
        value = scalar_to_string(data.data[vertical_dim, j][horizontal_dim, i])
        return f"{{{formatted_coords}}}: {value}"

    return format_coord


def _bin_index(coord: sc.Variable, size: int, pos: float) -> int | None:
    """Return the index of the bin containing ``pos`` or ``None`` if out of range."""
    edges = np.asarray(coord.values, dtype=float)
    if len(edges) == size:
        edges = _centers_to_edges(edges)
    descending = edges[0] > edges[-1]
    if descending:
        edges = edges[::-1]
    index = int(np.searchsorted(edges, pos, side="right")) - 1
    if not 0 <= index < size:
        return None
    return size - 1 - index if descending else index


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mid = (centers[1:] + centers[:-1]) / 2
    return np.concatenate([[2 * centers[0] - mid[0]], mid, [2 * centers[-1] - mid[-1]]])


def _unwrap_flat_index(index: int, sizes: dict[str, int]) -> dict[str, int]:
    indices = []
    for dim, size in reversed(sizes.items()):
        index, offset = divmod(index, size)
        indices.append((dim, offset))
    return dict(reversed(indices))  # Restore the input order.
