# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any, Optional, Union

import plopp as pp
import scipp as sc

if TYPE_CHECKING:
    try:
        from plopp.widgets import Box
    except ModuleNotFoundError:
        Box = object


def instrument_view(
    data: Union[sc.DataArray, sc.DataGroup, dict],
    dim: Optional[str] = None,
    pixel_size: Optional[Union[float, sc.Variable]] = None,
    **kwargs: Any,
) -> Box:
    """
    Three-dimensional visualization of the DREAM instrument.
    The instrument view is capable of slicing the input data with a slider widget along
    a dimension (e.g. ``tof``) by using the ``dim`` argument.
    It will also generate checkboxes to hide/show the different modules that make up
    the DREAM detectors.

    Parameters
    ----------
    data:
        Data to visualize. The data can be a single detector module (``DataArray``),
        or a group of detector modules (``dict`` or ``DataGroup``).
        The data must contain a ``position`` coordinate.
    dim:
        Dimension to use for the slider. No slider will be shown if this is None.
    pixel_size:
        Size of the pixels.
    **kwargs:
        Additional arguments are forwarded to the scatter3d figure
        (see https://scipp.github.io/plopp/generated/plopp.scatter3d.html).
    """
    from plopp.widgets import Box

    view = InstrumentView(data, dim=dim, pixel_size=pixel_size, **kwargs)
    return Box(view.children)


def _to_data_group(data: Union[sc.DataArray, sc.DataGroup, dict]) -> sc.DataGroup:
    if isinstance(data, sc.DataArray):
        data = sc.DataGroup({data.name or 'data': data})
    elif isinstance(data, dict):
        data = sc.DataGroup(data)
    return data


@pp.node
def _pre_process(da: sc.DataArray, dim: str) -> sc.DataArray:
    dims = list(da.dims)
    if dim is not None:
        dims.remove(dim)
    out = da.flatten(dims=dims, to='pixel')
    sel = sc.isfinite(out.coords['position'])
    return out[sel]


class InstrumentView:
    """Instrument view for DREAM."""

    def __init__(
        self,
        data: Union[sc.DataArray, sc.DataGroup, dict],
        dim: Optional[str] = None,
        pixel_size: Optional[Union[float, sc.Variable]] = None,
        **kwargs,
    ):
        from plopp.widgets import SliceWidget, slice_dims

        self.data = _to_data_group(data)
        self.pre_process_nodes = {
            key: _pre_process(da, dim) for key, da in self.data.items()
        }

        self.children = []

        if dim is not None:
            self.slider = SliceWidget(next(iter(self.data.values())), dims=[dim])
            self.slider.controls[dim]['slider'].layout = {'width': '600px'}
            self.slider_node = pp.widget_node(self.slider)
            self.slice_nodes = {
                key: slice_dims(n, self.slider_node)
                for key, n in self.pre_process_nodes.items()
            }
            to_scatter = self.slice_nodes
            self.children.append(self.slider)
        else:
            self.slice_nodes = self.pre_process_nodes
            to_scatter = self.pre_process_nodes

        self.scatter = pp.scatter3d(
            to_scatter,
            pos='position',
            pixel_size=1.0 * sc.Unit('cm') if pixel_size is None else pixel_size,
            **kwargs,
        )

        self.children.insert(0, self.scatter)

        if len(self.data) > 1:
            self._add_module_control()

    def _add_module_control(self):
        import ipywidgets as ipw

        self.fig = self.scatter[0]
        self.cutting_tool = self.scatter[1]
        self.artist_mapping = {
            name: key for name, key in zip(self.data.keys(), self.fig.artists.keys())
        }
        self.checkboxes = {
            key: ipw.Checkbox(
                value=True,
                description=f"{escape(key)}",
                indent=False,
                layout={"width": "initial"},
            )
            for key in self.data
        }

        self.modules_widget = ipw.HBox(
            [ipw.HTML(value="Modules: &nbsp;&nbsp;&nbsp;&nbsp;")]
            + list(self.checkboxes.values())
        )
        for key, ch in self.checkboxes.items():
            ch.key = key
            ch.observe(self._check_visibility, names='value')
        self.cutting_tool.cut_x.button.observe(self._check_visibility, names="value")
        self.cutting_tool.cut_y.button.observe(self._check_visibility, names="value")
        self.cutting_tool.cut_z.button.observe(self._check_visibility, names="value")
        self.children.insert(0, self.modules_widget)

    def _check_visibility(self, _):
        # Note that this brute force method of looping over all artists is not optimal
        # but it is non-invasive in the sense that it does not require changes to the
        # plopp code. If performance becomes an issue, we will consider a different
        # approach.
        for name, ch in self.checkboxes.items():
            key = self.artist_mapping[name]
            val = ch.value
            self.fig.artists[key].points.visible = val
            for c in "xyz":
                cut_nodes = getattr(self.cutting_tool, f'cut_{c}').select_nodes
                if key in cut_nodes:
                    self.fig.artists[cut_nodes[key].id].points.visible = val
