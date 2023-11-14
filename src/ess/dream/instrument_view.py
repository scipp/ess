# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from html import escape
from typing import Union

import plopp as pp
import scipp as sc


def _to_data_group(data: Union[sc.DataArray, sc.DataGroup, dict]) -> sc.DataGroup:
    if isinstance(data, sc.DataArray):
        data = sc.DataGroup({data.name or 'data': data})
    elif isinstance(data, dict):
        data = sc.DataGroup(data)
    return data


@pp.node
def slice_range(da, trunc_range):
    min_tr, max_tr = trunc_range
    return da['tof', min_tr:max_tr].sum('tof')


@pp.node
def post_process(da, dim):
    dims = list(da.dims)
    if dim is not None:
        dims.remove(dim)
    out = da.flatten(dims=dims, to='pixel')
    sel = sc.isfinite(out.coords['x'])
    return out[sel]


class InstrumentView:
    def __init__(self, data, dim=None, pixel_size=None, **kwargs):
        import ipywidgets as ipw

        self.data = _to_data_group(data)

        self.post_process_nodes = {
            key: post_process(da, dim) for key, da in self.data.items()
        }

        self.children = []

        if dim is not None:
            # Once https://github.com/scipp/plopp/issues/277 is resolved, we can
            # use Plopp's range slicer so that the value of the coordinates are
            # displayed next to the slider, instead of the raw indices.
            self.slider = ipw.IntRangeSlider(
                value=[0, self.data.sizes[dim] - 1],
                max=self.data.sizes[dim] - 1,
                description=dim,
                layout={'width': '700px'},
                continuous_update=False,
            )
            self.slider_node = pp.widget_node(self.slider)
            self.slice_nodes = {
                key: slice_range(n, trunc_range=self.slider_node)
                for key, n in self.post_process_nodes.items()
            }
            to_scatter = self.slice_nodes
            self.children.append(self.slider)
        else:
            self.slice_nodes = self.post_process_nodes
            to_scatter = self.post_process_nodes

        self.scatter = pp.scatter3d(
            to_scatter,
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


def instrument_view(data, dim=None, pixel_size=None, **kwargs):
    """
    Three-dimensional visualization of the DREAM instrument.

    Parameters
    ----------
    data:
        Data to visualize.
    dim:
        Dimension to use for the slider. No slider will be shown if this is None.
    pixel_size:
        Size of the pixels.
    **kwargs:
        Additional arguments are forwarded to the scatter3d figure
        (see https://scipp.github.io/plopp/reference/generated/plopp.scatter3d.html).
    """
    from plopp.widgets import Box

    view = InstrumentView(data, dim=dim, pixel_size=pixel_size, **kwargs)
    return Box(view.children)
