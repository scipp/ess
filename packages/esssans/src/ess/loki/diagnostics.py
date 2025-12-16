# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Detector diagnostics for LOKI."""

from __future__ import annotations

import ipywidgets as ipw
import matplotlib.pyplot as plt
import plopp as pp
import scipp as sc


def _slice_or_sum(
    da: sc.DataGroup, layer_ind: int, layer_sum: bool, straw_ind: int, straw_sum: bool
) -> sc.DataGroup:
    out = da.copy(deep=False)
    if layer_sum:
        out = out.sum('layer')
    else:
        out = out["layer", layer_ind - 1]
    if straw_sum:
        out = out.sum('straw')
    else:
        out = out["straw", straw_ind - 1]
    return out


class LokiBankViewer(ipw.VBox):
    def __init__(self, data: sc.DataGroup, **kwargs):
        """Widget to view LOKI detector banks.

        Parameters
        ----------
        data:
            DataGroup containing LOKI detector banks.
        kwargs:
            Additional arguments are forwarded to the plotting function.
        """
        self.data = data
        self._lock = False

        self.layer_slider = ipw.IntSlider(
            min=1, max=4, description="Layer", style={"description_width": 'initial'}
        )
        self.layer_sum = ipw.Checkbox(
            description="Sum", value=False, indent=False, layout={"width": "initial"}
        )

        self.layer_ind_node = pp.widget_node(self.layer_slider)
        self.layer_sum_node = pp.widget_node(self.layer_sum)

        self.straw_slider = ipw.IntSlider(
            min=1, max=7, description="Straw", style={"description_width": 'initial'}
        )
        self.straw_sum = ipw.Checkbox(
            description="Sum", value=False, indent=False, layout={"width": "initial"}
        )
        self.straw_ind_node = pp.widget_node(self.straw_slider)
        self.straw_sum_node = pp.widget_node(self.straw_sum)

        self.layer_link = ipw.jslink(
            (self.layer_sum, 'value'), (self.layer_slider, 'disabled')
        )
        self.straw_link = ipw.jslink(
            (self.straw_sum, 'value'), (self.straw_slider, 'disabled')
        )

        self.slice_node = pp.Node(
            _slice_or_sum,
            da=self.data,
            layer_ind=self.layer_ind_node,
            layer_sum=self.layer_sum_node,
            straw_ind=self.straw_ind_node,
            straw_sum=self.straw_sum_node,
        )

        with plt.ioff():
            self.figure, axs = plt.subplots(3, 3, figsize=(12, 9))

        self.figs = []
        for i, ax in enumerate(axs.flatten()):
            bank = f"loki_detector_{i}"
            self.figs.append(
                pp.plot(
                    pp.Node(lambda da, key: da[key], da=self.slice_node, key=bank),
                    ax=ax,
                    title=bank,
                    **kwargs,
                )
            )
        self.figure.canvas.header_visible = False

        # Create color map controls
        self.cmap_vmin = ipw.Text(
            description='Min:',
            layout={'width': '130px'},
            style={"description_width": 'initial'},
        )
        self.cmap_vmax = ipw.Text(
            description='Max:',
            layout={'width': '130px'},
            style={"description_width": 'initial'},
        )
        self.cmap_vmin.observe(self.update_cmin, names='value')
        self.cmap_vmax.observe(self.update_cmax, names='value')

        self.log_button = ipw.ToggleButton(
            description="log",
            layout={"width": "40px"},
            value=True if kwargs.get('norm', 'linear') == 'log' else False,
        )
        self.log_button.observe(self.toggle_log, names="value")

        layer_box = ipw.HBox(
            [self.layer_slider, self.layer_sum],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )
        straw_box = ipw.HBox(
            [self.straw_slider, self.straw_sum],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )
        space = ipw.HTML('<div style="width: 20px;"></div>')

        super().__init__(
            [
                ipw.HBox(
                    [
                        layer_box,
                        space,
                        straw_box,
                        space,
                        ipw.Label("Colorscale:"),
                        self.cmap_vmin,
                        self.cmap_vmax,
                        self.log_button,
                    ]
                ),
                self.figure.canvas,
            ]
        )

    def toggle_log(self, change: dict) -> None:
        self.reset_cmap_range()
        for f in self.figs:
            f.view.colormapper.norm = "log" if change["new"] else "linear"

    def _update_min_or_max(self, change: dict, bound: str) -> None:
        try:
            new = float(change["new"])
            if self.log_button.value and new <= 0:
                return
        except ValueError:
            new = None

        for f in self.figs:
            if new is None:
                getattr(f.view.colormapper, f"_c{bound}").pop('user', None)
            else:
                setattr(f.view.colormapper, f"c{bound}", new)
            if not self._lock:
                f.view.colormapper.autoscale()
        if not self._lock:
            self.figure.canvas.draw_idle()

    def update_cmin(self, change: dict) -> None:
        self._update_min_or_max(change, 'min')

    def update_cmax(self, change: dict) -> None:
        self._update_min_or_max(change, 'max')

    def reset_cmap_range(self, _=None) -> None:
        self._lock = True
        self.cmap_vmin.value = ''
        self.cmap_vmax.value = ''
        self._lock = False


def _to_data_group(data: sc.DataArray | sc.DataGroup | dict) -> sc.DataGroup:
    if isinstance(data, sc.DataArray):
        data = sc.DataGroup({data.name or "data": data})
    elif isinstance(data, dict):
        data = sc.DataGroup(data)
    return data


@pp.node
def _pre_process(da: sc.DataArray, dim: str) -> sc.DataArray:
    dims = list(da.dims)
    if dim is not None:
        dims.remove(dim)
    out = da.flatten(dims=dims, to="pixel")
    sel = sc.isfinite(out.coords["position"])
    return out[sel]


class InstrumentView(ipw.VBox):
    """
    Three-dimensional visualization of the Loki instrument.
    The instrument view is capable of slicing the input data with a slider widget along
    a dimension (e.g. ``tof``) by using the ``dim`` argument.
    It will also generate toggle buttons to hide/show the different modules that make up
    the Loki detectors.

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
    autoscale:
        If ``True``, the color scale will be automatically adjusted to the data as it
        gets updated. This can be somewhat expensive with many pixels, so it is set to
        ``False`` by default.
    **kwargs:
        Additional arguments are forwarded to the scatter3d figure
        (see https://scipp.github.io/plopp/generated/plopp.scatter3d.html).
    """

    def __init__(
        self,
        data: sc.DataArray | sc.DataGroup | dict,
        dim: str | None = None,
        pixel_size: float | sc.Variable | None = None,
        **kwargs,
    ):
        from plopp.widgets import SliceWidget, slice_dims

        if dim and isinstance(data, sc.DataArray) and dim in data.dims[:-1]:
            data = data.transpose([d for d in data.dims if d != dim] + [dim])

        if dim and isinstance(data, sc.DataGroup):
            data = data.copy(deep=False)
            for k, v in data.items():
                if dim in v.dims[:-1]:
                    data[k] = v.transpose([d for d in v.dims if d != dim] + [dim])

        self.data = _to_data_group(data)
        self.pre_process_nodes = {
            key: _pre_process(da, dim) for key, da in self.data.items()
        }

        self._widgets = []

        if dim is not None:
            self.slider = SliceWidget(next(iter(self.data.values())), dims=[dim])
            self.slider.controls[dim].slider.layout = {"width": "600px"}
            self.slider_node = pp.widget_node(self.slider)
            self.slice_nodes = {
                key: slice_dims(n, self.slider_node)
                for key, n in self.pre_process_nodes.items()
            }
            to_scatter = self.slice_nodes
            self._widgets.append(self.slider)
        else:
            self.slice_nodes = self.pre_process_nodes
            to_scatter = self.pre_process_nodes

        kwargs.setdefault('cbar', True)
        self.fig = pp.scatter3d(
            to_scatter,
            pos="position",
            pixel_size=1.0 * sc.Unit("cm") if pixel_size is None else pixel_size,
            **kwargs,
        )

        self._widgets.insert(0, self.fig)

        if len(self.data) > 1:
            self._add_module_control()

        super().__init__(self._widgets)

    def _add_module_control(self):
        import ipywidgets as ipw

        self.cutting_tool = self.fig.bottom_bar[0]
        self._node_backup = list(self.cutting_tool._original_nodes)
        self.artist_mapping = dict(
            zip(self.data.keys(), self.fig.artists.keys(), strict=True)
        )
        self.buttons = {
            key: ipw.ToggleButton(
                value=True, description=f"{i}", layout={"width": "initial"}
            )
            for i, key in enumerate(self.data)
        }

        self.modules_widget = ipw.HBox(
            [
                ipw.HTML(value="Detector banks: &nbsp;&nbsp;&nbsp;&nbsp;"),
                *self.buttons.values(),
            ]
        )
        for key, b in self.buttons.items():
            b.key = key
            b.observe(self._check_visibility, names="value")
        self._widgets.insert(0, self.modules_widget)

    def _check_visibility(self, _):
        active_nodes = [
            node_id
            for key, node_id in self.artist_mapping.items()
            if self.buttons[key].value
        ]
        for n in self._node_backup:
            self.fig.artists[n.id].points.visible = n.id in active_nodes
        self.cutting_tool._original_nodes = [
            n for n in self._node_backup if n.id in active_nodes
        ]
        self.cutting_tool.update_state()
