# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import math
import warnings
from itertools import chain
from typing import Literal

import ipywidgets as ipw
import matplotlib.pyplot as plt
import plopp as pp
import scipp as sc
from matplotlib.backend_bases import MouseEvent


class DimSlider(ipw.HBox, ipw.ValueWidget):
    def __init__(self, sizes: dict[str, int], dim: str, visible_dims: list[str]):
        self._lock = False
        self._sizes = sizes
        self._visible_dims = visible_dims
        self.dim_selector = ipw.Dropdown(
            options=[dim, *visible_dims],
            value=dim,
            description="",
            layout={"width": "initial"},
        )
        self.slider = ipw.IntSlider(
            min=1, max=self._sizes[dim], value=1, style={"description_width": "initial"}
        )
        self.sum = ipw.Checkbox(
            description="Sum", value=False, indent=False, layout={"width": "initial"}
        )

        # self.ind_node = pp.widget_node(self.slider)
        # self.sum_node = pp.widget_node(self.sum)

        self._link = ipw.jslink((self.sum, 'value'), (self.slider, 'disabled'))

        self.value = (self.dim, self.slider.value)

        self.dim_selector.observe(self._dim_changed, names='value')
        self.slider.observe(self._on_subwidget_change, names='value')
        self.sum.observe(self._on_subwidget_change, names='value')

        super().__init__(
            [self.dim_selector, self.slider, self.sum],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )

    def on_dim_change(self, callback: callable):
        self.dim_selector.observe(callback, names='value')
        # self.slider.observe(callback, names='value')
        # self.sum.observe(callback, names='value')

    @property
    def dim(self):
        return self.dim_selector.value

    # @property
    # def value(self):
    #     return (self.dim, self.slider.value)

    def _update_options(self, visible_dims: list[str]):
        backup = self.dim_selector.value
        self.dim_selector.options = [self.dim, *visible_dims]
        self._lock = True
        self.dim_selector.value = backup
        self._lock = False

    def _dim_changed(self, _=None):
        if self._lock:
            return
        self.slider.max = self._sizes[self.dim]
        self._on_subwidget_change()

    def _on_subwidget_change(self, _=None):
        if self.sum.value:
            self.value = (self.dim, None)
        else:
            self.value = (self.dim, self.slider.value)


def _add_missing_coordinates(da: sc.DataArray) -> sc.DataArray:
    return da.assign_coords(
        {dim: sc.arange(dim=dim, start=1, stop=sz + 1) for dim, sz in da.sizes.items()}
    )


def _slice_or_sum(
    dg: sc.DataGroup, slices: list[tuple[str, int | None]]
) -> sc.DataGroup:
    out = sc.DataGroup()
    for key, da in dg.items():
        item = da
        for sl in slices:
            dim, ind = sl
            if ind is None:
                item = item.sum(dim)
            else:
                if ind > item.sizes[dim]:
                    # Make a blank data array to avoid indexing errors
                    item = sc.full_like(item[dim, 0], value=float("nan"))
                else:
                    item = item[dim, ind - 1]
        out[key] = item
    return out


class BankExplorer(ipw.VBox):
    def __init__(
        self,
        data: sc.DataGroup,
        figsize=(12, 9),
        show_dims: list[str] | None = None,
        **kwargs,
    ):
        """Widget to view detector banks.

        Parameters
        ----------
        data:
            DataGroup containing detector banks.
        show_dims:
            List of dimensions to display. If ``None``, the last two dimensions are
            displayed.
        figsize:
            Size of the figures in inches (width, height).
        kwargs:
            Additional arguments are forwarded to the plotting function.
        """
        self._lock = False

        self.data = sc.DataGroup(
            {k: _add_missing_coordinates(da) for k, da in data.items()}
        )
        max_sizes = {
            dim: max(da.sizes[dim] for da in self.data.values())
            for dim in self.data.dims
        }

        self._visible_dims = show_dims
        if self._visible_dims is None:
            self._visible_dims = list(self.data.dims)[-2:]

        other_dims = [dim for dim in self.data.dims if dim not in self._visible_dims]

        self.dim_sliders = [
            DimSlider(sizes=max_sizes, dim=dim, visible_dims=self._visible_dims)
            for dim in other_dims
        ]
        dim_nodes = [pp.widget_node(slider) for slider in self.dim_sliders]

        self.collect_node = pp.Node(lambda *args: args, *dim_nodes)

        # self.layer_slider = ipw.IntSlider(
        #     min=1, max=4, description="Layer", style={"description_width": 'initial'}
        # )
        # self.layer_sum = ipw.Checkbox(
        #     description="Sum", value=False, indent=False, layout={"width": "initial"}
        # )

        # self.layer_ind_node = pp.widget_node(self.layer_slider)
        # self.layer_sum_node = pp.widget_node(self.layer_sum)

        # self.straw_slider = ipw.IntSlider(
        #     min=1, max=7, description="Straw", style={"description_width": 'initial'}
        # )
        # self.straw_sum = ipw.Checkbox(
        #     description="Sum", value=False, indent=False, layout={"width": "initial"}
        # )
        # self.straw_ind_node = pp.widget_node(self.straw_slider)
        # self.straw_sum_node = pp.widget_node(self.straw_sum)

        # self.layer_link = ipw.jslink(
        #     (self.layer_sum, 'value'), (self.layer_slider, 'disabled')
        # )
        # self.straw_link = ipw.jslink(
        #     (self.straw_sum, 'value'), (self.straw_slider, 'disabled')
        # )

        self.slice_node = pp.Node(_slice_or_sum, dg=self.data, slices=self.collect_node)

        with plt.ioff():
            ncols = 3
            nrows = math.ceil(len(self.data) / ncols)
            self.main_figure, axs = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=figsize
            )
            # bank_figures = [
            #     plt.subplots(figsize=figsize) for _ in range(len(self.data))
            # ]

        self.subplots = []
        self.tab_figs = []
        self.nodes = []
        for i, (bank, ax) in enumerate(
            zip(self.data.keys(), axs.flatten(), strict=True)
        ):
            # bank = f"loki_detector_{i}"
            n = pp.Node(lambda dg, key: dg[key], dg=self.slice_node, key=bank)
            self.nodes.append(n)
            self.subplots.append(
                pp.imagefigure(n, ax=ax, title=bank, cbar=True, **kwargs)
            )
            ax._bank_explorer_subplot_index = i
            self.tab_figs.append(
                pp.imagefigure(
                    pp.Node(sc.full_like(n(), value=float("nan"))),
                    title=bank,
                    cbar=True,
                    # Need to make some horizontal space for the toolbar
                    figsize=(0.96 * figsize[0], figsize[1]),
                    **kwargs,
                )
            )

        self.main_figure.canvas.header_visible = False
        self.main_figure.canvas.mpl_connect('button_press_event', self.switch_to_tab)

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

        is_log = next(
            iter({kwargs.get('norm'), kwargs.get('logc')} - {None}), None
        ) in ('log', True)
        self.log_button = ipw.ToggleButton(
            description="log",
            layout={"width": "40px"},
            value=is_log,
        )
        self.log_button.observe(self.toggle_log, names="value")

        # layer_box = ipw.HBox(
        #     [self.layer_slider, self.layer_sum],
        #     layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        # )
        # straw_box = ipw.HBox(
        #     [self.straw_slider, self.straw_sum],
        #     layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        # )
        space = ipw.HTML('<div style="width: 20px;"></div>')

        self.tabs = ipw.Tab(
            layout={'width': f'{figsize[0] * self.main_figure.get_dpi() + 40}px'}
        )
        # self.tabs.children = [self.main_figure.canvas] + [
        #     f.fig.canvas for f in self.tab_figs
        # ]
        self.tabs.children = [self.main_figure.canvas, *self.tab_figs]
        self.tabs.titles = ["All", *self.data.keys()]
        self.tabs.observe(self.update_node_routing, names='selected_index')

        for sl in self.dim_sliders:
            sl.on_dim_change(self.on_dim_change)

        # dim_slider_widgets = [
        #     widget
        #     for dslider in self.dim_sliders.values()
        #     for widget in (dslider, ipw.HTML('<div style="width: 20px;"></div>'))
        # ]

        super().__init__(
            [
                ipw.HBox(
                    [
                        widget
                        for dslider in self.dim_sliders
                        for widget in (
                            dslider,
                            ipw.HTML('<div style="width: 20px;"></div>'),
                        )
                    ]
                    + [
                        ipw.Label("Colorscale:"),
                        self.cmap_vmin,
                        self.cmap_vmax,
                        self.log_button,
                    ]
                ),
                self.tabs,
            ]
        )

    def on_dim_change(self, _):
        slider_dims = [sl.dim for sl in self.dim_sliders]
        self._visible_dims = [dim for dim in self.data.dims if dim not in slider_dims]
        for f in chain(self.subplots, self.tab_figs):
            # We clear the dims on the view and the canvas, to make them believe they
            # are empty, after which they are ready to receive new data with new
            # dimensions.
            for key in f.view._dims:
                f.view._dims[key] = None
            f.canvas.dims = {}
        self.update_node_routing({"new": self.tabs.selected_index})

    def switch_to_tab(self, event: MouseEvent) -> None:
        """
        Switch to the tab corresponding to the subplot that was clicked.

        Parameters
        ----------
        event:
            The mouse event triggered by clicking on a subplot.
        """
        if hasattr(event.inaxes, '_bank_explorer_subplot_index'):
            index = event.inaxes._bank_explorer_subplot_index
            self.tabs.selected_index = index + 1

    def update_colormapper_norm(self) -> None:
        if self.tabs.selected_index == 0:
            figs = self.subplots
        else:
            figs = [self.tab_figs[self.tabs.selected_index - 1]]
        for f in figs:
            f.view.colormapper.norm = "log" if self.log_button.value else "linear"

    def toggle_log(self, _: dict) -> None:
        self.reset_cmap_range()
        self.update_colormapper_norm()

    def _update_min_or_max(self, change: dict, bound: str) -> None:
        try:
            new = float(change["new"])
            if self.log_button.value and new <= 0:
                return
        except ValueError:
            new = None

        if self.tabs.selected_index == 0:
            figs = self.subplots
        else:
            figs = [self.tab_figs[self.tabs.selected_index - 1]]

        for f in figs:
            if new is None:
                getattr(f.view.colormapper, f"_c{bound}").pop('user', None)
            else:
                setattr(f.view.colormapper, f"c{bound}", new)
            if not self._lock:
                f.view.colormapper.autoscale()
        if not self._lock:
            if self.tabs.selected_index == 0:
                self.main_figure.canvas.draw_idle()
            else:
                self.tab_figs[self.tabs.selected_index - 1].canvas.draw()

    def update_cmin(self, change: dict) -> None:
        self._update_min_or_max(change, 'min')

    def update_cmax(self, change: dict) -> None:
        self._update_min_or_max(change, 'max')

    def reset_cmap_range(self, _=None) -> None:
        self._lock = True
        self.cmap_vmin.value = ''
        self.cmap_vmax.value = ''
        self._lock = False

    def update_node_routing(self, change: dict) -> None:
        for n in self.nodes:
            n.views.clear()
        if change['new'] == 0:
            for f, n in zip(self.subplots, self.nodes, strict=True):
                n.add_view(f.view)
        else:
            n = self.nodes[change['new'] - 1]
            f = self.tab_figs[change['new'] - 1]
            n.add_view(f.view)
        self.slice_node.notify_children(message=None)
        self.update_colormapper_norm()
        self._lock = True
        self.update_cmin({'new': self.cmap_vmin.value})
        self._lock = False
        self.update_cmax({'new': self.cmap_vmax.value})
