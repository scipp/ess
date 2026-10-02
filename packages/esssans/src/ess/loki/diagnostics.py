# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Detector diagnostics for LOKI."""

from __future__ import annotations

import ipywidgets as ipw
import matplotlib.pyplot as plt
import plopp as pp
import scipp as sc


def _add_missing_coordinates(da: sc.DataArray) -> sc.DataArray:
    return da.assign_coords(
        {dim: sc.arange(dim=dim, start=1, stop=sz + 1) for dim, sz in da.sizes.items()}
    )


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
    def __init__(self, data: sc.DataGroup, figsize=(12, 9), **kwargs):
        """Widget to view LOKI detector banks.

        Parameters
        ----------
        data:
            DataGroup containing LOKI detector banks.
        figsize:
            Size of the figures in inches (width, height).
        kwargs:
            Additional arguments are forwarded to the plotting function.
        """
        self.data = sc.DataGroup(
            {k: _add_missing_coordinates(da) for k, da in data.items()}
        )
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
            self.main_figure, axs = plt.subplots(3, 3, figsize=figsize)
            bank_figures = [plt.subplots(figsize=figsize) for _ in range(9)]

        self.subplots = []
        self.tab_figs = []
        self.nodes = []
        for i, ax in enumerate(axs.flatten()):
            bank = f"loki_detector_{i}"
            n = pp.Node(lambda da, key: da[key], da=self.slice_node, key=bank)
            self.nodes.append(n)
            self.subplots.append(
                pp.imagefigure(n, ax=ax, title=bank, cbar=True, **kwargs)
            )
            self.tab_figs.append(
                pp.imagefigure(
                    ax=bank_figures[i][1],
                    title=bank,
                    cbar=True,
                    figsize=figsize,
                    **kwargs,
                )
            )
            bank_figures[i][0].canvas.header_visible = False

        self.main_figure.canvas.header_visible = False

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

        self.tabs = ipw.Tab(
            layout={'width': f'{figsize[0] * self.main_figure.get_dpi() + 40}px'}
        )
        self.tabs.children = [self.main_figure.canvas] + [
            f.fig.canvas for f in self.tab_figs
        ]
        self.tabs.titles = ["All", *(f"Bank {i}" for i in range(len(self.tab_figs)))]
        self.tabs.observe(self.update_node_routing, names='selected_index')

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
                self.tabs,
            ]
        )

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
