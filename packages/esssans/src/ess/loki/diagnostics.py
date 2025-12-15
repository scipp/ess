# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Detector diagnostics for LOKI."""

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
    def __init__(self, data: sc.DataGroup):
        """Widget to view LOKI detector banks.

        Parameters
        ----------
        data:
            DataGroup containing LOKI detector banks.
        """
        self.data = data

        self.layer_slider = ipw.IntSlider(
            min=1, max=4, description="Layer", style={"description_width": 'initial'}
        )
        self.layer_sum = ipw.Checkbox(
            description="Sum all layers",
            value=False,
            indent=False,
            layout={"width": "initial"},
        )
        self.layer_ind_node = pp.widget_node(self.layer_slider)
        self.layer_sum_node = pp.widget_node(self.layer_sum)

        self.straw_slider = ipw.IntSlider(
            min=1, max=7, description="Straw", style={"description_width": 'initial'}
        )
        self.straw_sum = ipw.Checkbox(
            description="Sum all straws",
            value=False,
            indent=False,
            layout={"width": "initial"},
        )
        self.straw_ind_node = pp.widget_node(self.straw_slider)
        self.straw_sum_node = pp.widget_node(self.straw_sum)

        self.layer_link = ipw.jslink(
            (self.layer_sum, 'value'), (self.layer_slider, 'disabled')
        )
        self.straw_link = ipw.jslink(
            (self.straw_sum, 'value'), (self.straw_slider, 'disabled')
        )

        slice_node = pp.Node(
            _slice_or_sum,
            da=self.data,
            layer_ind=self.layer_ind_node,
            layer_sum=self.layer_sum_node,
            straw_ind=self.straw_ind_node,
            straw_sum=self.straw_sum_node,
        )

        with plt.ioff():
            fig, axs = plt.subplots(3, 3, figsize=(12, 9))

        figs = []
        for i, ax in enumerate(axs.flatten()):
            bank = f"loki_detector_{i}"
            figs.append(
                pp.plot(
                    pp.Node(lambda da, key: da[key], da=slice_node, key=bank),
                    ax=ax,
                    title=bank,
                )
            )
        fig.canvas.header_visible = False

        self.log_button = ipw.ToggleButton(description="Log colormap")

        def toggle_log(change):
            for f in figs:
                f.view.colormapper.norm = "log" if change["new"] else "linear"

        self.log_button.observe(toggle_log, names="value")

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
                ipw.HBox([layer_box, space, straw_box, space, self.log_button]),
                fig.canvas,
            ]
        )
