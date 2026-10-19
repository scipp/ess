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


class DimSelector(ipw.HBox):
    def __init__(self, visible_dims: list[str], all_dims: list[str]):
        self.xdim_selector = ipw.Dropdown(
            options=all_dims,
            value=visible_dims[0],
            description="Horizontal:",
            layout={"width": "initial"},
        )
        self.ydim_selector = ipw.Dropdown(
            options=all_dims,
            value=visible_dims[1],
            description="Vertical:",
            layout={"width": "initial"},
        )
        self.lock = False

        super().__init__(
            [
                ipw.Label(value="Select dimensions:"),
                self.xdim_selector,
                self.ydim_selector,
            ],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )
        self.xdim_selector.observe(self._check_xdim, names='value')
        self.ydim_selector.observe(self._check_ydim, names='value')

    def on_dim_change(self, callback: callable):
        self.xdim_selector.observe(callback, names='value')
        self.ydim_selector.observe(callback, names='value')

    def _check_xdim(self, change):
        if self.lock:
            return
        if change['new'] == self.ydim_selector.value:
            self.ydim_selector.value = change['old']

    def _check_ydim(self, change):
        if self.lock:
            return
        if change['new'] == self.xdim_selector.value:
            self.xdim_selector.value = change['old']


class DimSlider(ipw.HBox, ipw.ValueWidget):
    def __init__(self, sizes: dict[str, int], dim: str, visible_dims: list[str]):
        self._lock = False
        self._sizes = sizes
        self._visible_dims = visible_dims
        # self.dim_selector = ipw.Dropdown(
        #     options=[dim, *visible_dims],
        #     value=dim,
        #     description="",
        #     layout={"width": "initial"},
        # )
        self._dim = dim
        self.slider = ipw.IntSlider(
            description=f"{dim}:",
            min=1,
            max=self._sizes[dim],
            value=1,
            style={"description_width": "initial"},
        )
        self.sum = ipw.Checkbox(
            description="Sum", value=False, indent=False, layout={"width": "initial"}
        )

        # self.ind_node = pp.widget_node(self.slider)
        # self.sum_node = pp.widget_node(self.sum)

        self._link = ipw.jslink((self.sum, 'value'), (self.slider, 'disabled'))

        self.value = (self.dim, self.slider.value)

        # self.dim_selector.observe(self._dim_changed, names='value')
        self.slider.observe(self._on_subwidget_change, names='value')
        self.sum.observe(self._on_subwidget_change, names='value')

        super().__init__(
            [self.slider, self.sum],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )

    # def on_dim_change(self, callback: callable):
    #     self.dim_selector.observe(callback, names='value')
    #     # self.slider.observe(callback, names='value')
    #     # self.sum.observe(callback, names='value')

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        self.slider.description = f"{value}:"
        self.slider.max = self._sizes[value]

    # @property
    # def value(self):
    #     return (self.dim, self.slider.value)

    # def update_options(self, visible_dims: list[str]):
    #     new_dims = [self.dim, *visible_dims]
    #     if set(new_dims) == set(self.dim_selector.options):
    #         return
    #     backup = self.dim_selector.value
    #     self.dim_selector.options = new_dims
    #     self.dim_selector.value = backup

    # def _dim_changed(self, _=None):
    #     if self._lock:
    #         return
    #     self.slider.max = self._sizes[self.dim]
    #     self._on_subwidget_change()

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
    da: sc.DataArray, slices: list[tuple[str, int | None]]
) -> sc.DataArray:
    out = da
    for sl in slices:
        dim, ind = sl
        if dim not in out.sizes:
            continue
        if ind is None:
            out = out.sum(dim)
        else:
            if ind > out.sizes[dim]:
                # Make a blank data array to avoid indexing errors
                out = sc.full_like(out[dim, 0], value=float("nan"))
            else:
                out = out[dim, ind - 1]
    return out


# def _slice_or_sum(
#     dg: sc.DataGroup, slices: list[tuple[str, int | None]]
# ) -> sc.DataGroup:
#     out = sc.DataGroup()
#     for key, da in dg.items():
#         item = da
#         for sl in slices:
#             dim, ind = sl
#             if dim not in item.sizes:
#                 continue
#             if ind is None:
#                 item = item.sum(dim)
#             else:
#                 if ind > item.sizes[dim]:
#                     # Make a blank data array to avoid indexing errors
#                     item = sc.full_like(item[dim, 0], value=float("nan"))
#                 else:
#                     item = item[dim, ind - 1]
#         out[key] = item
#     return out


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
        self.masks = {
            k: sc.zeros(sizes=da.sizes, dtype=bool) for k, da in self.data.items()
        }
        self.max_sizes = {
            dim: max(da.sizes.get(dim, 0) for da in self.data.values())
            for dim in self.data.dims
        }

        self._visible_dims = show_dims
        if self._visible_dims is None:
            self._visible_dims = list(self.data.dims)[-2:]

        other_dims = [dim for dim in self.data.dims if dim not in self._visible_dims]

        self.bank_dim_sliders = {
            key: [
                DimSlider(
                    sizes=self.data[key].sizes, dim=dim, visible_dims=self._visible_dims
                )
                for dim in other_dims
            ]
            for key in self.data.keys()
        }

        self.collect_nodes = {
            key: pp.Node(
                lambda *args: args,
                *[pp.widget_node(slider) for slider in self.bank_dim_sliders[key]],
            )
            for key in self.data.keys()
        }

        self.mask_node = pp.Node(lambda: self.masks)
        self.apply_masks_node = pp.Node(
            lambda dg, masks: sc.DataGroup(
                {key: da.assign_masks({"mask": masks[key]}) for key, da in dg.items()}
            ),
            dg=self.data,
            masks=self.mask_node,
        )

        self.bank_selection_nodes = {
            key: pp.Node(lambda dg, key: dg[key], dg=self.apply_masks_node, key=key)
            for key in self.data.keys()
        }

        self.slice_nodes = {
            key: pp.Node(
                _slice_or_sum,
                da=self.bank_selection_nodes[key],
                slices=self.collect_nodes[key],
            )
            for key in self.data.keys()
        }

        # There is a significant performance penalty in Matplotlib if we display the
        # data from the same node on both the 'All detectors' subplots and the
        # individual bank figures. To avoid this, we have to create separate dim
        # sliders for the 'All detectors' view. In turn, we need separate nodes in the
        # node tree below.
        self.all_dim_sliders = [
            DimSlider(sizes=self.max_sizes, dim=dim, visible_dims=self._visible_dims)
            for dim in other_dims
        ]
        self.all_collect_node = pp.Node(
            lambda *args: args,
            *[pp.widget_node(slider) for slider in self.all_dim_sliders],
        )
        self.allbanks_slice_nodes = {
            key: pp.Node(
                _slice_or_sum,
                da=self.bank_selection_nodes[key],
                slices=self.all_collect_node,
            )
            for key in self.data.keys()
        }

        # # Master dim sliders
        # self.master_dim_sliders = [
        #     DimSlider(sizes=self.max_sizes, dim=dim, visible_dims=self._visible_dims)
        #     for dim in other_dims
        # ]

        # return

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
        # self.nodes = []
        for i, (bank, ax) in enumerate(
            zip(self.data.keys(), axs.flatten(), strict=True)
        ):
            self.subplots.append(
                pp.imagefigure(
                    self.allbanks_slice_nodes[bank],
                    ax=ax,
                    title=bank,
                    cbar=True,
                    **kwargs,
                )
            )
            ax._bank_explorer_subplot_index = i
            self.tab_figs.append(
                pp.imagefigure(
                    self.slice_nodes[bank],
                    title=bank,
                    cbar=True,
                    # Need to make some horizontal space for the toolbar
                    figsize=(0.96 * figsize[0], figsize[1]),
                    **kwargs,
                )
            )

        self.main_figure.canvas.header_visible = False
        self._tab_switch_connection = self.main_figure.canvas.mpl_connect(
            'button_press_event', self.switch_to_tab
        )

        self.dim_selector = DimSelector(
            visible_dims=self._visible_dims, all_dims=self.data.dims
        )

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

        self.dim_selector.on_dim_change(self.on_dim_change)

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
        # self.tabs.children = [ipw.VBox([self._all_dim_sliders, self.main_figure.canvas]),
        #                       *[ipw.VBox([self.banks_dim_sliders[i], f]) for i, f in enumerate(self.tab_figs)]]

        self.tabs.children = [
            ipw.VBox(
                [
                    ipw.HBox(
                        [
                            widget
                            for dslider in self.all_dim_sliders
                            for widget in (dslider, space)
                        ]
                    ),
                    self.main_figure.canvas,
                ]
            ),
            *[
                ipw.VBox(
                    [
                        ipw.HBox(
                            [
                                widget
                                for dslider in self.bank_dim_sliders[key]
                                for widget in (dslider, space)
                            ]
                        ),
                        f,
                    ]
                )
                for key, f in zip(self.data.keys(), self.tab_figs, strict=True)
            ],
        ]

        self.tabs.titles = ["All", *self.data.keys()]
        # self.tabs.observe(self.update_slider_control, names='selected_index')

        # for sl in self.master_dim_sliders:
        #     sl.on_dim_change(self.on_dim_change)

        # dim_slider_widgets = [
        #     widget
        #     for dslider in self.dim_sliders.values()
        #     for widget in (dslider, ipw.HTML('<div style="width: 20px;"></div>'))
        # ]

        # self.links = {}
        # self.link_master_to_all()

        super().__init__(
            [
                ipw.HBox(
                    # [
                    #     widget
                    #     for dslider in self.master_dim_sliders
                    #     for widget in (
                    #         dslider,
                    #         ipw.HTML('<div style="width: 20px;"></div>'),
                    #     )
                    # ]
                    [
                        self.dim_selector,
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

    # def unlink_all(self):
    #     for link in self.links.values():
    #         link.unlink()

    # def link_master_to_all(self):
    #     self.unlink_all()
    #     for dim_slider, master_slider in zip(
    #         self.all_dim_sliders, self.master_dim_sliders, strict=True
    #     ):
    #         self.links[f"slider_allbanks_{dim_slider.dim}"] = ipw.jslink(
    #             (master_slider.slider, 'value'), (dim_slider.slider, 'value')
    #         )
    #         self.links[f"checkbox_allbanks_{dim_slider.dim}"] = ipw.jslink(
    #             (master_slider.sum, 'value'), (dim_slider.sum, 'value')
    #         )

    # def link_master_to_one(self, key: str):
    #     self.unlink_all()
    #     for bank_slider, master_slider in zip(
    #         self.bank_dim_sliders[key], self.master_dim_sliders, strict=True
    #     ):
    #         self.links[f"{key}_{bank_slider.dim}"] = ipw.jslink(
    #             (master_slider.slider, 'value'), (bank_slider.slider, 'value')
    #         )
    #         self.links[f"checkbox_{key}_{bank_slider.dim}"] = ipw.jslink(
    #             (master_slider.sum, 'value'), (bank_slider.sum, 'value')
    #         )

    # def update_slider_control(self, change: dict) -> None:
    #     if change['new'] == 0:
    #         self.link_master_to_all()
    #         for sl in self.master_dim_sliders:
    #             sl.max = self.max_sizes[sl.dim]
    #     else:
    #         key = self.tabs.titles[self.tabs.selected_index]
    #         self.unlink_all()
    #         for sl in self.master_dim_sliders:
    #             sl.max = self.data[key].sizes[sl.dim]
    #         self.link_master_to_one(key)

    def on_dim_change(self, change):
        # slider_dims = [sl.dim for sl in self.master_dim_sliders]
        # visible_dims = [dim for dim in self.data.dims if dim not in slider_dims]
        for f in chain(self.subplots, self.tab_figs):
            # We clear the dims on the view and the canvas, to make them believe they
            # are empty, after which they are ready to receive new data with new
            # dimensions.
            for key in f.view._dims:
                f.view._dims[key] = None
            f.canvas.dims = {}
            f.view.artists.clear()
            f.ax.clear()
        # self.update_node_routing({"new": self.tabs.selected_index})
        for sl in chain(
            # self.master_dim_sliders,
            self.all_dim_sliders,
            *self.bank_dim_sliders.values(),
        ):
            if sl.dim == change['new']:
                sl.dim = change['old']
                # sl.slider.max = self.data.sizes[sl.dim]
            # sl.update_options(visible_dims)

        # self.apply_masks_node.notify_children(message=None)

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

    # def update_node_routing(self, change: dict) -> None:
    #     return
    #     for n in self.nodes:
    #         n.views.clear()
    #     if change['new'] == 0:
    #         for f, n in zip(self.subplots, self.nodes, strict=True):
    #             n.add_view(f.view)
    #         self.tab_switch_connection = self.main_figure.canvas.mpl_connect(
    #             'button_press_event', self.switch_to_tab
    #         )
    #     else:
    #         n = self.nodes[change['new'] - 1]
    #         f = self.tab_figs[change['new'] - 1]
    #         n.add_view(f.view)
    #         self.main_figure.canvas.mpl_disconnect(self._tab_switch_connection)
    #         self._tab_switch_connection = None
    #         self._mask_connection = f.fig.canvas.mpl_connect(
    #             'button_press_event', self.add_mask
    #         )
    #     self.slice_node.notify_children(message=None)
    #     self.update_colormapper_norm()
    #     self._lock = True
    #     self.update_cmin({'new': self.cmap_vmin.value})
    #     self._lock = False
    #     self.update_cmax({'new': self.cmap_vmax.value})

    def add_mask(self, event: MouseEvent) -> None:
        if event.inaxes is None:
            return

        print("Mouse event:", event)
        # Current tab index
        current_tab_index = self.tabs.selected_index
        # Fig view dims:
        ydim = self.tab_figs[current_tab_index - 1].view._dims['y']
        # Find closest y integer and mask the row
        y = int(round(event.ydata))

        # Find slider dims and indices
        slices = [(ydim, y)] + [sl.value for sl in self.dim_sliders]

        key = list(self.data.keys())[current_tab_index - 1]
        print("Current key:", key)
        m = self.masks[key]
        for s in slices:
            m = m[s]
        # Set the mask at the computed location
        m.values[...] = ~m.values[...]
        self.mask_node.notify_children(message=None)
