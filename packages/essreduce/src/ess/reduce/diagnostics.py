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


class MaskControl(ipw.HBox):
    def __init__(self):
        self._modes = {
            key: ipw.ToggleButton(icon=icon, value=False, layout={"width": "38px"})
            for key, icon in (
                ("horizontal", "grip-lines"),
                ("vertical", "grip-lines-vertical"),
                ("single", "dot-circle"),
            )
        }
        self._lock = False
        self.mode = None
        for button in self._modes.values():
            button.observe(self._sync_modes, names='value')
        super().__init__([ipw.Label(value="Masking:"), *self._modes.values()])

    def _sync_modes(self, change: dict):
        if self._lock or (not change['new']):
            return
        if not change['new']:
            self.mode = None
            return
        self._lock = True
        for mode, button in self._modes.items():
            if button.value:
                if button is change['owner']:
                    self.mode = mode
                else:
                    button.value = False
        self._lock = False
        # for mode, button in self._modes.items():
        #     if button.value:
        #         self.mode = mode

    def on_change(self, callback: callable):
        for button in self._modes.values():
            button.observe(callback, names='value')


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
            description=f"{dim} ",
            min=0,
            max=self._sizes[dim] - 1,
            value=0,
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
    return da.assign_coords({dim: sc.arange(dim, sz) for dim, sz in da.sizes.items()})


def _slice_or_reduce(
    da: sc.DataArray, slices: list[tuple[str, int | None]], op: str
) -> sc.DataArray:
    out = da
    for sl in slices:
        dim, ind = sl
        if dim not in out.sizes:
            continue
        if ind is None:
            out = getattr(sc, op)(out, dim)
        else:
            if ind >= out.sizes[dim]:
                # Make a blank data array to avoid indexing errors
                out = sc.full_like(out[dim, 0], value=float("nan"))
            else:
                out = out[dim, ind]
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
        figsize=(11, 8),
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

        # We select the individual banks from the data group using separate nodes.
        # Each bank data is then sliced according to the positions of the dim sliders.
        self.bank_selection_nodes = {
            key: pp.Node(lambda dg, key: dg[key], dg=self.data, key=key)
            for key in self.data.keys()
        }
        self.slice_data_nodes = {
            key: pp.Node(
                _slice_or_reduce,
                da=self.bank_selection_nodes[key],
                slices=self.collect_nodes[key],
                op="sum",
            )
            for key in self.data.keys()
        }

        # Before displaying on the figures, a reduction operation is sometimes applied.
        # If we apply the masks early in the node tree (e.g. just after the input data),
        # the masked data will be zeroed during the reduction and will no longer be
        # easily visible on the figures. Instead, we slices the masks as well and only
        # apply them just before displaying the data on the figures.
        self.mask_node = pp.Node(lambda: self.masks)
        self.mask_selection_nodes = {
            key: pp.Node(lambda masks, key: masks[key], masks=self.mask_node, key=key)
            for key in self.data.keys()
        }
        self.slice_masks_nodes = {
            key: pp.Node(
                _slice_or_reduce,
                da=self.mask_selection_nodes[key],
                slices=self.collect_nodes[key],
                op="any",
            )
            for key in self.data.keys()
        }
        self.apply_masks_nodes = {
            key: pp.Node(
                lambda da, mask: da.assign_masks({"mask": mask}),
                da=self.slice_data_nodes[key],
                mask=self.slice_masks_nodes[key],
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
        self.all_slice_data_nodes = {
            key: pp.Node(
                _slice_or_reduce,
                da=self.bank_selection_nodes[key],
                slices=self.all_collect_node,
                op="sum",
            )
            for key in self.data.keys()
        }
        self.all_slice_masks_nodes = {
            key: pp.Node(
                _slice_or_reduce,
                da=self.mask_selection_nodes[key],
                slices=self.all_collect_node,
                op="any",
            )
            for key in self.data.keys()
        }
        self.all_apply_masks_nodes = {
            key: pp.Node(
                lambda da, mask: da.assign_masks({"mask": mask}),
                da=self.all_slice_data_nodes[key],
                mask=self.all_slice_masks_nodes[key],
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

        self.subplots = {}
        self.tab_figs = {}
        # self.nodes = []
        for i, (bank, ax) in enumerate(
            zip(self.data.keys(), axs.flatten(), strict=True)
        ):
            self.subplots[bank] = pp.imagefigure(
                self.all_apply_masks_nodes[bank],
                ax=ax,
                title=bank,
                cbar=True,
                **kwargs,
            )
            ax._bank_explorer_subplot_index = i
            self.tab_figs[bank] = pp.imagefigure(
                self.apply_masks_nodes[bank],
                title=bank,
                cbar=True,
                # Need to make some horizontal space for the toolbar
                figsize=(0.96 * figsize[0], figsize[1]),
                **kwargs,
            )

        self.main_figure.canvas.header_visible = False
        self._tab_switch_connection = self.main_figure.canvas.mpl_connect(
            'button_press_event', self.switch_to_tab
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
                for key, f in zip(self.data.keys(), self.tab_figs.values(), strict=True)
            ],
        ]

        self._tab_index_to_key = {
            0: "All",
            **{i + 1: key for i, key in enumerate(self.data.keys())},
        }

        self.tabs.titles = list(self._tab_index_to_key.values())
        # self.tabs.observe(self.selected_tab_changed, names='selected_index')

        # for sl in self.master_dim_sliders:
        #     sl.on_dim_change(self.on_dim_change)

        # dim_slider_widgets = [
        #     widget
        #     for dslider in self.dim_sliders.values()
        #     for widget in (dslider, ipw.HTML('<div style="width: 20px;"></div>'))
        # ]

        # self.links = {}
        # self.link_master_to_all()

        self.mask_control = MaskControl()
        self.mask_connections = []
        self.mask_control.on_change(self.on_toggle_masking)

        super().__init__(
            [
                ipw.HBox(
                    [
                        self.mask_control,
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

    # def selected_tab_changed(self, change: dict) -> None:
    #     """
    #     Handle the event when the selected tab changes.

    #     Parameters
    #     ----------
    #     change:
    #         The change dictionary containing information about the tab change.
    #     """
    #     if change["new"] == 0:

    def update_colormapper_norm(self) -> None:
        ind = self.tabs.selected_index
        if ind == 0:
            figs = self.subplots.values()
        else:
            figs = [self.tab_figs[self._tab_index_to_key[ind]]]
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
            figs = self.subplots.values()
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

    def on_toggle_masking(self, change: dict) -> None:
        if self.mask_control.mode is not None:
            # print("Masking enabled")
            self.mask_connections = {
                key: f.fig.canvas.mpl_connect('button_press_event', self.add_mask)
                for key, f in self.tab_figs.items()
            }
        else:
            # print("Masking disabled")
            for key, conn in self.mask_connections.items():
                f = self.tab_figs[key]
                f.fig.canvas.mpl_disconnect(conn)
            self.mask_connections = {}

    def add_mask(self, event: MouseEvent) -> None:
        if event.inaxes is None:
            return

        print("Mouse event:", event)
        # Current tab index
        key = self._tab_index_to_key[self.tabs.selected_index]
        # Fig view dims:
        masking_mode = self.mask_control.mode

        xdim = self.tab_figs[key].view._dims['x']
        x = int(round(event.xdata))
        ydim = self.tab_figs[key].view._dims['y']
        y = int(round(event.ydata))

        # Find slider dims and indices
        slices = []
        if masking_mode in ("horizontal", "single"):
            slices.append((ydim, y))
        if masking_mode in ("vertical", "single"):
            slices.append((xdim, x))
        slices.extend([sl.value for sl in self.bank_dim_sliders[key]])

        # key = list(self.data.keys())[current_tab_index - 1]
        print("Current key:", key)
        print("Masking at:", slices)
        m = self.masks[key]
        for s in [sli for sli in slices if sli[1] is not None]:
            m = m[s]
        # Set the mask at the computed location
        m.values[...] = False if m.values.any() else True
        self.mask_node.notify_children(message=None)
