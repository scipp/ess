# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import math
from collections.abc import Callable
from typing import Literal

import ipywidgets as ipw
import matplotlib.pyplot as plt
import plopp as pp
import scipp as sc
from matplotlib.backend_bases import MouseEvent

BUTTON_STYLE = {"width": "40px"}

# def current_tab(self) -> int:
#     return self.tabs.selected_index


# def get_current_key(tabs) -> str:
#     return tabs.titles[tabs.selected_index]


class MaskControl(ipw.HBox):
    def __init__(
        self,
        # tabs,
        # tab_figs,
        # masks,
        # mask_selection_nodes,
        # all_mask_selection_nodes,
        # bank_dim_sliders,
        # all_dim_sliders,
    ):
        # self._tabs = tabs
        # self._tab_figs = tab_figs
        # self._masks = masks
        # self._mask_selection_nodes = mask_selection_nodes
        # self._all_mask_selection_nodes = all_mask_selection_nodes
        # self._bank_dim_sliders = bank_dim_sliders
        # self._all_dim_sliders = all_dim_sliders

        self.mode_buttons = {
            key: ipw.ToggleButton(
                icon=icon, value=False, tooltip=tip, layout=BUTTON_STYLE
            )
            for key, icon, tip in (
                ("horizontal", "grip-lines", "Mask horizontally"),
                ("vertical", "grip-lines-vertical", "Mask vertically"),
                ("single", "genderless", "Mask single pixel"),
            )
        }
        self.clear_button = ipw.Button(
            icon='trash', tooltip='Clear masks', layout=BUTTON_STYLE
        )
        self.continuous_button = ipw.ToggleButton(
            icon='unlock',
            value=False,
            tooltip='Enable/disable continuous masking',
            layout=BUTTON_STYLE,
        )
        self.filename_input = ipw.Text(value='', layout={"width": "150px"})
        self.save_button = ipw.Button(
            icon='save', tooltip='Save masks', layout=BUTTON_STYLE
        )

        self._lock = False
        self.mode = None
        for button in self.mode_buttons.values():
            button.observe(self._sync_modes, names='value')
        self.continuous_button.observe(self._on_continuous_toggle, names='value')

        # # self._connections = []
        # for button in self.mode_buttons.values():
        #     button.observe(self.toggle_masking, names='value')
        # self.clear_button.on_click(self.clear_masks)
        # self.save_button.on_click(self.save_masks)

        super().__init__(
            [
                ipw.HTML(value="Masking: &nbsp;&nbsp;&nbsp;"),
                *self.mode_buttons.values(),
                self.clear_button,
                self.continuous_button,
                ipw.HTML(value="&nbsp;&nbsp; Save to file: &nbsp;"),
                self.filename_input,
                self.save_button,
            ],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )

    def _sync_modes(self, change: dict):
        if self._lock:
            return
        if not change['new']:
            self.mode = None
            return
        self._lock = True
        for mode, button in self.mode_buttons.items():
            if button.value:
                if button is change['owner']:
                    self.mode = mode
                else:
                    button.value = False
        self._lock = False

    def _on_continuous_toggle(self, change: dict):
        if change['new']:
            self.continuous_button.icon = 'lock'
        else:
            self.continuous_button.icon = 'unlock'

    def on_change(self, callback: callable):
        for button in self.mode_buttons.values():
            button.observe(callback, names='value')

    def on_clear(self, callback: callable):
        self.clear_button.on_click(callback)

    def on_save(self, callback: callable):
        self.save_button.on_click(callback)

    # def toggle_masking(self, _) -> None:
    #     if self.mode is not None:
    #         self._connections = {
    #             key: f.fig.canvas.mpl_connect('button_press_event', self.handle_click)
    #             for key, f in self._tab_figs.items()
    #         }
    #         self._previous_pixel = None
    #     else:
    #         for key, conn in self._connections.items():
    #             f = self._tab_figs[key]
    #             f.fig.canvas.mpl_disconnect(conn)
    #         self._connections = {}

    # def handle_click(self, event: MouseEvent) -> None:
    #     self._previous_pixel = None
    #     self.add_mask(event)

    #     # if not self.mask_control.continuous_check.value:
    #     #     # self.add_mask(event)
    #     #     # self._previous_pixel = None
    #     #     return

    #     # if self._previous_pixel is None:
    #     #     self.add_mask(event)
    #     if self.continuous_button.value:
    #         self._mouse_move_connection = event.canvas.mpl_connect(
    #             'motion_notify_event', self.add_mask
    #         )
    #         self._mouse_release_connection = event.canvas.mpl_connect(
    #             'button_release_event', self.stop_continuous_mask
    #         )

    # def stop_continuous_mask(self, event: MouseEvent) -> None:
    #     event.canvas.mpl_disconnect(self._mouse_move_connection)
    #     event.canvas.mpl_disconnect(self._mouse_release_connection)
    #     self._mouse_move_connection = None
    #     self._mouse_release_connection = None
    #     # self._previous_pixel = None

    # def add_mask(self, event: MouseEvent) -> None:
    #     if event.inaxes is None:
    #         return

    #     key = get_current_key(self._tabs)

    #     # Ignore the event if the pan/zoom tool is active
    #     if self._tab_figs[key].toolbar.tools['panzoom'].value is not None:
    #         return

    #     x = round(event.xdata)
    #     y = round(event.ydata)
    #     # masking_mode = self.mode
    #     if self._previous_pixel is None:
    #         self._previous_pixel = (x, y)
    #     else:
    #         if (self.mode == "single") and ((x, y) == self._previous_pixel):
    #             return
    #         if (self.mode == "horizontal") and (y == self._previous_pixel[1]):
    #             return
    #         if (self.mode == "vertical") and (x == self._previous_pixel[0]):
    #             return

    #     xdim = self._tab_figs[key].view._dims['x']
    #     ydim = self._tab_figs[key].view._dims['y']

    #     # Find slider dims and indices
    #     slices = []
    #     if self.mode in ("horizontal", "single"):
    #         slices.append((ydim, y))
    #     if self.mode in ("vertical", "single"):
    #         slices.append((xdim, x))
    #     slices.extend([sl.value for sl in self._bank_dim_sliders[key]])

    #     m = slice_mask(self._masks[key], slices)
    #     m.values[...] = False if m.values.any() else True
    #     self._mask_selection_nodes[key].notify_children(message=None)

    #     # Set the previous pixel to the current one
    #     self._previous_pixel = (x, y)

    # def clear_masks(self, _) -> None:
    #     ind = self.tabs.selected_index
    #     if ind == 0:
    #         keys = self._masks.keys()
    #         slices = [sl.value for sl in self._all_dim_sliders]
    #     else:
    #         keys = [self._tab_index_to_key[ind]]
    #         slices = [sl.value for sl in self._bank_dim_sliders[keys[0]]]

    #     for key in keys:
    #         m = slice_mask(self._masks[key], slices)
    #         m.values[...] = False
    #         self._all_mask_selection_nodes[key].notify_children(message=None)
    #         self._mask_selection_nodes[key].notify_children(message=None)

    # def save_masks(self, _) -> None:
    #     filename = self.filename_input.value
    #     if not filename:
    #         return
    #     sc.DataGroup(self._masks).save_hdf5(filename)


class ColorControl(ipw.HBox):
    def __init__(self, log: bool = False):

        # self._tabs = tabs
        # self._tab_figs = tab_figs
        # self._subplots = subplots
        self._lock = False

        self.vmin = ipw.Text(
            description='Min:',
            layout={'width': '130px'},
            style={"description_width": 'initial'},
        )
        self.vmax = ipw.Text(
            description='Max:',
            layout={'width': '130px'},
            style={"description_width": 'initial'},
        )
        # self.vmin.observe(self.update_color_range, names='value')
        # self.vmax.observe(self.update_color_range, names='value')

        self.log_button = ipw.ToggleButton(
            description="log", layout=BUTTON_STYLE, value=log
        )
        # self.log_button.observe(self.toggle_log, names="value")

        self.fit_button = ipw.Button(description="fit", layout=BUTTON_STYLE)
        # self.fit_button.on_click(self.fit_cmap_range)

        super().__init__(
            [
                ipw.Label("Colorscale:"),
                self.vmin,
                self.vmax,
                self.log_button,
                self.fit_button,
            ],
            layout={
                'border': '1px solid black',
                'padding': '0px 10px 0px 10px',
            },
        )

    def on_limit_change(self, callback: Callable) -> None:
        self.vmin.observe(callback, names='value')
        self.vmax.observe(callback, names='value')

    def on_log_toggle(self, callback: Callable) -> None:
        self.log_button.observe(callback, names='value')

    def on_fit_click(self, callback: Callable) -> None:
        self.fit_button.on_click(callback)

    # def update_colormapper_norm(self) -> None:
    #     key = get_current_key(self._tabs)
    #     # ind = self._tabs.selected_index
    #     if key == "All":
    #         figs = self._subplots.values()
    #     else:
    #         figs = [self._tab_figs[key]]
    #     for f in figs:
    #         f.view.colormapper.norm = "log" if self.log_button.value else "linear"

    # def toggle_log(self, _: dict) -> None:
    #     self.reset_cmap_range()
    #     self.update_colormapper_norm()

    # def update_color_range(self, _=None) -> None:
    #     bounds = {
    #         key: float(item.value) if len(item.value) > 0 else None
    #         for key, item in [("cmin", self.vmin), ("cmax", self.vmax)]
    #     }

    #     if self.log_button.value:
    #         for bound in bounds.values():
    #             if (bound is not None) and (bound <= 0):
    #                 return

    #     key = get_current_key(self._tabs)

    #     if key == "All":
    #         figs = list(self._subplots.values())
    #     else:
    #         figs = [self._tab_figs[key]]

    #     for f in figs:
    #         for bound, new in bounds.items():
    #             if new is None:
    #                 getattr(f.view.colormapper, f"_{bound}").pop('user', None)
    #             else:
    #                 setattr(f.view.colormapper, bound, new)
    #         if not self._lock:
    #             f.view.colormapper.autoscale()
    #     if not self._lock:
    #         # If this is the subplots, by taking the first figure and calling draw,
    #         # it is actually calling draw on the main figure canvas so it updates all
    #         # subplots in one go.
    #         figs[0].canvas.draw()

    #         # if is_subplots:
    #         #     self.main_figure.canvas.draw_idle()
    #         # else:
    #         #     self.tab_figs[key].canvas.draw()

    # def reset_cmap_range(self, _=None) -> None:
    #     self._lock = True
    #     self.vmin.value = ''
    #     self.vmax.value = ''
    #     self._lock = False

    # def fit_cmap_range(self, _=None) -> None:
    #     # ind = self.tabs.selected_index
    #     # key = self._tab_index_to_key[ind]
    #     self.reset_cmap_range()

    #     key = get_current_key(self._tabs)

    #     if key == "All":
    #         figs = list(self._subplots.values())
    #     else:
    #         figs = [self._tab_figs[key]]
    #     for f in figs:
    #         f.view.colormapper.autoscale()
    #     figs[0].canvas.draw()
    #     # if ind == 0:
    #     #     self.main_figure.canvas.draw_idle()
    #     # else:
    #     #     self.tab_figs[key].canvas.draw()


class DimSlider(ipw.HBox, ipw.ValueWidget):
    def __init__(self, sizes: dict[str, int], dim: str, visible_dims: list[str]):
        self._lock = False
        self._sizes = sizes
        self._visible_dims = visible_dims
        self._dim = dim

        self.slider = ipw.IntSlider(
            description=f"{dim} ",
            min=0,
            max=self._sizes[dim] - 1,
            value=0,
            style={"description_width": "initial"},
        )
        self.sum = ipw.Checkbox(
            description="sum", value=False, indent=False, layout={"width": "initial"}
        )

        self._link = ipw.jslink((self.sum, 'value'), (self.slider, 'disabled'))
        self.value = (self.dim, self.slider.value)

        self.slider.observe(self._on_subwidget_change, names='value')
        self.sum.observe(self._on_subwidget_change, names='value')

        super().__init__(
            [self.slider, self.sum],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )

    @property
    def dim(self):
        return self._dim

    def _on_subwidget_change(self, _=None):
        if self.sum.value:
            self.value = (self.dim, None)
        else:
            self.value = (self.dim, self.slider.value)


def _add_missing_coordinates(da: sc.DataArray) -> sc.DataArray:
    return da.assign_coords({dim: sc.arange(dim, sz) for dim, sz in da.sizes.items()})


def _slice_or_reduce(
    da: sc.DataArray, slices: list[tuple[str, int | None]], op: Literal["sum", "all"]
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
                out = sc.full_like(
                    out[dim, 0], value={"sum": float("nan"), "all": False}[op]
                )
            else:
                out = out[dim, ind]
    return out


def _slice_mask(mask: sc.Variable, slices: list[tuple[str, slice]]) -> sc.Variable:
    out = mask
    for s in [sli for sli in slices if sli[1] is not None]:
        out = out[s]
    return out


def _box_of_sliders(sliders: list[DimSlider]) -> ipw.HBox:
    return ipw.HBox(
        [
            widget
            for dslider in sliders
            for widget in (dslider, ipw.HTML('<div style="width: 20px;"></div>'))
        ]
    )


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
        figsize:
            Size of the figures in inches (width, height).
        show_dims:
            List of dimensions to display. If ``None``, the last two dimensions are
            displayed.
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
        # easily visible on the figures. Instead, we slice the masks as well and only
        # apply them just before displaying the data on the figures.
        self.mask_selection_nodes = {
            key: pp.Node(lambda masks, key: masks[key], masks=self.masks, key=key)
            for key in self.data.keys()
        }
        self.slice_masks_nodes = {
            key: pp.Node(
                _slice_or_reduce,
                da=self.mask_selection_nodes[key],
                slices=self.collect_nodes[key],
                op="all",
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
        self.all_mask_selection_nodes = {
            key: pp.Node(lambda masks, key: masks[key], masks=self.masks, key=key)
            for key in self.data.keys()
        }
        self.all_slice_masks_nodes = {
            key: pp.Node(
                _slice_or_reduce,
                da=self.all_mask_selection_nodes[key],
                slices=self.all_collect_node,
                op="all",
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

        with plt.ioff():
            ncols = 3
            nrows = math.ceil(len(self.data) / ncols)
            self.main_figure, axs = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=figsize
            )

        self.subplots = {}
        self.tab_figs = {}
        _kwargs = {**{"hide_log_buttons": True, "autoscale": False}, **kwargs}
        for i, (bank, ax) in enumerate(
            zip(self.data.keys(), axs.flatten(), strict=True)
        ):
            self.subplots[bank] = pp.imagefigure(
                self.all_apply_masks_nodes[bank],
                ax=ax,
                title=bank,
                cbar=True,
                **_kwargs,
            )
            ax._bank_explorer_subplot_index = i
            self.tab_figs[bank] = pp.imagefigure(
                self.apply_masks_nodes[bank],
                title=bank,
                cbar=True,
                # Need to make some horizontal space for the toolbar
                figsize=(0.96 * figsize[0], figsize[1]),
                **_kwargs,
            )

        self.main_figure.canvas.header_visible = False
        self._tab_switch_connection = self.main_figure.canvas.mpl_connect(
            'button_press_event', self.switch_to_tab
        )

        # # Create color map controls
        # self.cmap_vmin = ipw.Text(
        #     description='Min:',
        #     layout={'width': '130px'},
        #     style={"description_width": 'initial'},
        # )
        # self.cmap_vmax = ipw.Text(
        #     description='Max:',
        #     layout={'width': '130px'},
        #     style={"description_width": 'initial'},
        # )
        # self.cmap_vmin.observe(self.update_color_range, names='value')
        # self.cmap_vmax.observe(self.update_color_range, names='value')

        # is_log = next(
        #     iter({kwargs.get('norm'), kwargs.get('logc')} - {None}), None
        # ) in ('log', True)
        # self.log_button = ipw.ToggleButton(
        #     description="log", layout=BUTTON_STYLE, value=is_log
        # )
        # self.log_button.observe(self.toggle_log, names="value")

        # self.fit_button = ipw.Button(description="fit", layout=BUTTON_STYLE)
        # self.fit_button.on_click(self.fit_cmap_range)

        # space = ipw.HTML('<div style="width: 20px;"></div>')

        self.tabs = ipw.Tab(
            layout={'width': f'{figsize[0] * self.main_figure.get_dpi() + 40}px'}
        )

        self.tabs.children = [
            ipw.VBox([_box_of_sliders(self.all_dim_sliders), self.main_figure.canvas]),
            *[
                ipw.VBox([_box_of_sliders(self.bank_dim_sliders[key]), f])
                for key, f in zip(self.data.keys(), self.tab_figs.values(), strict=True)
            ],
        ]

        # self._tab_index_to_key = {
        #     0: "All",
        #     **{i + 1: key for i, key in enumerate(self.data.keys())},
        # }

        self.tabs.titles = ["All", *self.data.keys()]
        self.tabs.observe(self.selected_tab_changed, names='selected_index')

        self.mask_control = MaskControl()
        self._mask_connections = {}
        self.mask_control.on_change(self.toggle_masking)
        self.mask_control.on_clear(self.clear_masks)
        self.mask_control.on_save(self.save_masks)

        self.color_control = ColorControl()
        self.color_control.on_limit_change(self.update_color_range)
        self.color_control.on_log_toggle(self.toggle_log)
        self.color_control.on_fit_click(self.fit_cmap_range)

        super().__init__(
            [
                ipw.HBox(
                    [self.mask_control, self.color_control],
                    layout={
                        'width': self.tabs.layout.width,
                        'justify_content': 'space-between',
                        'align_items': 'center',
                    },
                ),
                self.tabs,
            ]
        )

    @property
    def current_tab(self) -> int:
        return self.tabs.selected_index

    @property
    def current_key(self) -> str:
        return self.tabs.titles[self.current_tab]

    @property
    def current_figs(self) -> list:
        key = self.current_key
        if key == "All":
            return list(self.subplots.values())
        else:
            return [self.tab_figs[key]]

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

    def selected_tab_changed(self, change: dict) -> None:
        """
        Handle the event when the selected tab changes.

        Parameters
        ----------
        change:
            The change dictionary containing information about the tab change.
        """
        self.update_colormapper_norm()
        self.update_color_range()
        self.update_mask_connections()

        ind = change["new"]
        if ind == 0:
            old = change["old"]
            key = self.tabs.titles[old]
            self.all_mask_selection_nodes[key].notify_children(message=None)
        else:
            self.mask_selection_nodes[self.current_key].notify_children(message=None)

    def update_colormapper_norm(self) -> None:
        for f in self.current_figs:
            f.view.colormapper.norm = (
                "log" if self.color_control.log_button.value else "linear"
            )

    def toggle_log(self, _: dict) -> None:
        self.reset_cmap_range()
        self.update_colormapper_norm()

    def update_color_range(self, _=None) -> None:
        bounds = {
            key: float(item.value) if len(item.value) > 0 else None
            for key, item in [
                ("cmin", self.color_control.vmin),
                ("cmax", self.color_control.vmax),
            ]
        }

        if self.color_control.log_button.value:
            for bound in bounds.values():
                if (bound is not None) and (bound <= 0):
                    return

        # ind = self.current_tab
        # key = self._tab_index_to_key[ind]

        # if ind == 0:
        #     figs = self.subplots.values()
        # else:
        #     figs = [self.tab_figs[key]]

        # key = get_current_key(self._tabs)
        # if key == "All":
        #     figs = list(self._subplots.values())
        # else:
        #     figs = [self._tab_figs[key]]

        figs = self.current_figs
        for f in figs:
            for bound, new in bounds.items():
                if new is None:
                    getattr(f.view.colormapper, f"_{bound}").pop('user', None)
                else:
                    setattr(f.view.colormapper, bound, new)
            if not self._lock:
                f.view.colormapper.autoscale()
        if not self._lock:
            # If this is the subplots, by taking the first figure and calling draw,
            # it is actually calling draw on the main figure canvas so it updates all
            # subplots in one go.
            figs[0].canvas.draw()
            # if ind == 0:
            #     self.main_figure.canvas.draw_idle()
            # else:
            #     self.tab_figs[key].canvas.draw()

    def reset_cmap_range(self, _=None) -> None:
        self._lock = True
        self.color_control.vmin.value = ''
        self.color_control.vmax.value = ''
        self._lock = False

    def fit_cmap_range(self, _=None) -> None:
        # ind = self.tabs.selected_index
        # key = self._tab_index_to_key[ind]

        self.reset_cmap_range()
        # key = get_current_key(self.tabs)

        # if key == "All":
        #     figs = list(self.subplots.values())
        # else:
        #     figs = [self.tab_figs[key]]

        figs = self.current_figs

        for f in figs:
            f.view.colormapper.autoscale()
        figs[0].canvas.draw()

        # if ind == 0:
        #     figs = self.subplots.values()
        # else:
        #     figs = [self.tab_figs[key]]
        # for f in figs:
        #     f.view.colormapper.autoscale()
        # if ind == 0:
        #     self.main_figure.canvas.draw_idle()
        # else:
        #     self.tab_figs[key].canvas.draw()

    def update_mask_connections(self) -> None:
        self.clear_mask_connections()
        self._mask_connections.update(
            {
                key: f.fig.canvas.mpl_connect(
                    'button_press_event', self.handle_mask_click
                )
                for key, f in self.tab_figs.items()
            }
        )

    def clear_mask_connections(self) -> None:
        if not self._mask_connections:
            return
        for key, conn in self._mask_connections.items():
            f = self.tab_figs[key]
            f.fig.canvas.mpl_disconnect(conn)
        self._mask_connections.clear()

    def toggle_masking(self, _) -> None:
        if self.mask_control.mode is not None:
            if not self._mask_connections:
                self.update_mask_connections()
            self._previous_pixel = None
        else:
            self.clear_mask_connections()

    def handle_mask_click(self, event: MouseEvent) -> None:
        self._previous_pixel = None
        self.add_mask(event)

        # if not self.mask_control.continuous_check.value:
        #     # self.add_mask(event)
        #     # self._previous_pixel = None
        #     return

        # if self._previous_pixel is None:
        #     self.add_mask(event)
        if self.mask_control.continuous_button.value:
            self._mouse_move_connection = event.canvas.mpl_connect(
                'motion_notify_event', self.add_mask
            )
            self._mouse_release_connection = event.canvas.mpl_connect(
                'button_release_event', self.stop_continuous_mask
            )

    def stop_continuous_mask(self, event: MouseEvent) -> None:
        event.canvas.mpl_disconnect(self._mouse_move_connection)
        event.canvas.mpl_disconnect(self._mouse_release_connection)
        self._mouse_move_connection = None
        self._mouse_release_connection = None
        # self._previous_pixel = None

    def add_mask(self, event: MouseEvent) -> None:
        if event.inaxes is None:
            return

        # Current tab index
        # key = self._tab_index_to_key[self.tabs.selected_index]
        key = self.current_key

        # Ignore the event if the pan/zoom tool is active
        if self.tab_figs[key].toolbar.tools['panzoom'].value is not None:
            return

        x = round(event.xdata)
        y = round(event.ydata)
        masking_mode = self.mask_control.mode
        if self._previous_pixel is None:
            self._previous_pixel = (x, y)
        else:
            if (masking_mode == "single") and ((x, y) == self._previous_pixel):
                return
            if (masking_mode == "horizontal") and (y == self._previous_pixel[1]):
                return
            if (masking_mode == "vertical") and (x == self._previous_pixel[0]):
                return

        xdim = self.tab_figs[key].view._dims['x']
        ydim = self.tab_figs[key].view._dims['y']

        # Find slider dims and indices
        slices = []
        if masking_mode in ("horizontal", "single"):
            slices.append((ydim, y))
        if masking_mode in ("vertical", "single"):
            slices.append((xdim, x))
        slices.extend([sl.value for sl in self.bank_dim_sliders[key]])

        m = _slice_mask(self.masks[key], slices)
        m.values[...] = False if m.values.all() else True
        self.mask_selection_nodes[key].notify_children(message=None)

        self._previous_pixel = (x, y)

    def clear_masks(self, _) -> None:
        # ind = self.tabs.selected_index
        if self.current_tab == 0:
            keys = self.masks.keys()
            slices = [sl.value for sl in self.all_dim_sliders]
        else:
            keys = [self.current_key]
            slices = [sl.value for sl in self.bank_dim_sliders[keys[0]]]

        for key in keys:
            m = _slice_mask(self.masks[key], slices)
            m.values[...] = False
            self.all_mask_selection_nodes[key].notify_children(message=None)
            self.mask_selection_nodes[key].notify_children(message=None)

    def save_masks(self, _) -> None:
        filename = self.mask_control.filename_input.value
        if not filename:
            return
        sc.DataGroup(self.masks).save_hdf5(filename)
