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


class MaskControl(ipw.HBox):
    def __init__(self):
        """
        Widget for controlling masking options.
        The widget includes buttons for selecting masking modes (horizontal, vertical,
        single pixel), and options for clearing and saving masks.
        """

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
            icon='save', tooltip='Save masks', layout=BUTTON_STYLE, disabled=True
        )
        self.filename_input.observe(
            lambda change: setattr(
                self.save_button, 'disabled', not bool(change['new'])
            ),
            names='value',
        )

        self._lock = False
        for button in self.mode_buttons.values():
            button.observe(self._sync_modes, names='value')
        self.continuous_button.observe(self._on_continuous_toggle, names='value')

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

    def _sync_modes(self, change: dict) -> None:
        if self._lock:
            return
        if not change['new']:
            return
        self._lock = True
        for button in self.mode_buttons.values():
            if button.value and (button is not change['owner']):
                button.value = False
        self._lock = False

    @property
    def mode(self) -> str | None:
        for mode, button in self.mode_buttons.items():
            if button.value:
                return mode
        return None

    @mode.setter
    def mode(self, value: str | None) -> None:
        if value is None:
            for button in self.mode_buttons.values():
                button.value = False
        else:
            if value not in self.mode_buttons:
                raise ValueError(f"Invalid mode: {value}")
            self.mode_buttons[value].value = True

    def _on_continuous_toggle(self, change: dict) -> None:
        if change['new']:
            self.continuous_button.icon = 'lock'
        else:
            self.continuous_button.icon = 'unlock'

    def on_clear(self, callback: callable) -> None:
        """
        Register a callback to be called when the clear button is clicked.
        """
        self.clear_button.on_click(callback)

    def on_save(self, callback: callable) -> None:
        """
        Register a callback to be called when the save button is clicked.
        """
        self.save_button.on_click(callback)


class ColorControl(ipw.HBox):
    def __init__(self, log: bool = False):
        """
        The color control widget allows controlling the color scale limits and toggling
        logarithmic scaling.

        Parameters
        ----------
        log:
            Whether to enable logarithmic scaling initially, by default ``False``.
        """
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
        self.log_button = ipw.ToggleButton(
            description="log", layout=BUTTON_STYLE, value=log
        )
        self.fit_button = ipw.Button(description="fit", layout=BUTTON_STYLE)

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
        """
        Register a callback to be called when the color scale limits are changed.
        """
        self.vmin.observe(callback, names='value')
        self.vmax.observe(callback, names='value')

    def on_log_toggle(self, callback: Callable) -> None:
        """
        Register a callback to be called when the logarithmic toggle button is clicked.
        """
        self.log_button.observe(callback, names='value')

    def on_fit_click(self, callback: Callable) -> None:
        """
        Register a callback to be called when the fit button is clicked.
        """
        self.fit_button.on_click(callback)


class DimSlider(ipw.HBox, ipw.ValueWidget):
    def __init__(self, sizes: dict[str, int], dim: str, visible_dims: list[str]):
        """
        The DimSlider widget is used to control at which index to slice a particular
        dimension of a data array. In addition, it provides an option to sum over the
        dimension instead of selecting a specific index.

        Parameters
        ----------
        sizes:
            A dictionary mapping dimension names to their sizes.
        dim:
            The dimension this slider controls.
        visible_dims:
            A list of currently visible dimensions.
        """
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
    """
    Add missing coordinates to a DataArray. The coordinates are created as arange
    arrays based on the size of each dimension.

    Parameters
    ----------
    da:
        The DataArray to process.
    """
    return da.assign_coords({dim: sc.arange(dim, sz) for dim, sz in da.sizes.items()})


def _slice_or_reduce(
    da: sc.DataArray, slices: list[tuple[str, int | None]], op: Literal["sum", "all"]
) -> sc.DataArray:
    """
    Slice or reduce a DataArray based on the provided slices and operation.
    If the index for a dimension is ``None``, the specified operation is applied along
    that dimension. Otherwise, the DataArray is sliced at the given index.
    If an index is out of bounds for a dimension, a blank DataArray is created to avoid
    indexing errors.

    Parameters
    ----------
    da:
        The DataArray to process.
    slices:
        A list of tuples, each containing a dimension name and an index or ``None``.
    op:
        The reduction operation to apply when the index is ``None``. Supported values
        are ``"sum"`` and ``"all"``.
    """
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
        """GUI to view and mask detector banks.

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
        self.main_figure.canvas.mpl_connect('button_press_event', self.switch_to_tab)

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
        self.tabs.titles = ["All", *self.data.keys()]
        self.tabs.observe(self.selected_tab_changed, names='selected_index')

        self.mask_control = MaskControl()
        # self.mask_control.on_change(self.toggle_masking)
        self.mask_control.on_clear(self.clear_masks)
        self.mask_control.on_save(self.save_masks)
        for f in self.tab_figs.values():
            f.fig.canvas.mpl_connect('button_press_event', self.handle_mask_click)

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
        """
        Get the index of the currently selected tab.
        """
        return self.tabs.selected_index

    @property
    def current_key(self) -> str:
        """
        Get the key corresponding to the currently selected tab.
        """
        return self.tabs.titles[self.current_tab]

    @property
    def current_figs(self) -> list:
        """
        Get the list of figures corresponding to the currently selected tab.
        If the "All" tab is selected, return all subplots; otherwise, return the
        a list with a single item containing the figure for the current tab.
        """
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
        # Ignore the event if it occurred outside any axes
        if event.inaxes is None:
            return
        # Ignore the event if the pan/zoom tool is active
        if event.inaxes.get_navigate_mode() is not None:
            return
        if hasattr(event.inaxes, '_bank_explorer_subplot_index'):
            index = event.inaxes._bank_explorer_subplot_index
            self.tabs.selected_index = index + 1

    def _get_old_and_new_sliders(self, old_tab: int, new_tab: int):
        old_key = self.tabs.titles[old_tab]
        new_key = self.tabs.titles[new_tab]
        old_sliders = (
            self.all_dim_sliders if old_tab == 0 else self.bank_dim_sliders[old_key]
        )
        new_sliders = (
            self.all_dim_sliders if new_tab == 0 else self.bank_dim_sliders[new_key]
        )
        return old_sliders, new_sliders

    def update_slider_sums(self, old_tab: int, new_tab: int) -> None:
        old_sliders, new_sliders = self._get_old_and_new_sliders(old_tab, new_tab)
        old_slider_sums = {sl.dim: sl.sum.value for sl in old_sliders}
        for sl in new_sliders:
            if sl.sum.value != old_slider_sums[sl.dim]:
                sl.sum.value = old_slider_sums[sl.dim]

    def update_slider_inds(self, old_tab: int, new_tab: int) -> None:
        old_sliders, new_sliders = self._get_old_and_new_sliders(old_tab, new_tab)
        old_slider_inds = dict(slider.value for slider in old_sliders)
        for sl in new_sliders:
            if sl.sum.value:
                continue
            new_value = min(old_slider_inds[sl.dim], sl.slider.max)
            if sl.slider.value != new_value:
                sl.slider.value = new_value

    def selected_tab_changed(self, change: dict) -> None:
        """
        When the selected tab changes, update the color range and normalization.
        In addition, if we switched from a single figure tab to the "All" tab, we update
        the slider positions to match those from the previously selected tab.
        If we switched from the "All" tab to a single figure tab, we update the slider
        positions for the new tab based on the previous "All" tab slider positions,
        without exceeding the maximum allowed values for the sliders of the new tab.

        Parameters
        ----------
        change:
            The change dictionary containing information about the tab change.
        """
        self.update_colormapper_norm()
        self.update_color_range()

        new_tab, old_tab = change["new"], change["old"]
        self.update_slider_sums(old_tab, new_tab)
        if 0 in (old_tab, new_tab):
            self.update_slider_inds(old_tab, new_tab)

    def update_colormapper_norm(self) -> None:
        """
        Update the colormapper normalization (logarithmic or linear) for all current
        figures.
        """
        for f in self.current_figs:
            f.view.colormapper.norm = (
                "log" if self.color_control.log_button.value else "linear"
            )

    def toggle_log(self, _: dict) -> None:
        """
        Toggle the colormapper normalization between logarithmic and linear.
        """
        self.reset_cmap_range()
        self.update_colormapper_norm()

    def update_color_range(self, _=None) -> None:
        """
        Update the color range (vmin and vmax) for all current figures based on the
        values in the color control widgets.
        """
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

    def reset_cmap_range(self, _=None) -> None:
        """
        Reset the color map range (vmin and vmax) to their default state.
        """
        self._lock = True
        self.color_control.vmin.value = ''
        self.color_control.vmax.value = ''
        self._lock = False

    def fit_cmap_range(self, _=None) -> None:
        """
        Fit the color map range (vmin and vmax) to the data for all current figures.
        """
        self.reset_cmap_range()
        figs = self.current_figs
        for f in figs:
            f.view.colormapper.autoscale()
        figs[0].canvas.draw()

    def handle_mask_click(self, event: MouseEvent) -> None:
        """
        If the mouse button is clicked within an axes and a masking mode is selected,
        initiate the masking process:
        - We add a mask at the current position.
        - If continuous masking is enabled, we connect mouse move and release events to
          allow continuous masking.
        """
        # Ignore the event if it occurred outside any axes
        if event.inaxes is None:
            return
        # Ignore the event if the pan/zoom tool is active
        if event.inaxes.get_navigate_mode() is not None:
            return
        # Ignore the event if no masking mode is selected
        if self.mask_control.mode is None:
            return

        self._previous_pixel = None
        self.add_mask(event)

        if self.mask_control.continuous_button.value:
            self._mouse_move_connection = event.canvas.mpl_connect(
                'motion_notify_event', self.add_mask
            )
            self._mouse_release_connection = event.canvas.mpl_connect(
                'button_release_event', self.stop_continuous_mask
            )

    def stop_continuous_mask(self, event: MouseEvent) -> None:
        """
        Stop the continuous masking process by disconnecting the mouse move and release
        events.
        """
        event.canvas.mpl_disconnect(self._mouse_move_connection)
        event.canvas.mpl_disconnect(self._mouse_release_connection)
        self._mouse_move_connection = None
        self._mouse_release_connection = None

    def add_mask(self, event: MouseEvent) -> None:
        """
        Add a mask at the current mouse position based on the selected masking mode.
        If continuous masking is enabled, this method will be called repeatedly as the
        mouse moves.
        """
        # Still need to protect against outside of axes events as it may come from
        # other events like mouse move during continuous masking.
        if event.inaxes is None:
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

        key = self.current_key
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
        """
        Clear currently visible masks for the current tab or the entire dataset.
        """
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
        """
        Save the current masks to an HDF5 file specified by the filename input.
        If no filename is provided, the method will return without saving. This should
        in principle never happen because the save button should be disabled when no
        filename is provided.
        """
        filename = self.mask_control.filename_input.value
        if not filename:
            return
        sc.DataGroup(self.masks).save_hdf5(filename)
