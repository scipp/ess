# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import math
from collections.abc import Callable
from types import MappingProxyType

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import plopp as pp
import scipp as sc
from matplotlib.backend_bases import MouseEvent
from matplotlib.gridspec import GridSpec
from plopp.graphics.bbox import axis_bounds

from ..masking import write_pixel_masks

BUTTON_STYLE = {"width": "3.0em"}
THIN_BUTTON_STYLE = {'width': '1.23em', 'padding': '0px'}


class MaskControl(ipw.HBox):
    def __init__(
        self,
        mask_dims: MappingProxyType[str, MappingProxyType[str, tuple[str, ...]]],
    ):
        """
        Widget for controlling masking options.

        Parameters
        ----------
        mask_dims:
            Mapping of mask names to mask dimensions.
        """
        self.mask_dims = mask_dims

        self.selector = ipw.Dropdown(
            options=["", *self.mask_dims.keys()],
            value="",
            description='Edit mask:',
            layout={'width': '200px'},
        )
        self.clear_button = ipw.Button(
            icon='trash', tooltip='Clear masks', layout=BUTTON_STYLE, disabled=True
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

        super().__init__(
            [
                self.selector,
                self.clear_button,
                ipw.HTML(value="&nbsp;&nbsp; Save to file: &nbsp;"),
                self.filename_input,
                self.save_button,
            ],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )

    @property
    def selected(self) -> str:
        return self.selector.value

    def select(self, mask: str) -> None:
        self.selector.value = mask

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
        self._vmin = ipw.Text(
            description='Min:',
            layout={'width': '130px'},
            style={"description_width": 'initial'},
            continuous_update=False,
        )
        self._vmax = ipw.Text(
            description='Max:',
            layout={'width': '130px'},
            style={"description_width": 'initial'},
            continuous_update=False,
        )
        self.log_button = ipw.ToggleButton(
            description="log", layout=BUTTON_STYLE, value=log
        )
        self.fit_button = ipw.Button(description="fit", layout=BUTTON_STYLE)

        super().__init__(
            [
                ipw.Label("Colorscale:"),
                self._vmin,
                self._vmax,
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
        self._vmin.observe(callback, names='value')
        self._vmax.observe(callback, names='value')

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

    @property
    def vmin(self) -> float | None:
        try:
            return float(self._vmin.value)
        except ValueError:
            return None

    @vmin.setter
    def vmin(self, value: float) -> None:
        if np.isfinite(value):
            self._vmin.value = str(value)

    @property
    def vmax(self) -> float | None:
        try:
            return float(self._vmax.value)
        except ValueError:
            return None

    @vmax.setter
    def vmax(self, value: float) -> None:
        if np.isfinite(value):
            self._vmax.value = str(value)


class DimSlider(ipw.HBox, ipw.ValueWidget):
    def __init__(self, sizes: dict[str, int], dim: str):
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
        """
        self._sizes = sizes
        self._dim = dim

        if dim not in self._sizes:
            slider_max = None
            disabled = True
        else:
            slider_max = self._sizes[dim] - 1
            disabled = False

        self.slider = ipw.IntSlider(
            description=f"{dim} ",
            min=0,
            max=slider_max,
            value=0,
            style={"description_width": "initial"},
            disabled=disabled,
        )
        self.prev = ipw.Button(
            icon="chevron-left",
            tooltip="Move slider left",
            disabled=True,
            layout=THIN_BUTTON_STYLE,
        )
        self.next = ipw.Button(
            icon="chevron-right",
            tooltip="Move slider right",
            layout=THIN_BUTTON_STYLE,
            disabled=disabled,
        )
        self.value = (self.dim, None if disabled else self.slider.value)

        self.slider.observe(self._on_subwidget_change, names='value')
        self.slider.observe(self._update_navigation_status, names='value')
        self.prev.on_click(
            lambda _: setattr(self.slider, 'value', self.slider.value - 1)
        )
        self.next.on_click(
            lambda _: setattr(self.slider, 'value', self.slider.value + 1)
        )

        super().__init__(
            [self.slider, self.prev, self.next],
            layout={'border': '1px solid black', 'padding': '0px 10px 0px 10px'},
        )

    @property
    def dim(self):
        return self._dim

    def _on_subwidget_change(self, _=None):
        self.value = (self.dim, self.slider.value)

    def _update_navigation_status(self, _=None):
        self.prev.disabled = self.slider.value <= self.slider.min
        self.next.disabled = self.slider.value >= self.slider.max


def _add_missing_coordinates(da: sc.DataArray) -> sc.DataArray:
    """
    Add missing coordinates to a DataArray. The coordinates are created as arange
    arrays based on the size of each dimension.

    Parameters
    ----------
    da:
        The DataArray to process.
    """
    return da.assign_coords(
        {
            dim: sc.arange(dim, sz)
            for dim, sz in da.sizes.items()
            if dim not in da.coords
        }
    )


def _slice_data(da: sc.DataArray, slices: list[tuple[str, int | None]]) -> sc.DataArray:
    """
    Slice a DataArray based on the provided slices and operation.
    If an index is out of bounds for a dimension, a blank DataArray is created to avoid
    indexing errors.

    Parameters
    ----------
    da:
        The DataArray to process.
    slices:
        A list of tuples, each containing a dimension name and an index or ``None``.
    """
    out = da
    for sl in slices:
        dim, ind = sl
        if dim not in out.sizes:
            continue
        if ind >= out.sizes[dim]:
            # Make a blank data array to avoid indexing errors
            out = sc.full_like(out[dim, 0], value=float("nan"), dtype='float64')
        else:
            out = out[dim, ind]
    return out


def _apply_masks(da: sc.DataArray, masks: dict[str, sc.Variable]) -> sc.DataArray:
    return da.assign_masks(masks)


def _box_of_sliders(sliders: list[DimSlider]) -> ipw.HBox:
    return ipw.HBox(
        [
            widget
            for dslider in sliders
            for widget in (dslider, ipw.HTML('<div style="width: 20px;"></div>'))
        ]
    )


def _find_closest(a: np.ndarray, x: float) -> int:
    i = np.searchsorted(a, x)
    if i == 0:
        return 0
    if i == len(a):
        return len(a) - 1
    return i - 1 if abs(a[i - 1] - x) <= abs(a[i] - x) else i


def _slice_mask(mask: sc.Variable, slices: list[tuple[str, int]]) -> sc.Variable:
    out = mask
    for s in slices:
        if s[0] in out.dims:
            out = out[s]
    return out


def _find_limits(figs):
    vmin = np.inf
    vmax = -np.inf
    for f in figs:
        (a,) = f.view.artists.values()
        bounds = axis_bounds(keys=('cmin', 'cmax'), x=a.data, scale='linear')
        if bounds["cmin"] is not None:
            vmin = min(vmin, bounds["cmin"])
        if bounds["cmax"] is not None:
            vmax = max(vmax, bounds["cmax"])
    return vmin, vmax


def _update_view(figs):
    for f in figs:
        f.view.colormapper.autoscale()
    figs[0].canvas.draw()


class DetectorExplorer(ipw.VBox):
    def __init__(
        self,
        data: sc.DataGroup | sc.DataArray,
        masks: MappingProxyType[str, sc.Variable]
        | MappingProxyType[str, MappingProxyType[str, sc.Variable]]
        | None = None,
        figsize: tuple[int, int] = (11, 8),
        show_dims: list[str] | None = None,
        **kwargs,
    ):
        """GUI to view and mask detector banks.

        Parameters
        ----------
        data:
            DataGroup or DataArray containing detector banks.
        masks:
            Dictionary of masks to apply to the data.
            Keys should match the detector bank names, and inside each bank,
            the keys should be the mask names.
            In the case of a single detector bank, the keys should just be the mask
            names themselves.
        figsize:
            Size of the figures in inches (width, height).
        show_dims:
            List of dimensions to display. If ``None``, the last two dimensions are
            displayed.
        kwargs:
            Additional arguments are forwarded to the plotting function.
        """
        self._lock = False

        if not hasattr(data, 'items'):
            key = data.name or 'detector_0'
            data = sc.DataGroup({key: data})

        if (masks is not None) and isinstance(next(iter(masks.values())), sc.Variable):
            masks = {key: masks}

        self.data = sc.DataGroup(
            {k: _add_missing_coordinates(da) for k, da in data.items()}
        )

        self.original_masks = masks
        self.masks = sc.DataGroup({k: sc.DataGroup() for k in self.data.keys()})
        if masks is not None:
            mask_dims = {}
            # Make a set of all mask names to check for consistency across banks.
            all_mask_names = {
                n: m.dims for group in masks.values() for n, m in group.items()
            }
            # Make empty masks everywhere to begin with
            for k, da in self.data.items():
                for name, dims in all_mask_names.items():
                    self.masks[k][name] = sc.zeros(
                        sizes={dim: da.sizes[dim] for dim in dims}, dtype=bool
                    )

            for key, group in masks.items():
                for name, mask in group.items():
                    if name not in mask_dims:
                        mask_dims[name] = mask.dims
                    elif mask.dims != mask_dims[name]:
                        raise ValueError(
                            f"Mask '{name}' has inconsistent dimensions: {mask.dims}, "
                            f"expected: {mask_dims[name]}"
                        )
                    self.masks[key][name] = mask.copy(deep=True)

        self.max_sizes = {
            dim: max(da.sizes.get(dim, 0) for da in self.data.values())
            for dim in self.data.dims
        }
        self._visible_dims = show_dims
        if self._visible_dims is None:
            self._visible_dims = list(self.data.dims)[-2:]

        if len(self._visible_dims) != 2:
            raise ValueError("Exactly two visible dimensions must be specified.")

        other_dims = [dim for dim in self.data.dims if dim not in self._visible_dims]

        self.bank_dim_sliders = {
            key: [DimSlider(sizes=self.data[key].sizes, dim=dim) for dim in other_dims]
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

        self.apply_masks_nodes = {
            key: pp.Node(
                _apply_masks, da=self.bank_selection_nodes[key], masks=self.masks[key]
            )
            for key in self.data.keys()
        }
        self.slice_data_nodes = {
            key: pp.Node(
                _slice_data,
                da=self.apply_masks_nodes[key],
                slices=self.collect_nodes[key],
            )
            for key in self.data.keys()
        }
        self.to_plot_nodes = {
            key: pp.Node(
                lambda da: da.transpose(self._visible_dims),
                da=self.slice_data_nodes[key],
            )
            for key in self.data.keys()
        }

        # There is a significant performance penalty in Matplotlib if we display the
        # data from the same node on both the 'All detectors' subplots and the
        # individual bank figures. To avoid this, we have to create separate dim
        # sliders for the 'All detectors' view. In turn, we need separate nodes in the
        # node tree below.
        self.all_dim_sliders = [
            DimSlider(sizes=self.max_sizes, dim=dim) for dim in other_dims
        ]
        self.all_collect_node = pp.Node(
            lambda *args: args,
            *[pp.widget_node(slider) for slider in self.all_dim_sliders],
        )
        self.all_slice_data_nodes = {
            key: pp.Node(
                _slice_data,
                da=self.apply_masks_nodes[key],
                slices=self.all_collect_node,
            )
            for key in self.data.keys()
        }
        self.all_to_plot_nodes = {
            key: pp.Node(
                lambda da: da.transpose(self._visible_dims),
                da=self.all_slice_data_nodes[key],
            )
            for key in self.data.keys()
        }

        nkeys = len(self.data)
        ncols = min(3, nkeys)
        nrows = math.ceil(len(self.data) / ncols)
        with plt.ioff():
            self.main_figure = plt.figure(layout="constrained", figsize=figsize)
        gs = GridSpec(
            ncols=ncols + 1,
            nrows=nrows,
            figure=self.main_figure,
            width_ratios=[1.0] * ncols + [0.06],
        )
        axs = np.array(
            [
                self.main_figure.add_subplot(gs[i // ncols, i % ncols])
                for i in range(nrows * ncols)
            ]
        )
        cax = self.main_figure.add_subplot(gs[:, -1])

        self.subplots = {}
        self.tab_figs = {}
        _kwargs = {**{"hide_log_buttons": True, "autoscale": False}, **kwargs}
        for i, bank in enumerate(self.data.keys()):
            self.subplots[bank] = pp.imagefigure(
                self.all_to_plot_nodes[bank],
                ax=axs[i],
                title=bank,
                cbar=i == nkeys - 1,
                cax=cax,
                **_kwargs,
            )
            if i % ncols != 0:
                axs[i].set_ylabel('')
            if i < (ncols * (nrows - 1)):
                axs[i].set_xlabel('')
            axs[i]._bank_explorer_subplot_index = i
            self.tab_figs[bank] = pp.imagefigure(
                self.to_plot_nodes[bank],
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
        children = [
            ipw.VBox([_box_of_sliders(self.bank_dim_sliders[key]), f])
            for key, f in zip(self.data.keys(), self.tab_figs.values(), strict=True)
        ]
        if len(self.data) > 1:
            children = [
                ipw.VBox(
                    [_box_of_sliders(self.all_dim_sliders), self.main_figure.canvas]
                ),
                *children,
            ]

        self.tabs.children = children
        titles = list(self.data.keys())
        if len(self.data) > 1:
            titles = ["All", *titles]
        self.tabs.titles = titles
        self.tabs.observe(self.selected_tab_changed, names='selected_index')

        if masks is not None:
            self.mask_control = MaskControl(mask_dims=mask_dims)
            self.mask_control.on_clear(self.clear_masks)
            self.mask_control.on_save(self.save_masks)
            for f in self.tab_figs.values():
                f.fig.canvas.mpl_connect('button_press_event', self.toggle_mask)
        else:
            self.mask_control = None

        self.color_control = ColorControl()
        self.color_control.on_limit_change(self.update_color_range)
        self.color_control.on_log_toggle(self.toggle_log)
        self.color_control.on_fit_click(self.fit_cmap_range)
        vmin, vmax = _find_limits(self.tab_figs.values())
        self.color_control.vmin = vmin
        self.color_control.vmax = vmax

        top_controls = [self.mask_control or ipw.Label(value=""), self.color_control]

        super().__init__(
            [
                ipw.HBox(
                    top_controls,
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

    def update_slider_inds(self, old_tab: int, new_tab: int) -> None:
        old_sliders, new_sliders = self._get_old_and_new_sliders(old_tab, new_tab)
        old_slider_sliders = {sl.dim: sl.slider for sl in old_sliders}
        for sl in new_sliders:
            new_value = min(old_slider_sliders[sl.dim].value, sl.slider.max)
            if (
                (not sl.slider.disabled)
                and (not old_slider_sliders[sl.dim].disabled)
                and (sl.slider.value != new_value)
            ):
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
        self.sync_color_range()
        self.update_colormapper_norm()
        _update_view(self.current_figs)

        new_tab, old_tab = change["new"], change["old"]
        if 0 in (old_tab, new_tab):
            self.update_slider_inds(old_tab, new_tab)
        if self.mask_control is not None:
            self.mask_control.clear_button.disabled = True if new_tab == 0 else False

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
        self.sync_color_range()
        self.update_colormapper_norm()

    def update_color_range(self, _) -> None:
        self.sync_color_range()
        _update_view(self.current_figs)

    def sync_color_range(self) -> None:
        """
        Update the color range (vmin and vmax) for all current figures based on the
        values in the color control widgets.
        """
        bounds = {"cmin": self.color_control.vmin, "cmax": self.color_control.vmax}
        if self.color_control.log_button.value:
            for key, bound in bounds.items():
                if (bound is not None) and (bound <= 0):
                    bounds[key] = None

        figs = self.current_figs
        for f in figs:
            for bound, new in bounds.items():
                if new is None:
                    getattr(f.view.colormapper, f"_{bound}").pop('user', None)
                else:
                    setattr(f.view.colormapper, bound, new)

    def fit_cmap_range(self, _=None) -> None:
        """
        Fit the color map range (vmin and vmax) to the data for all current figures.
        """
        figs = self.current_figs
        vmin, vmax = _find_limits(figs)
        current_vmax = self.color_control.vmax
        if current_vmax is None:
            current_vmax = float('inf')
        # If the new vmin is greater than the current vmax, we have to update the max
        # bound first.
        if vmin > current_vmax:
            self.color_control.vmax = vmax
            self.color_control.vmin = vmin
        else:
            self.color_control.vmin = vmin
            self.color_control.vmax = vmax
        _update_view(figs)

    def toggle_mask(self, event: MouseEvent) -> None:
        """
        Toggle a mask at the current mouse position based on the selected masking mode.
        """
        # Ignore the event if no masking mode is selected
        if not self.mask_control.selected:
            return
        # Ignore the event if it occurred outside any axes
        if event.inaxes is None:
            return
        # Ignore the event if the pan/zoom tool is active
        if event.inaxes.get_navigate_mode() is not None:
            return
        # Ignore the event if it occurred outside the current tab's axes (e.g. colorbar)
        if event.inaxes is not self.tab_figs[self.current_key].ax:
            return

        key = self.current_key
        ydim, xdim = self._visible_dims

        # Find the closest coordinate index to the x, y event coords
        xcoord = self.data[key].coords[xdim]
        if self.data[key].coords.is_edges(xdim):
            xcoord = sc.midpoints(xcoord, dim=xdim)
        ycoord = self.data[key].coords[ydim]
        if self.data[key].coords.is_edges(ydim):
            ycoord = sc.midpoints(ycoord, dim=ydim)
        ix = _find_closest(xcoord.values, event.xdata)
        iy = _find_closest(ycoord.values, event.ydata)

        # Check if click touched any visible masks
        mask_removed = False

        sliced_masks = {}

        for name, mask in self.masks[key].items():
            slices = []
            mask_dims = self.mask_control.mask_dims[name]
            if xdim in mask_dims:
                slices.append((xdim, ix))
            if ydim in mask_dims:
                slices.append((ydim, iy))
            slices.extend(
                [sl.value for sl in self.bank_dim_sliders[key] if sl.dim in mask_dims]
            )

            m = _slice_mask(mask, slices)
            if m.values.any():
                mask_removed = True
                m.values[...] = False

            sliced_masks[name] = m

        if not mask_removed:
            mask_name = self.mask_control.selected
            m = sliced_masks[mask_name]
            m.values[...] = False if m.values.all() else True

        self.apply_masks_nodes[key].notify_children(message=None)

    def clear_masks(self, _) -> None:
        """
        Clear all currently visible masks for the current tab/bank.
        """
        key = self.current_key
        slices = [sl.value for sl in self.bank_dim_sliders[key]]

        for mask in self.masks[key].values():
            m = _slice_mask(mask, slices)
            m.values[...] = False

        self.apply_masks_nodes[key].notify_children(message=None)

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

        # When we save, compare with original masks and remove any masks that are
        # all false that were not in the originals. This is needed because for logic
        # simplicity, we created masks for all detector banks to begin with, but they
        # shouldn't appear in the output if they were never modified.
        to_file = {}
        for key, masks in self.masks.items():
            for name, mask in masks.items():
                if not mask.values.any() and (
                    (key not in self.original_masks)
                    or (name not in self.original_masks[key])
                ):
                    continue
                if key not in to_file:
                    to_file[key] = {}
                to_file[key][name] = mask

        write_pixel_masks(filename, to_file, instrument="loki")
