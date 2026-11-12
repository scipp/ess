# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Viewer for high-dimensional data."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import ipywidgets as ipw
import plopp as pp
import scipp as sc
from plopp.core.utils import scalar_to_string
from plopp.widgets.slicing import CombinedSlicer


class FlatBankViewer(ipw.VBox):
    """Image plot of data with controls for flattening and slicing its dims."""

    def __init__(
        self,
        data: Mapping[str, sc.DataArray] | sc.DataArray,
        dims: Iterable[str] | None = None,
        *,
        sliced: Iterable[str] = (),
        horizontal: Iterable[str] = (),
        vertical: Iterable[str] = (),
        cbar: bool = True,
        rasterized: bool = True,
        **fig_kwargs: Any,
    ) -> None:
        """Create a new view.

        Parameters
        ----------
        data:
            Data to show. A mapping (e.g., a data group) with one entry per bank
            adds buttons for selecting a bank. The banks may have different dims
            but must all have the same unit.
        dims:
            Order of the dimension controls. With multiple banks, dims that are not
            in a bank are ignored for that bank.
        sliced:
            Dims that start in slice mode.
        horizontal:
            Dims that start flattened onto the horizontal axis.
            The first one that is in a bank is the primary dim of that bank.
        vertical:
            Dims that start flattened onto the vertical axis.
            The first one that is in a bank is the primary dim of that bank.
        cbar:
            Whether to show a colorbar.
        rasterized:
            If ``True``, the figure is rasterized which improves rendering
            speed but reduces resolution.
        **fig_kwargs:
            Additional arguments passed to :func:`plopp.imagefigure`.

        Dims that are not in a bank are ignored for that bank.
        Dims that are in none of ``sliced``, ``horizontal``, and ``vertical`` use
        default settings. Settings may get adjusted to ensure that both axes have
        a dim, as long as at least two dims are flattened.
        """
        self._fig_kwargs = {"cbar": cbar, "rasterized": rasterized} | fig_kwargs
        self._figure = None
        self._figure_key = None
        self._message = ipw.HTML(
            "Put at least two dims into flatten mode to show a plot.",
            layout={"display": "none"},
        )
        self._figure_box = ipw.Box([self._message])
        self._configs_box = ipw.VBox()
        self._plot_data = None
        self._unwrap_indices = None

        self._dims = None if dims is None else tuple(dims)
        if isinstance(data, sc.DataArray):
            self._banks = {"": data}
            selector = None
        else:
            self._banks = _validate_banks(data)
            selector = ipw.ToggleButtons(options=list(self._banks))
            selector.observe(lambda change: self._select(change["new"]), names="value")
        self._layout = _validate_layout(
            self._banks,
            sliced=tuple(sliced),
            horizontal=tuple(horizontal),
            vertical=tuple(vertical),
        )
        # Graphs and controls are created on first selection of a bank and kept
        # afterwards, so each bank keeps its settings.
        self._views: dict[str, _BankView] = {}
        self._current = next(iter(self._banks))
        self._select(self._current)

        children = [self._figure_box, self._configs_box]
        if selector is not None:
            children.insert(0, ipw.HBox([ipw.Label("Bank:"), selector]))
        super().__init__(children)

    @property
    def configs(self) -> "_DimensionConfigs":
        """Dimension controls of the current bank."""
        return self._views[self._current].configs

    @property
    def node(self) -> pp.Node:
        """Node that produces the flattened and sliced data of the current bank."""
        return self._views[self._current].node

    def _select(self, bank: str) -> None:
        self._current = bank
        if bank not in self._views:
            da = self._banks[bank]
            dims = self._dims
            if dims is not None:
                dims = [dim for dim in dims if dim in da.dims]
            self._views[bank] = _BankView(
                da,
                dims,
                _initial_settings(da.dims, **self._layout),
                partial(self._on_bank_data, bank),
            )
        view = self._views[bank]
        self._configs_box.children = [view.configs]
        # The sink does not cache (`_on_bank_data` returns None), so this calls
        # `_show` with the cached data of `view.node`, or computes it for new banks.
        view.sink()

    def _on_bank_data(self, bank: str, data: sc.DataArray | None) -> None:
        # Only the current bank's controls are visible, but be defensive anyway.
        if bank == self._current:
            self._show(data)

    def _show(self, data: sc.DataArray | None) -> None:
        if data is None:
            self._message.layout.display = None
            if self._figure is not None:
                self._figure.layout.display = "none"
            return
        self._message.layout.display = "none"

        vertical_dim, horizontal_dim = data.dims
        data = _to_plot_data(data)
        self._plot_data = data
        # Same order as used by `flatten`.
        sizes = self._views[self._current].sizes
        _, vertical, horizontal = _sort_dims(sizes, self.configs.settings)
        self._unwrap_indices = _unwrap_flat_indices_2d(
            {dim: sizes[dim] for dim in horizontal},
            {dim: sizes[dim] for dim in vertical},
        )

        # Replacing the figure widget makes the frontend re-create the canvas, which
        # flickers and shifts the layout. So use a single figure and modify that.
        if self._figure is None:
            input_node = pp.Node(data)
            self._figure = pp.imagefigure(input_node, **self._fig_kwargs)
            self._figure_key = input_node.id
            self._figure_box.children = [self._message, self._figure]
            # Must come after the figure's first update, which sets Plopp's formatter.
            self._figure.ax.format_coord = self._format_coord
        else:
            self._figure.layout.display = None
            artist = self._figure.artists[self._figure_key]
            if artist.data.shape != data.shape:
                # Plopp's image artists cannot change their shape; replace the artist.
                self._figure.artists.pop(self._figure_key).remove()
                # Plopp does not unregister this when removing an artist.
                self._figure.canvas._coord_formatters.remove(artist.format_coord)

        self._figure.canvas.xlabel = horizontal_dim
        self._figure.canvas.ylabel = vertical_dim
        # Re-creates the artist if it was removed above and redraws the canvas.
        self._figure.update({self._figure_key: data})

    def _format_coord(self, x: float, y: float) -> str:
        """Show the indices of all flattened dims and the value under the cursor."""
        # Use round because axis coords are in the middle of bins.
        i, j = round(x), round(y)
        n_vertical, n_horizontal = self._plot_data.shape
        if not (0 <= i < n_horizontal and 0 <= j < n_vertical):
            return ""
        indices = ", ".join(
            f"{dim.capitalize()}: {index}"
            for dim, index in self._unwrap_indices(i, j).items()
        )
        data = self._plot_data.data["vertical", j]["horizontal", i]
        value = scalar_to_string(data)
        return f"{{{indices}}}: {value}"


class _BankView:
    """Controls and graph for a single data array."""

    def __init__(
        self,
        da: sc.DataArray,
        dims: Iterable[str] | None,
        settings: Mapping[str, "_DimensionSettings"],
        on_data: Callable[[sc.DataArray | None], None],
    ) -> None:
        self.sizes = dict(da.sizes)
        self.configs = _DimensionConfigs(da, dims, settings)
        self.node = _apply_slices(
            _flatten(pp.Node(da), self.configs.settings_node), self.configs.slices_node
        )
        # Leaf node: Plopp calls it with new data whenever anything upstream changes.
        self.sink = pp.Node(on_data, self.node)


def _validate_banks(data: Mapping[str, Any]) -> dict[str, sc.DataArray]:
    if not data:
        raise ValueError("Need at least one data array.")
    for key, value in data.items():
        if not isinstance(value, sc.DataArray):
            raise TypeError(
                f"Expected all entries to be data arrays, got {type(value).__name__} "
                f"for '{key}'."
            )
    return dict(data)


def _validate_layout(
    banks: Mapping[str, sc.DataArray],
    *,
    sliced: tuple[str, ...],
    horizontal: tuple[str, ...],
    vertical: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    listed = [*sliced, *horizontal, *vertical]
    if duplicates := sorted(dim for dim, count in Counter(listed).items() if count > 1):
        raise ValueError(
            f"Dims {duplicates} are listed more than once in "
            "`sliced`, `horizontal`, and `vertical`."
        )
    all_dims = {dim for da in banks.values() for dim in da.dims}
    if unknown := [dim for dim in listed if dim not in all_dims]:
        raise ValueError(f"Dims {unknown} are not in the data.")
    return {"sliced": sliced, "horizontal": horizontal, "vertical": vertical}


def _initial_settings(
    dims: Iterable[str],
    *,
    sliced: tuple[str, ...],
    horizontal: tuple[str, ...],
    vertical: tuple[str, ...],
) -> dict[str, "_DimensionSettings"]:
    """Settings for the listed dims that are in ``dims``."""
    dims = set(dims)
    settings = {
        dim: _DimensionSettings(mode="slice", axis="vertical", primary=False)
        for dim in sliced
        if dim in dims
    }
    for axis, axis_dims in (("horizontal", horizontal), ("vertical", vertical)):
        present = [dim for dim in axis_dims if dim in dims]
        for i, dim in enumerate(present):
            settings[dim] = _DimensionSettings(
                mode="flatten", axis=axis, primary=i == 0
            )
    return settings


def _to_plot_data(da: sc.DataArray) -> sc.DataArray:
    """Rename the dims to fixed names and use index coords.

    This way, a single figure can show any layout of flattened dims.
    Also, this returns a new data array which protects the input from
    modifications in `fig.update`.
    """
    vertical_dim, horizontal_dim = da.dims
    renamed = da.rename_dims({vertical_dim: "vertical", horizontal_dim: "horizontal"})
    return renamed.drop_coords(list(renamed.coords)).assign_coords(
        {dim: sc.arange(dim, size, unit=None) for dim, size in renamed.sizes.items()}
    )


# Called with the DimensionConfig that changed and the traitlets change dict.
_DimensionCallback = Callable[["_DimensionConfig", dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class _DimensionSettings:
    mode: Literal["flatten", "slice"]
    axis: Literal["vertical", "horizontal"]
    primary: bool


_DEFAULT_SETTINGS = _DimensionSettings(mode="flatten", axis="vertical", primary=False)


class _DimensionConfig(ipw.HBox):
    def __init__(
        self,
        dim: str,
        size: int,
        coord: sc.Variable,
        settings: _DimensionSettings = _DEFAULT_SETTINGS,
    ) -> None:
        self.dim = dim
        self._slider_change_handlers = ipw.CallbackDispatcher()
        self._config_change_handlers = ipw.CallbackDispatcher()

        self._mode_button = ipw.ToggleButtons(
            options=["flatten", "slice"],
            value=settings.mode,
            style={"button_width": "9ch"},
            layout=ipw.Layout(width="max-content", margin="0 1em 0 0.5em"),
        )
        self._slicer = CombinedSlicer(dim=dim, size=size, coord=coord)
        # DimensionConfig shows its own dim label.
        self._slicer.int_slicer.dim_label.layout.display = "none"
        self._slicer.range_slicer.dim_label.layout.display = "none"
        self._axis = ipw.RadioButtons(
            options=["vertical", "horizontal"],
            value=settings.axis,
            description="axis",
        )
        self._axis.add_class("sc-horizontal-radio")
        self._primary = ipw.Checkbox(
            value=settings.primary, description="Primary", indent=False
        )

        self._mode_button.observe(self._on_mode_switch, names="value")
        # `CombinedSlicer.slider` is always the range slider; in single-handle mode
        # the single slider drives it as `(i, i)`, so this catches both modes as well
        # as edits in the text boxes.
        self._slicer.slider.observe(self._emit_slider_change, names="value")
        self._axis.observe(self._emit_config_change, names="value")
        self._primary.observe(self._emit_config_change, names="value")
        self._update_visibility()

        label = ipw.Label(dim, layout=ipw.Layout(width="10ch"))

        style = ipw.HTML(
            """
<style>
.sc-horizontal-radio .widget-radio-box {
    flex-direction: row !important;
    padding: 0 !important;
    margin: 0 !important;
}
.sc-horizontal-radio .widget-radio-box label {
    margin-right: 10px;
    height: auto !important;
    line-height: normal !important;
}
.sc-horizontal-radio.widget-radio {
    width: unset !important;
}
.sc-horizontal-radio > label {
    display: none;
}
</style>
""",
            layout={"display": "none"},
        )
        super().__init__(
            [
                style,
                label,
                self._mode_button,
                self._slicer,
                self._axis,
                self._primary,
            ],
            layout=ipw.Layout(align_items="center"),
        )

    def _update_visibility(self) -> None:
        slicing = self.mode == "slice"
        self._slicer.layout.display = "flex" if slicing else "none"
        self._axis.layout.display = self._primary.layout.display = (
            "none" if slicing else "flex"
        )

    def _on_mode_switch(self, change: dict[str, Any]) -> None:
        self._update_visibility()
        self._emit_config_change(change)

    @property
    def mode(self) -> Literal["flatten", "slice"]:
        return self._mode_button.value

    @property
    def slider_value(self) -> tuple[int, int]:
        """Inclusive index range ``(start, stop)``; ``(i, i)`` in single-handle mode."""
        return self._slicer.value

    @property
    def axis(self) -> Literal["vertical", "horizontal"]:
        return self._axis.value

    @axis.setter
    def axis(self, value: Literal["vertical", "horizontal"]) -> None:
        self._axis.value = value

    @property
    def primary(self) -> bool:
        return self._primary.value

    @primary.setter
    def primary(self, value: bool) -> None:
        self._primary.value = value

    @property
    def settings(self) -> _DimensionSettings:
        return _DimensionSettings(mode=self.mode, axis=self.axis, primary=self.primary)

    def on_slider_change(
        self, callback: _DimensionCallback, remove: bool = False
    ) -> None:
        """Register a callback for (cheap) slider moves."""
        self._slider_change_handlers.register_callback(callback, remove=remove)

    def on_config_change(
        self, callback: _DimensionCallback, remove: bool = False
    ) -> None:
        """Register a callback for any other (expensive) configuration change."""
        self._config_change_handlers.register_callback(callback, remove=remove)

    def _emit_slider_change(self, change: dict[str, Any]) -> None:
        self._slider_change_handlers(self, change)

    def _emit_config_change(self, change: dict[str, Any]) -> None:
        self._config_change_handlers(self, change)


class _UncachedNode(pp.Node):
    """Input node that re-evaluates ``func`` on every request instead of caching."""

    def request_data(self) -> Any:
        return self.func()


class _DimensionConfigs(ipw.VBox):
    def __init__(
        self,
        da: sc.DataArray,
        dims: Iterable[str] | None = None,
        settings: Mapping[str, _DimensionSettings] | None = None,
    ) -> None:
        """Create controls for the dims of ``da``.

        ``settings`` holds the initial settings; dims without an entry use the
        defaults. These may get adjusted to fulfill the constraints, e.g., to
        ensure both axes have a dim.
        """
        self._slider_change_handlers = ipw.CallbackDispatcher()
        self._config_change_handlers = ipw.CallbackDispatcher()

        dims = da.dims if dims is None else dims
        settings = {} if settings is None else settings
        self._configs = {
            dim: _DimensionConfig(
                dim,
                size=da.sizes[dim],
                coord=_slicer_coord(da, dim),
                settings=settings.get(dim, _DEFAULT_SETTINGS),
            )
            for dim in dims
        }
        for config in self._configs.values():
            config.on_slider_change(self._slider_change_handlers)
            config.on_config_change(self._on_child_config_change)

        # Set while `_enforce_constraints` modifies child widgets, to suppress the
        # events those modifications would otherwise emit.
        self._enforcing = False
        self._enforce_constraints(source=None, widget=None)
        self._last_settings = self._get_settings()

        # Two separate graph inputs so that slider moves only invalidate the
        # nodes downstream of `slices_node`; Plopp's per-node cache keeps the
        # (expensive) results downstream of `settings_node` intact.
        # `slices_node` must not cache: its value depends on the modes, but mode
        # switches only notify `settings_node` (notifying both would make shared
        # children run twice). Its children still get re-run via `settings_node`.
        self.slices_node = _UncachedNode(self._get_slices)
        self.slices_node.pretty_name = "DimensionConfigs <slices>"
        self.settings_node = pp.Node(self._get_settings)
        self.settings_node.pretty_name = "DimensionConfigs <settings>"
        self.on_slider_change(
            lambda _, change: self.slices_node.notify_children(change)
        )
        self.on_config_change(
            lambda _, change: self.settings_node.notify_children(change)
        )

        super().__init__(list(self._configs.values()))

    def __getitem__(self, dim: str) -> _DimensionConfig:
        return self._configs[dim]

    @property
    def settings(self) -> dict[str, _DimensionSettings]:
        return self._get_settings()

    def on_slider_change(
        self, callback: _DimensionCallback, remove: bool = False
    ) -> None:
        """Register a callback for slider moves in any child."""
        self._slider_change_handlers.register_callback(callback, remove=remove)

    def on_config_change(
        self, callback: _DimensionCallback, remove: bool = False
    ) -> None:
        """Register a callback for any non-slider change in any child."""
        self._config_change_handlers.register_callback(callback, remove=remove)

    def _get_slices(self) -> dict[str, tuple[int, int]]:
        return {
            dim: config.slider_value
            for dim, config in self._configs.items()
            if config.mode == "slice"
        }

    def _get_settings(self) -> dict[str, _DimensionSettings]:
        return {dim: config.settings for dim, config in self._configs.items()}

    def _on_child_config_change(
        self, config: _DimensionConfig, change: dict[str, Any]
    ) -> None:
        if self._enforcing:
            # Caused by `_enforce_constraints`; the outer call emits the event.
            return
        self._enforce_constraints(source=config, widget=change["owner"])
        settings = self._get_settings()
        if settings == self._last_settings:
            # E.g., the user unchecked the only primary dim of an axis, and it got
            # re-checked.
            return
        self._last_settings = settings
        self._config_change_handlers(config, change)

    def _enforce_constraints(
        self, source: _DimensionConfig | None, widget: ipw.Widget | None
    ) -> None:
        """Ensure there is at least one dim on each axis and exactly one primary dim
        per axis.

        Only applies when at least two dims are in flatten mode.
        ``source`` and ``widget`` identify the user's change, which takes precedence
        over the state of other dims.
        """
        flat = [c for c in self._configs.values() if c.mode == "flatten"]
        if len(flat) < 2:
            return
        user_axis = source is not None and widget is source._axis
        user_mode = source is not None and widget is source._mode_button
        user_primary = source is not None and widget is source._primary

        self._enforcing = True
        try:
            for axis in ("vertical", "horizontal"):
                if not any(c.axis == axis for c in flat):
                    # Never undo the axis the user just picked. But prefer moving a
                    # dim that was just switched to flatten mode.
                    candidates = [c for c in flat if not (user_axis and c is source)]
                    preferred = source if user_mode else None
                    _choose_dim_to_move(candidates, preferred).axis = axis

            for axis in ("vertical", "horizontal"):
                group = [c for c in flat if c.axis == axis]
                if user_primary and source in group:
                    # Checked: takes over from the previous primary.
                    # Unchecked: gets re-checked, like a radio button.
                    keep = source
                else:
                    # Keep the existing primary over one brought in by `source`,
                    # e.g., by moving a primary dim to this axis.
                    primaries = [c for c in group if c.primary]
                    keep = next(
                        (c for c in primaries if c is not source),
                        (primaries or group)[0],
                    )
                for c in group:
                    c.primary = c is keep
        finally:
            self._enforcing = False


def _choose_dim_to_move(
    candidates: list[_DimensionConfig], preferred: _DimensionConfig | None
) -> _DimensionConfig:
    if preferred in candidates:
        return preferred
    non_primary = [c for c in candidates if not c.primary]
    return (non_primary or candidates)[-1]


def _slicer_coord(da: sc.DataArray, dim: str) -> sc.Variable:
    # Same fallback as Plopp's slice widgets.
    if dim in da.coords:
        return da.coords[dim]
    return sc.arange(dim, da.sizes[dim], unit=None)


@pp.node
def _flatten(
    da: sc.DataArray, settings: Mapping[str, _DimensionSettings]
) -> sc.DataArray | None:
    sliced, vertical, horizontal = _sort_dims(da.dims, settings)
    if not vertical or not horizontal:
        # Fewer than two dims in flatten mode; nothing to plot.
        return None

    return (
        da.transpose([*sliced, *vertical, *horizontal])
        .flatten(horizontal, to=" * ".join(horizontal))
        .flatten(vertical, to=" * ".join(vertical))
    )


def _sort_dims(
    reference_dims: Iterable[str], settings: Mapping[str, _DimensionSettings]
) -> tuple[list[str], list[str], list[str]]:
    sliced = []
    vertical = []
    horizontal = []
    # Preserve order in da for cheaper transpose
    for dim in reference_dims:
        match settings[dim]:
            case _DimensionSettings(mode="slice"):
                sliced.append(dim)
            case _DimensionSettings(axis="vertical", primary=False):
                vertical.append(dim)
            case _DimensionSettings(axis="vertical", primary=True):
                vertical.insert(0, dim)
            case _DimensionSettings(axis="horizontal", primary=False):
                horizontal.append(dim)
            case _DimensionSettings(axis="horizontal", primary=True):
                horizontal.insert(0, dim)
    return sliced, vertical, horizontal


@pp.node
def _apply_slices(
    da: sc.DataArray | None, slices: Mapping[str, tuple[int, int]]
) -> sc.DataArray | None:
    if da is None:
        return None
    for dim, (start, stop) in slices.items():
        da = da[dim, start] if start == stop else da[dim, start : stop + 1].sum(dim)
    return da


def _unwrap_flat_indices_2d(
    x_sizes: dict[str, int], y_sizes: dict[str, int]
) -> Callable[[int, int], dict[str, int]]:
    def unwrap(x: int, y: int) -> dict[str, int]:
        return {**_unwrap_flat_index(x, x_sizes), **_unwrap_flat_index(y, y_sizes)}

    return unwrap


def _unwrap_flat_index(index: int, sizes: dict[str, int]) -> dict[str, int]:
    indices = []
    for dim, size in reversed(sizes.items()):
        index, offset = divmod(index, size)
        indices.append((dim, offset))
    return dict(reversed(indices))  # Restore the input order.
