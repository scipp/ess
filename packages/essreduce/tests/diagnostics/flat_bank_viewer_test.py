# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import math

import plopp as pp
import pytest
import scipp as sc

from ess.reduce.diagnostics.flat_bank_viewer import (
    FlatBankViewer,
    _DimensionConfig,
    _DimensionConfigs,
    _DimensionSettings,
    _unwrap_flat_indices_2d,
)

DIMS = ("arc", "channel", "tube", "length")
SHAPE = (2, 3, 4, 5)


def make_data(dims: tuple[str, ...], shape: tuple[int, ...]) -> sc.DataArray:
    return sc.DataArray(
        sc.arange("x", float(math.prod(shape)), unit="counts").fold(
            "x", dims=dims, shape=shape
        ),
        coords={
            dim: sc.arange(dim, size, unit=None)
            for dim, size in zip(dims, shape, strict=True)
        },
    )


@pytest.fixture
def da() -> sc.DataArray:
    return make_data(DIMS, SHAPE)


@pytest.fixture
def interactive_backend():
    # Without an interactive backend, Plopp creates figures that are not widgets.
    pytest.importorskip("ipympl")
    import matplotlib.pyplot as plt

    original = plt.get_backend()
    plt.switch_backend("module://ipympl.backend_nbagg")
    yield
    plt.switch_backend(original)


def view_test(func):
    """Mark a test that creates a figure."""
    func = pytest.mark.usefixtures("interactive_backend")(func)
    # Emitted by ipympl's toolbar, not by the code under test.
    return pytest.mark.filterwarnings(
        "ignore:Passing unrecognized arguments to super\\(Toolbar\\):DeprecationWarning"
    )(func)


def flat(axis: str, primary: bool = False) -> _DimensionSettings:
    return _DimensionSettings(mode="flatten", axis=axis, primary=primary)


SLICE = _DimensionSettings(mode="slice", axis="vertical", primary=False)


def set_mode(configs: _DimensionConfigs, dim: str, mode: str) -> None:
    # Simulates a click on the mode buttons.
    configs[dim]._mode_button.value = mode


def summarize(configs: _DimensionConfigs) -> dict[str, str | tuple[str, bool]]:
    return {
        dim: "slice" if s.mode == "slice" else (s.axis, s.primary)
        for dim, s in configs.settings.items()
    }


def assert_valid_axes(configs: _DimensionConfigs) -> None:
    flattened = [s for s in configs.settings.values() if s.mode == "flatten"]
    for axis in ("vertical", "horizontal"):
        on_axis = [s for s in flattened if s.axis == axis]
        assert on_axis
        assert sum(s.primary for s in on_axis) == 1


def record_events(configs: _DimensionConfigs) -> dict[str, list[str]]:
    events = {"slider": [], "config": []}
    configs.on_slider_change(lambda config, _: events["slider"].append(config.dim))
    configs.on_config_change(lambda config, _: events["config"].append(config.dim))
    return events


def test_dimension_configs_initial_settings(da: sc.DataArray) -> None:
    settings = {
        "tube": SLICE,
        "length": flat("vertical", primary=True),
        "arc": flat("horizontal", primary=True),
    }
    configs = _DimensionConfigs(da, settings=settings)
    assert summarize(configs) == {
        "arc": ("horizontal", True),
        "channel": ("vertical", False),
        "tube": "slice",
        "length": ("vertical", True),
    }
    assert configs["tube"]._slicer.layout.display == "flex"
    assert configs["tube"]._axis.layout.display == "none"
    assert configs["arc"]._slicer.layout.display == "none"


@pytest.mark.parametrize(
    ("mode", "displays"),
    [("slice", ("flex", "none", "none")), ("flatten", ("none", "flex", "flex"))],
)
def test_switching_mode_updates_visible_controls(
    da: sc.DataArray,
    mode: str,
    displays: tuple[str, ...],
) -> None:
    config = _DimensionConfig(
        "tube",
        size=da.sizes["tube"],
        coord=da.coords["tube"],
        settings=SLICE if mode == "flatten" else flat("vertical"),
    )
    config._mode_button.value = mode
    assert (
        tuple(
            widget.layout.display
            for widget in (config._slicer, config._axis, config._primary)
        )
        == displays
    )


@pytest.mark.parametrize("axis", ["vertical", "horizontal"])
def test_dimension_configs_initial_settings_get_fixed_to_satisfy_constraints(
    da: sc.DataArray,
    axis: str,
) -> None:
    settings = {dim: flat(axis, primary=True) for dim in da.dims}
    configs = _DimensionConfigs(da, settings=settings)
    assert_valid_axes(configs)


def test_moving_slider_emits_only_slider_event(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da)
    set_mode(configs, "tube", "slice")
    events = record_events(configs)
    configs["tube"]._slicer.slider.value = (1, 2)
    assert events == {"slider": ["tube"], "config": []}


@pytest.mark.parametrize("change", ["axis", "mode"])
def test_moving_last_dim_off_an_axis_moves_another_dim_there(
    da: sc.DataArray,
    change: str,
) -> None:
    configs = _DimensionConfigs(da)
    (horizontal,) = [
        dim for dim, s in configs.settings.items() if s.axis == "horizontal"
    ]
    events = record_events(configs)

    if change == "axis":
        configs[horizontal].axis = "vertical"
        assert configs[horizontal].axis == "vertical"  # The user's choice is kept.
    else:
        set_mode(configs, horizontal, "slice")
        assert configs[horizontal].mode == "slice"
    assert_valid_axes(configs)
    assert events == {"slider": [], "config": [horizontal]}  # One event per action.


def test_switching_dim_to_flatten_mode_fills_empty_axis(da: sc.DataArray) -> None:
    settings = {
        "arc": flat("vertical", primary=True),
        "channel": SLICE,
        "tube": SLICE,
        "length": SLICE,
    }
    configs = _DimensionConfigs(da, settings=settings)
    set_mode(configs, "tube", "flatten")
    # Moves the dim that was just switched, not the existing one.
    assert summarize(configs)["arc"] == ("vertical", True)
    assert summarize(configs)["tube"] == ("horizontal", True)


@pytest.mark.parametrize(
    ("dim", "checked", "primary", "config_events"),
    [
        pytest.param("channel", True, "channel", ["channel"], id="replace-primary"),
        pytest.param("arc", False, "arc", [], id="revert-unchecking-only-primary"),
    ],
)
def test_primary_changes_preserve_one_primary_and_emit_only_effective_changes(
    da: sc.DataArray,
    dim: str,
    checked: bool,
    primary: str,
    config_events: list[str],
) -> None:
    settings = {
        "arc": flat("vertical", primary=True),
        "channel": flat("vertical"),
        "tube": flat("vertical"),
        "length": flat("horizontal", primary=True),
    }
    configs = _DimensionConfigs(da, settings=settings)
    events = record_events(configs)
    configs[dim].primary = checked
    assert summarize(configs) == {
        "arc": ("vertical", primary == "arc"),
        "channel": ("vertical", primary == "channel"),
        "tube": ("vertical", False),
        "length": ("horizontal", True),
    }
    assert_valid_axes(configs)
    assert events == {"slider": [], "config": config_events}


def test_moving_primary_dim_to_other_axis_keeps_existing_primary(
    da: sc.DataArray,
) -> None:
    settings = {
        "arc": flat("vertical", primary=True),
        "channel": flat("vertical"),
        "tube": flat("horizontal"),
        "length": flat("horizontal", primary=True),
    }
    configs = _DimensionConfigs(da, settings=settings)
    configs["arc"].axis = "horizontal"
    assert summarize(configs) == {
        "arc": ("horizontal", False),
        "channel": ("vertical", True),
        "tube": ("horizontal", False),
        "length": ("horizontal", True),
    }


def test_constraints_are_not_enforced_with_fewer_than_two_flattened_dims(
    da: sc.DataArray,
) -> None:
    configs = _DimensionConfigs(da)
    for dim in ("channel", "tube", "length"):
        set_mode(configs, dim, "slice")
    events = record_events(configs)
    configs["arc"].axis = "horizontal"
    assert configs["arc"].axis == "horizontal"
    assert events["config"] == ["arc"]


def test_graph_slider_move_only_reruns_slice_branch(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da)
    calls = []

    @pp.node
    def use_settings(settings):
        calls.append("settings")
        return settings

    @pp.node
    def use_slices(settings, slices):
        calls.append(("slices", slices))
        return slices

    sink = pp.Node(
        lambda _: calls.append("sink"),
        use_slices(use_settings(configs.settings_node), configs.slices_node),
    )
    sink()
    assert calls == ["settings", ("slices", {}), "sink"]

    calls.clear()
    set_mode(configs, "tube", "slice")
    # Each node runs exactly once and sees the new slice.
    assert calls == [
        "settings",
        ("slices", {"tube": configs["tube"].slider_value}),
        "sink",
    ]

    calls.clear()
    configs["tube"]._slicer.slider.value = (2, 2)
    assert calls == [("slices", {"tube": (2, 2)}), "sink"]

    calls.clear()
    set_mode(configs, "tube", "flatten")
    assert calls == ["settings", ("slices", {}), "sink"]


@view_test
def test_view_shows_flattened_data(da: sc.DataArray) -> None:
    view = FlatBankViewer(
        da, horizontal=["length"], vertical=["channel", "arc", "tube"]
    )
    assert len(view.children) == 2  # No bank selector.
    assert view.node().dims == ("channel * arc * tube", "length")
    assert view._plot_data.sizes == {"vertical": 24, "horizontal": 5}
    assert view._figure.canvas.ylabel == "channel * arc * tube"
    assert view._figure.canvas.xlabel == "length"
    expected = view.node().data.rename_dims(
        {"channel * arc * tube": "vertical", "length": "horizontal"}
    )
    assert sc.identical(view._plot_data.data, expected)


@view_test
def test_view_updates_figure_in_place(da: sc.DataArray) -> None:
    view = FlatBankViewer(da)
    figure = view._figure

    set_mode(view.configs, "tube", "slice")
    assert view._figure is figure
    assert "tube" not in view._figure.canvas.ylabel
    assert "tube" not in view._figure.canvas.xlabel
    assert math.prod(view._plot_data.shape) == 2 * 3 * 5

    view.configs["tube"]._slicer.slider.value = (3, 3)
    assert view._figure is figure
    assert sc.identical(
        view._plot_data.data.sum(),
        da["tube", 3].data.sum(),
    )


@view_test
def test_view_shows_message_with_fewer_than_two_flattened_dims(
    da: sc.DataArray,
) -> None:
    view = FlatBankViewer(da)
    for dim in ("channel", "tube", "length"):
        set_mode(view.configs, dim, "slice")
    assert view.node() is None
    assert view._message.layout.display is None
    assert view._figure.layout.display == "none"

    set_mode(view.configs, "length", "flatten")
    assert view._message.layout.display == "none"
    assert view._figure.layout.display is None


@view_test
def test_view_hover_shows_indices_of_all_flattened_dims(da: sc.DataArray) -> None:
    view = FlatBankViewer(
        da, sliced=["tube"], horizontal=["length"], vertical=["channel", "arc"]
    )
    view.configs["tube"]._slicer.slider.value = (1, 1)
    # x=4 -> length 4; y=5 -> channel 2, arc 1
    text = view._format_coord(4.2, 4.8)
    assert text.startswith("{Length: 4, Channel: 2, Arc: 1}: ")
    expected = da["arc", 1]["channel", 2]["tube", 1]["length", 4].value
    assert str(expected) in text


@view_test
@pytest.mark.parametrize(("x", "y"), [(-1, 0), (5, 0), (0, -1), (0, 6)])
def test_view_hover_outside_data_is_empty(da: sc.DataArray, x: float, y: float) -> None:
    view = FlatBankViewer(
        da, sliced=["tube"], horizontal=["length"], vertical=["channel", "arc"]
    )
    assert view._format_coord(x, y) == ""


@pytest.fixture
def banks(da: sc.DataArray) -> sc.DataGroup:
    return sc.DataGroup(
        {
            "a": da,
            "b": make_data(("wire", "strip"), (6, 7)),
            "c": make_data(("arc", "pixel", "tube"), (2, 3, 4)),
        }
    )


def bank_selector(view: FlatBankViewer):
    return view.children[0].children[1]


@view_test
def test_view_with_data_group_switches_banks(banks: sc.DataGroup) -> None:
    view = FlatBankViewer(banks)
    selector = bank_selector(view)
    assert list(selector.options) == ["a", "b", "c"]
    figure = view._figure

    selector.value = "b"
    assert view._configs_box.children == (view.configs,)
    assert view._figure is figure
    assert math.prod(view._plot_data.shape) == 6 * 7
    assert {view._figure.canvas.xlabel, view._figure.canvas.ylabel} == {
        "wire",
        "strip",
    }


@view_test
def test_view_with_data_group_keeps_settings_per_bank(banks: sc.DataGroup) -> None:
    view = FlatBankViewer(banks)
    selector = bank_selector(view)
    set_mode(view.configs, "tube", "slice")
    view.configs["tube"]._slicer.slider.value = (2, 2)
    expected = view._plot_data

    selector.value = "c"
    assert view.configs["tube"].mode == "flatten"
    selector.value = "a"
    assert view.configs["tube"].mode == "slice"
    assert sc.identical(view._plot_data, expected)


@view_test
def test_view_ignores_updates_from_other_banks(banks: sc.DataGroup) -> None:
    view = FlatBankViewer(banks)
    selector = bank_selector(view)
    selector.value = "b"
    configs_b = view.configs
    selector.value = "a"
    shown = view._plot_data
    set_mode(configs_b, "wire", "slice")
    assert view._plot_data is shown


@view_test
def test_view_initial_layout_per_bank(banks: sc.DataGroup) -> None:
    view = FlatBankViewer(
        banks,
        sliced=["tube"],
        horizontal=["length", "strip"],
        vertical=["channel", "arc", "wire"],
    )
    assert summarize(view.configs) == {
        "arc": ("vertical", False),
        "channel": ("vertical", True),
        "tube": "slice",
        "length": ("horizontal", True),
    }
    assert view._figure.canvas.ylabel == "channel * arc"
    assert view._figure.canvas.xlabel == "length"

    bank_selector(view).value = "b"
    # The first listed dim that is in the bank is the primary.
    assert summarize(view.configs) == {
        "wire": ("vertical", True),
        "strip": ("horizontal", True),
    }

    bank_selector(view).value = "c"
    # 'pixel' is not listed and no listed horizontal dim is in the bank.
    assert summarize(view.configs) == {
        "arc": ("vertical", True),
        "pixel": ("horizontal", True),
        "tube": "slice",
    }


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ({"sliced": ["arc"], "vertical": ["arc"]}, "more than once"),
        ({"sliced": ["arc"], "horizontal": ["arc"]}, "more than once"),
        ({"horizontal": ["arc"], "vertical": ["arc"]}, "more than once"),
        ({"vertical": ["arc", "arc"]}, "more than once"),
        ({"horizontal": ["nope"]}, "not in the data"),
    ],
)
def test_view_rejects_invalid_layout(
    banks: sc.DataGroup,
    layout: dict[str, list[str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FlatBankViewer(banks, **layout)


def test_view_rejects_empty_bank_mapping() -> None:
    with pytest.raises(ValueError, match="at least one data array"):
        FlatBankViewer({})


def test_view_rejects_non_data_array_entries(banks: sc.DataGroup) -> None:
    banks["scalar"] = sc.scalar(1.0)
    with pytest.raises(TypeError, match="scalar"):
        FlatBankViewer(banks)


def test_unwrap_flat_indices_2d_single_dim() -> None:
    unwrap = _unwrap_flat_indices_2d({'a': 3}, {'b': 5})
    assert unwrap(0, 0) == {'a': 0, 'b': 0}
    assert unwrap(1, 0) == {'a': 1, 'b': 0}
    assert unwrap(2, 0) == {'a': 2, 'b': 0}
    assert unwrap(0, 4) == {'a': 0, 'b': 4}
    assert unwrap(2, 4) == {'a': 2, 'b': 4}


def test_unwrap_flat_indices_2d_two_x_dims() -> None:
    unwrap = _unwrap_flat_indices_2d({'a': 3, '2nd': 5}, {'b': 4})
    assert unwrap(0, 0) == {'a': 0, '2nd': 0, 'b': 0}
    assert unwrap(0, 1) == {'a': 0, '2nd': 0, 'b': 1}
    assert unwrap(0, 3) == {'a': 0, '2nd': 0, 'b': 3}

    assert unwrap(1, 0) == {'a': 0, '2nd': 1, 'b': 0}
    assert unwrap(1, 1) == {'a': 0, '2nd': 1, 'b': 1}
    assert unwrap(1, 3) == {'a': 0, '2nd': 1, 'b': 3}

    assert unwrap(4, 0) == {'a': 0, '2nd': 4, 'b': 0}
    assert unwrap(4, 1) == {'a': 0, '2nd': 4, 'b': 1}
    assert unwrap(4, 3) == {'a': 0, '2nd': 4, 'b': 3}

    assert unwrap(5, 0) == {'a': 1, '2nd': 0, 'b': 0}
    assert unwrap(5, 2) == {'a': 1, '2nd': 0, 'b': 2}
    assert unwrap(5, 3) == {'a': 1, '2nd': 0, 'b': 3}

    assert unwrap(11, 0) == {'a': 2, '2nd': 1, 'b': 0}
    assert unwrap(11, 2) == {'a': 2, '2nd': 1, 'b': 2}
    assert unwrap(11, 3) == {'a': 2, '2nd': 1, 'b': 3}

    assert unwrap(14, 0) == {'a': 2, '2nd': 4, 'b': 0}
    assert unwrap(14, 1) == {'a': 2, '2nd': 4, 'b': 1}
    assert unwrap(14, 2) == {'a': 2, '2nd': 4, 'b': 2}


def test_unwrap_flat_indices_2d_three_y_dims() -> None:
    unwrap = _unwrap_flat_indices_2d({'a': 2}, {'b': 4, 'c': 1, 'd': 3})
    assert unwrap(0, 0) == {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    assert unwrap(1, 0) == {'a': 1, 'b': 0, 'c': 0, 'd': 0}

    assert unwrap(0, 3) == {'a': 0, 'b': 1, 'c': 0, 'd': 0}
    assert unwrap(1, 3) == {'a': 1, 'b': 1, 'c': 0, 'd': 0}

    assert unwrap(0, 7) == {'a': 0, 'b': 2, 'c': 0, 'd': 1}
    assert unwrap(1, 7) == {'a': 1, 'b': 2, 'c': 0, 'd': 1}


def test_unwrap_flat_indices_2d_dim_order() -> None:
    unwrap = _unwrap_flat_indices_2d({'x': 3, 'a': 5, 'g': 1}, {'b': 4, 'y': 4})
    assert list(unwrap(0, 0)) == ['x', 'a', 'g', 'b', 'y']
