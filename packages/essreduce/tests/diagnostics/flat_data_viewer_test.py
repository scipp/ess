# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipp as sc
import scipp.testing

from ess.reduce.diagnostics.flat_data_viewer import FlatDataViewer

pytestmark = pytest.mark.filterwarnings(
    r"ignore:Passing unrecognized arguments to super\(Toolbar\):DeprecationWarning"
)


@pytest.fixture
def widget_backend():
    previous_backend = plt.get_backend()
    plt.switch_backend("module://ipympl.backend_nbagg")
    try:
        yield
    finally:
        plt.close("all")
        plt.switch_backend(previous_backend)


def make_data(**sizes: int) -> sc.DataArray:
    shape = tuple(sizes.values())
    return sc.DataArray(
        sc.array(
            dims=list(sizes),
            values=np.arange(np.prod(shape), dtype=float).reshape(shape),
            unit="counts",
        )
    )


def figures_of(viewer: FlatDataViewer) -> tuple[ipw.ToggleButtons | None, ipw.VBox]:
    """Return the bank selector (if any) and the box holding the figure."""
    [box] = viewer.children
    if isinstance(box.children[0], ipw.ToggleButtons):
        selector, figure_box = box.children
        return selector, figure_box
    return None, box


def plotted_data(figure) -> sc.DataArray:
    [node] = figure.graph_nodes.values()
    return node()


@pytest.mark.usefixtures("widget_backend")
def test_single_data_array_has_no_selector() -> None:
    da = make_data(b=3, c=4, d=5)
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["c"])

    selector, figure_box = figures_of(viewer)
    assert selector is None
    [figure] = figure_box.children
    assert plotted_data(figure).sizes == {"c": 4, "b * d": 15}


@pytest.mark.usefixtures("widget_backend")
def test_axis_labels_are_flattened_dims() -> None:
    da = make_data(a=2, b=3, c=4, d=5)
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["a", "c"])

    [figure] = figures_of(viewer)[1].children
    assert figure.ax.get_xlabel() == "b * d"
    assert figure.ax.get_ylabel() == "a * c"


@pytest.mark.usefixtures("widget_backend")
def test_plotted_data_matches_flattened_input() -> None:
    da = make_data(b=3, c=4, d=5)
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["c"])

    [figure] = figures_of(viewer)[1].children
    expected = da.transpose(["c", "b", "d"]).flatten(["b", "d"], to="b * d")
    sc.testing.assert_identical(plotted_data(figure).data, expected.data)


@pytest.mark.usefixtures("widget_backend")
def test_dims_on_neither_axis_are_sliced() -> None:
    da = make_data(a=2, b=3, c=4, d=5)
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["c"])

    [figure] = figures_of(viewer)[1].children
    assert set(plotted_data(figure).dims) == {"c", "b * d"}
    # The slider widget for the sliced dim lives in the figure's bottom bar.
    [slice_widget] = figure.bottom_bar.children
    assert set(slice_widget.value) == {"a"}


@pytest.mark.usefixtures("widget_backend")
def test_format_coord_shows_unwrapped_indices_and_value() -> None:
    da = make_data(b=3, c=4, d=5)
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["c"])

    [figure] = figures_of(viewer)[1].children
    # Flat horizontal index 7 -> b=1, d=2.
    result = figure.ax.format_coord(7.2, 2.1)
    expected_value = da["b", 1]["c", 2]["d", 2].value
    assert result == f"{{b: 1, d: 2, c: 2}}: {expected_value} counts"


@pytest.mark.usefixtures("widget_backend")
def test_format_coord_returns_empty_string_outside_of_data() -> None:
    da = make_data(b=3, c=4, d=5)
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["c"])

    [figure] = figures_of(viewer)[1].children
    assert figure.ax.format_coord(-5.0, 1.0) == ""
    assert figure.ax.format_coord(1.0, 100.0) == ""


@pytest.mark.usefixtures("widget_backend")
def test_format_coord_handles_descending_coord() -> None:
    da = make_data(b=3, c=4, d=5).assign_coords(
        c=sc.array(dims=["c"], values=[40.0, 30.0, 20.0, 10.0], unit="m")
    )
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["c"])

    [figure] = figures_of(viewer)[1].children
    result = figure.ax.format_coord(7.2, 31.0)
    expected_value = da["b", 1]["c", 1]["d", 2].value
    assert result == f"{{b: 1, d: 2, c: 1}}: {expected_value} counts"


@pytest.mark.usefixtures("widget_backend")
def test_no_separators_by_default() -> None:
    da = make_data(b=3, c=4, d=5)
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["c"])

    [figure] = figures_of(viewer)[1].children
    assert len(figure.ax.lines) == 0


@pytest.mark.usefixtures("widget_backend")
def test_separators_are_drawn_between_blocks_of_slowest_dim() -> None:
    da = make_data(b=3, c=4, d=5)
    viewer = FlatDataViewer(da, horizontal=["b", "d"], vertical=["c"], separators="red")

    [figure] = figures_of(viewer)[1].children
    vertical_lines = [
        line.get_xdata()[0]
        for line in figure.ax.lines
        if line.get_xdata()[0] == line.get_xdata()[1]
    ]
    horizontal_lines = [
        line.get_ydata()[0]
        for line in figure.ax.lines
        if line.get_ydata()[0] == line.get_ydata()[1]
    ]
    # b has 3 blocks of size 5 -> 2 interior boundaries.
    np.testing.assert_array_equal(vertical_lines, [4.5, 9.5])
    # c is the only vertical dim -> boundary between every row.
    np.testing.assert_array_equal(horizontal_lines, [0.5, 1.5, 2.5])
    assert all(line.get_color() == "red" for line in figure.ax.lines)


@pytest.mark.usefixtures("widget_backend")
def test_cbar_can_be_disabled() -> None:
    da = make_data(b=3, c=4)
    with_cbar = FlatDataViewer(da, horizontal=["b"], vertical=["c"])
    without_cbar = FlatDataViewer(da, horizontal=["b"], vertical=["c"], cbar=False)

    assert figures_of(with_cbar)[1].children[0].cax is not None
    assert figures_of(without_cbar)[1].children[0].cax is None


@pytest.mark.usefixtures("widget_backend")
def test_fig_kwargs_are_forwarded() -> None:
    da = make_data(b=3, c=4)
    viewer = FlatDataViewer(da, horizontal=["b"], vertical=["c"], title="My title")

    [figure] = figures_of(viewer)[1].children
    assert figure.ax.get_title() == "My title"


@pytest.mark.usefixtures("widget_backend")
def test_mapping_creates_selector_with_bank_names() -> None:
    da = make_data(b=3, c=4, d=5)
    dg = sc.DataGroup({"bank1": da, "bank2": da["d", 0]})
    viewer = FlatDataViewer(dg, horizontal=["b", "d"], vertical=["c"])

    selector, figure_box = figures_of(viewer)
    assert selector is not None
    assert selector.options == ("bank1", "bank2")
    assert selector.value == "bank1"
    [figure] = figure_box.children
    assert plotted_data(figure).sizes == {"c": 4, "b * d": 15}


@pytest.mark.usefixtures("widget_backend")
def test_selecting_bank_shows_its_figure_and_ignores_missing_dims() -> None:
    da = make_data(b=3, c=4, d=5)
    dg = {"bank1": da, "bank2": da["d", 0]}
    viewer = FlatDataViewer(dg, horizontal=["b", "d"], vertical=["c"])
    selector, figure_box = figures_of(viewer)
    [first_figure] = figure_box.children

    selector.value = "bank2"
    [second_figure] = figure_box.children
    assert second_figure is not first_figure
    # 'd' is not in bank2, so only 'b' is on the horizontal axis.
    assert plotted_data(second_figure).sizes == {"c": 4, "b": 3}

    selector.value = "bank1"
    # Figures are cached and reused.
    assert figure_box.children == (first_figure,)


@pytest.mark.usefixtures("widget_backend")
def test_per_bank_dims() -> None:
    dg = {"bank1": make_data(b=3, c=4), "bank2": make_data(x=2, y=6)}
    viewer = FlatDataViewer(
        dg,
        horizontal={"bank1": ["b"], "bank2": ["y"]},
        vertical={"bank1": ["c"], "bank2": ["x"]},
    )
    selector, figure_box = figures_of(viewer)
    assert plotted_data(figure_box.children[0]).sizes == {"c": 4, "b": 3}

    selector.value = "bank2"
    assert plotted_data(figure_box.children[0]).sizes == {"x": 2, "y": 6}


def test_raises_if_single_data_array_with_mapping_dims() -> None:
    da = make_data(b=3, c=4)
    with pytest.raises(TypeError, match="`horizontal` is a mapping"):
        FlatDataViewer(da, horizontal={"": ["b"]}, vertical=["c"])
    with pytest.raises(TypeError, match="`vertical` is a mapping"):
        FlatDataViewer(da, horizontal=["b"], vertical={"": ["c"]})


def test_raises_if_dim_is_listed_twice() -> None:
    da = make_data(b=3, c=4, d=5)
    with pytest.raises(ValueError, match=r"\['b'\]"):
        FlatDataViewer(da, horizontal=["b", "d"], vertical=["b", "c"])
