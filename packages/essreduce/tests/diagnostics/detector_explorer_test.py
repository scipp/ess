# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.reduce.diagnostics.detector_explorer import DetectorExplorer
from ess.reduce.masking import read_pixel_masks


@pytest.fixture(autouse=True)
def widget_backend():
    backend = plt.get_backend()
    plt.switch_backend('module://ipympl.backend_nbagg')
    yield
    plt.close('all')
    plt.switch_backend(backend)


def make_banks(sizes=None, count=3) -> sc.DataGroup:
    if sizes is None:
        sizes = {'layer': 2, 'y': 3, 'x': 4}
    return sc.DataGroup(
        {
            f'bank_{index}': sc.DataArray(
                sc.arange('pixel', math.prod(sizes.values()), unit='counts')
                .to(dtype='float64')
                .fold('pixel', sizes=sizes)
                + sc.scalar(index, unit='counts')
            )
            for index in range(count)
        }
    )


def make_y_masks(banks) -> sc.Variable:
    return {
        key: {
            "m": sc.zeros(
                sizes={dim: bank.sizes[dim] for dim in ('layer', 'y')}, dtype=bool
            )
        }
        for key, bank in banks.items()
    }


def make_x_y_masks(banks) -> sc.Variable:
    return {
        key: {
            "m_y": sc.zeros(
                sizes={dim: bank.sizes[dim] for dim in ('layer', 'y')}, dtype=bool
            ),
            "m_xy": sc.zeros(
                sizes={dim: bank.sizes[dim] for dim in ('layer', 'y', 'x')}, dtype=bool
            ),
        }
        for key, bank in banks.items()
    }


def test_initial_tabs_and_figures_follow_bank_order():
    data = make_banks()
    explorer = DetectorExplorer(data)

    assert explorer.tabs.titles == ('All', *data.keys())
    assert len(explorer.tabs.children) == len(data) + 1
    assert explorer.current_tab == 0
    assert explorer.current_key == 'All'
    assert explorer.current_figs == list(explorer.subplots.values())
    assert list(explorer.subplots) == list(data)
    assert list(explorer.tab_figs) == list(data)
    for bank in data:
        assert explorer.subplots[bank].ax.get_title() == bank
        assert explorer.tab_figs[bank].ax.get_title() == bank


@pytest.mark.parametrize('show_dims', [None, ['y', 'x'], ['x', 'y'], ['layer', 'x']])
def test_initial_display_slices_hidden_dimensions_and_orders_visible_dimensions(
    show_dims,
):
    data = make_banks()
    explorer = DetectorExplorer(data, show_dims=show_dims)
    visible_dims = ['y', 'x'] if show_dims is None else show_dims

    for bank, array in data.items():
        expected = array.assign_coords(
            {dim: sc.arange(dim, size) for dim, size in array.sizes.items()}
        )
        for dim in array.dims:
            if dim not in visible_dims:
                expected = expected[dim, 0]
        expected = expected.transpose(visible_dims)
        assert_identical(explorer.to_plot_nodes[bank](), expected)
        assert_identical(explorer.all_to_plot_nodes[bank](), expected)
        for figure in (explorer.subplots[bank], explorer.tab_figs[bank]):
            assert figure.view._dims == {'x': visible_dims[1], 'y': visible_dims[0]}


def test_missing_coordinates_are_added_without_modifying_input():
    data = make_banks()
    original = data.copy(deep=True)
    explorer = DetectorExplorer(data)

    for bank, array in data.items():
        assert_identical(array, original[bank])
        for dim, size in array.sizes.items():
            assert_identical(explorer.data[bank].coords[dim], sc.arange(dim, size))


def test_slider_bounds_follow_individual_bank_sizes_and_overall_maximum():
    data = make_banks({'time': 2, 'layer': 4, 'y': 3, 'x': 4})
    data['bank_1'] = data['bank_1']['time', :1]['layer', :2]
    data['bank_2'] = data['bank_2']['layer', :3]
    explorer = DetectorExplorer(data)

    for bank, sliders in explorer.bank_dim_sliders.items():
        for slider in sliders:
            assert slider.slider.max == data[bank].sizes[slider.dim] - 1
    for slider in explorer.all_dim_sliders:
        assert (
            slider.slider.max
            == max(array.sizes[slider.dim] for array in data.values()) - 1
        )


@pytest.mark.parametrize('count', [2, 3, 4, 5, 6, 7])
def test_bank_counts_including_incomplete_subplot_rows(count):
    data = make_banks({'y': 3, 'x': 4}, count=count)
    explorer = DetectorExplorer(data)

    assert len(explorer.subplots) == count
    assert len(explorer.tab_figs) == count
    assert explorer.tabs.titles == ('All', *data.keys())
    for bank in data:
        assert explorer.subplots[bank].ax.get_title() == bank


@pytest.mark.parametrize('sizes', [{'y': 3, 'x': 4}, {'layer': 2, 'y': 3, 'x': 4}])
def test_single_bank_has_only_its_bank_tab_and_no_overview_figure(sizes):
    explorer = DetectorExplorer(make_banks(sizes, count=1))

    assert list(explorer.tab_figs) == ['bank_0']
    assert explorer.tabs.titles == ('bank_0',)
    assert len(explorer.tabs.children) == 1
    assert explorer.current_tab == 0
    assert explorer.current_key == 'bank_0'
    assert explorer.current_figs == [explorer.tab_figs['bank_0']]
    assert explorer.tabs.children[0].children[1] is explorer.tab_figs['bank_0']


def test_single_bank_tab_uses_bank_sliders():
    explorer = DetectorExplorer(make_banks(count=1))
    slider = explorer.bank_dim_sliders['bank_0'][0]

    assert explorer.tabs.children[0].children[0].children[0] is slider
    slider.slider.value = 1
    assert_identical(
        explorer.to_plot_nodes['bank_0']().data,
        explorer.data['bank_0']['layer', 1].data,
    )


def test_single_bank_color_controls_update_bank_figure():
    explorer = DetectorExplorer(make_banks(count=1))

    explorer.color_control.vmin = 1
    explorer.color_control.vmax = 10

    colormapper = explorer.tab_figs['bank_0'].view.colormapper
    assert colormapper.cmin == 1
    assert colormapper.cmax == 10


def test_initial_masks_are_empty_and_independent_between_banks():
    data = make_banks()
    explorer = DetectorExplorer(data)

    for bank in data.keys():
        assert not explorer.masks[bank]
    explorer.masks['bank_0']['new'] = sc.ones(
        sizes=explorer.data['bank_0'].sizes, dtype=bool
    )
    for bank in ('bank_1', 'bank_2'):
        assert not explorer.masks[bank]


def test_slider_disabled_if_one_bank_does_not_have_slider_dim():
    data = make_banks()
    data['bank_1'] = data['bank_1'].sum('layer')
    explorer = DetectorExplorer(data)

    assert not explorer.bank_dim_sliders['bank_0'][0].slider.disabled
    assert explorer.bank_dim_sliders['bank_1'][0].slider.disabled
    assert not explorer.bank_dim_sliders['bank_2'][0].slider.disabled

    # Check that no error is raised when interacting with the sliders
    explorer.bank_dim_sliders['bank_0'][0].slider.value += 1
    explorer.bank_dim_sliders['bank_2'][0].slider.value += 1
    explorer.all_dim_sliders[0].slider.value += 1


def test_error_if_not_two_visible_dims():
    data = make_banks()
    with pytest.raises(ValueError, match="Exactly two visible dimensions"):
        DetectorExplorer(data, show_dims=['time'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_uneven_banks():
    data = make_banks({'time': 3, 'layer': 4, 'y': 3, 'x': 4})
    data['bank_1'] = data['bank_1']['layer', :2]
    data['bank_2'] = data['bank_2']['time', :2]
    return data


def select(explorer, name):
    explorer.tabs.selected_index = list(explorer.tabs.titles).index(name)


def sliders_of(explorer, name):
    if name == 'All':
        return {s.dim: s for s in explorer.all_dim_sliders}
    return {s.dim: s for s in explorer.bank_dim_sliders[name]}


def expected_display(explorer, bank, **indices):
    """Reference display: ``dim=None`` sums, ``dim=int`` slices."""
    out = explorer.data[bank].assign_masks(explorer.masks[bank])
    for dim, ind in indices.items():
        if ind is None:
            out = out.sum(dim)
        else:
            out = out[dim, ind]
    return out


def _event(name, canvas, ax, x, y, button=1):
    from matplotlib.backend_bases import MouseEvent

    px, py = ax.transData.transform((x, y)) if ax is not None else (-50, -50)
    return MouseEvent(name, canvas, px, py, button=button)


def fire(canvas, ax, name, x=0, y=0, button=1):
    canvas.callbacks.process(name, _event(name, canvas, ax, x, y, button))


def bank_canvas(explorer, bank):
    figure = explorer.tab_figs[bank]
    return figure.fig.canvas, figure.ax


def press(explorer, bank, x, y):
    canvas, ax = bank_canvas(explorer, bank)
    fire(canvas, ax, 'button_press_event', x, y)


def move(explorer, bank, x, y):
    canvas, ax = bank_canvas(explorer, bank)
    fire(canvas, ax, 'motion_notify_event', x, y)


def release(explorer, bank, x=0, y=0, outside=False):
    canvas, ax = bank_canvas(explorer, bank)
    fire(canvas, None if outside else ax, 'button_release_event', x, y)


def move_outside(explorer, bank):
    canvas, _ = bank_canvas(explorer, bank)
    fire(canvas, None, 'motion_notify_event')


def mask_at(sizes, mode, layer, x, y):
    if mode == 'single':
        expected = sc.zeros(sizes=sizes, dtype=bool)
        expected['layer', layer]['x', x]['y', y].value = True
    elif mode == 'horizontal':
        sizes = sizes.copy()
        del sizes['x']
        expected = sc.zeros(sizes=sizes, dtype=bool)
        expected['layer', layer]['y', y].value = True
    elif mode == 'vertical':
        sizes = sizes.copy()
        del sizes['y']
        expected = sc.zeros(sizes=sizes, dtype=bool)
        expected['layer', layer]['x', x].value = True
    return expected


# ---------------------------------------------------------------------------
# 2. Slicing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('view', ['bank', 'all'])
def test_slider_displays_selected_slice_including_first_and_last(index, view):
    explorer = DetectorExplorer(make_banks())
    sliders = sliders_of(explorer, 'bank_1' if view == 'bank' else 'All')
    nodes = explorer.to_plot_nodes if view == 'bank' else (explorer.all_to_plot_nodes)
    if view == 'all':
        select(explorer, 'All')
    sliders['layer'].slider.value = index

    result = nodes['bank_1']()
    assert_identical(result.data, explorer.data['bank_1']['layer', index].data)
    assert not result.masks


def test_all_index_beyond_smaller_bank_is_blank_and_unmasked():
    explorer = DetectorExplorer(make_uneven_banks())
    sliders = sliders_of(explorer, 'All')
    sliders['layer'].slider.value = 3

    result = explorer.all_to_plot_nodes['bank_1']()
    assert sc.isnan(result.data).values.all()
    assert not result.masks
    expected = explorer.data['bank_0']['time', 0]['layer', 3]
    assert_identical(explorer.all_to_plot_nodes['bank_0']().data, expected.data)

    sliders['layer'].slider.value = 0
    sliders['time'].slider.value = 2
    result = explorer.all_to_plot_nodes['bank_2']()
    assert sc.isnan(result.data).values.all()
    assert not result.masks


def test_slider_updates_preserve_manual_color_limits():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    explorer.color_control.vmin = 2
    explorer.color_control.vmax = 9
    explorer.bank_dim_sliders['bank_1'][0].slider.value = 1

    colormapper = explorer.tab_figs['bank_1'].view.colormapper
    assert (colormapper.cmin, colormapper.cmax) == (2, 9)
    assert explorer.color_control.vmin == 2


# ---------------------------------------------------------------------------
# 3. Tab switching
# ---------------------------------------------------------------------------

TRANSITIONS = [
    ('All', 'bank_1'),
    ('bank_1', 'All'),
    ('bank_0', 'bank_1'),
    ('bank_1', 'bank_0'),
]


def test_indices_transfer_clamped_when_entering_and_leaving_all():
    explorer = DetectorExplorer(make_uneven_banks())
    all_sliders = sliders_of(explorer, 'All')
    all_sliders['layer'].slider.value = 3
    all_sliders['time'].slider.value = 2

    select(explorer, 'bank_1')
    bank = sliders_of(explorer, 'bank_1')
    assert bank['layer'].slider.value == 1
    assert bank['time'].slider.value == 2

    bank['layer'].slider.value = 0
    bank['time'].slider.value = 1
    select(explorer, 'All')
    assert all_sliders['layer'].slider.value == 0
    assert all_sliders['time'].slider.value == 1

    select(explorer, 'bank_2')
    assert sliders_of(explorer, 'bank_2')['time'].slider.value == 1


def test_bank_to_bank_switch_preserves_destination_indices():
    explorer = DetectorExplorer(make_uneven_banks())
    select(explorer, 'bank_1')
    sliders_of(explorer, 'bank_1')['layer'].slider.value = 1
    select(explorer, 'bank_0')
    sliders_of(explorer, 'bank_0')['layer'].slider.value = 3
    select(explorer, 'bank_1')
    assert sliders_of(explorer, 'bank_1')['layer'].slider.value == 1
    select(explorer, 'bank_0')
    assert sliders_of(explorer, 'bank_0')['layer'].slider.value == 3


def test_color_limits_and_log_apply_to_newly_selected_figures():
    explorer = DetectorExplorer(make_banks())
    explorer.color_control.log_button.value = True
    explorer.color_control.vmin = 1
    explorer.color_control.vmax = 10
    mappers = [f.view.colormapper for f in explorer.subplots.values()]
    assert all((m.cmin, m.cmax, m.norm) == (1, 10, 'log') for m in mappers)
    tab_mapper = explorer.tab_figs['bank_1'].view.colormapper
    assert tab_mapper.norm == 'linear'

    select(explorer, 'bank_1')
    assert (tab_mapper.cmin, tab_mapper.cmax, tab_mapper.norm) == (1, 10, 'log')
    select(explorer, 'bank_0')
    assert explorer.tab_figs['bank_0'].view.colormapper.norm == 'log'
    select(explorer, 'All')
    assert all((m.cmin, m.cmax, m.norm) == (1, 10, 'log') for m in mappers)


# ---------------------------------------------------------------------------
# 4. Masking
# ---------------------------------------------------------------------------


def test_no_masks_supplied_does_not_show_mask_control():
    explorer = DetectorExplorer(make_banks())
    assert explorer.mask_control is None


def test_destination_plots_show_correct_data_and_masks():
    det_banks = make_banks()
    masks = make_y_masks(det_banks)
    masks['bank_0']['m']['layer', 0]['y', 1] = True
    masks['bank_1']['m']['layer', 0]['y', 0] = True
    explorer = DetectorExplorer(det_banks, masks=masks)

    result = explorer.all_to_plot_nodes['bank_0']()
    assert sc.identical(
        result.drop_coords(result.coords),
        det_banks['bank_0'].assign_masks(masks['bank_0'])['layer', 0],
    )
    result = explorer.all_to_plot_nodes['bank_1']()
    assert sc.identical(
        result.drop_coords(result.coords),
        det_banks['bank_1'].assign_masks(masks['bank_1'])['layer', 0],
    )

    select(explorer, 'bank_1')
    result = explorer.all_to_plot_nodes['bank_1']()
    assert sc.identical(
        result.drop_coords(result.coords),
        det_banks['bank_1'].assign_masks(masks['bank_1'])['layer', 0],
    )


def test_adding_a_mask_affects_the_correct_mask():
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm'
    press(explorer, 'bank_0', 2, 1)

    expected = sc.array(
        dims=['layer', 'y'], values=[[False, True, False], [False, False, False]]
    )
    assert_identical(explorer.masks['bank_0']['m'], expected)

    select(explorer, 'bank_1')
    press(explorer, 'bank_1', 0, 0)
    expected = sc.array(
        dims=['layer', 'y'], values=[[True, False, False], [False, False, False]]
    )
    assert_identical(explorer.masks['bank_1']['m'], expected)


def test_masking_affects_only_selected_bank_and_leaves_data_unchanged():
    det_banks = make_banks()
    original = det_banks.copy(deep=True)
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    select(explorer, 'bank_1')
    explorer.mask_control.selector.value = 'm'
    press(explorer, 'bank_1', 1, 1)
    select(explorer, 'bank_0')
    press(explorer, 'bank_0', 2, 1)

    assert "m" in explorer.masks['bank_1']
    assert "m" in explorer.masks['bank_0']
    for bank in det_banks:
        assert "m" not in original[bank].masks


def test_masking_masks_with_different_dimensions():
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_x_y_masks(det_banks))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm_y'
    press(explorer, 'bank_0', 2, 1)
    select(explorer, 'bank_1')
    explorer.mask_control.selector.value = 'm_xy'
    press(explorer, 'bank_1', 0, 0)
    assert "m_y" in explorer.masks['bank_0']
    assert set(explorer.masks['bank_0']['m_y'].dims) == {'layer', 'y'}
    assert explorer.masks['bank_0']['m_y'].any()
    assert not explorer.masks['bank_1']['m_y'].any()
    assert "m_xy" in explorer.masks['bank_1']
    assert set(explorer.masks['bank_1']['m_xy'].dims) == {'layer', 'x', 'y'}
    assert explorer.masks['bank_1']['m_xy'].any()
    assert not explorer.masks['bank_0']['m_xy'].any()


@pytest.mark.parametrize(
    ('x', 'y', 'iy'),
    [(2.4, 0.6, 1), (2.6, 1.4, 1), (0.2, 1.6, 2)],
)
def test_click_coordinates_are_rounded_to_nearest_pixel_y_masks(x, y, iy):
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm'
    press(explorer, 'bank_0', x, y)
    expected = sc.zeros(
        sizes={dim: det_banks['bank_0'].sizes[dim] for dim in ['layer', 'y']},
        dtype=bool,
    )
    expected['layer', 0]['y', iy] = True
    assert_identical(explorer.masks['bank_0']["m"], expected)


@pytest.mark.parametrize(
    ('x', 'y', 'ix', 'iy'),
    [(2.4, 0.6, 2, 1), (2.6, 1.4, 3, 1), (0.2, 1.6, 0, 2)],
)
def test_click_coordinates_are_rounded_to_nearest_pixel_xy_masks(x, y, ix, iy):
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_x_y_masks(det_banks))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm_xy'
    press(explorer, 'bank_0', x, y)
    expected = sc.zeros(
        sizes={dim: det_banks['bank_0'].sizes[dim] for dim in ['layer', 'y', 'x']},
        dtype=bool,
    )
    expected['layer', 0]['y', iy]['x', ix] = True
    assert_identical(explorer.masks['bank_0']["m_xy"], expected)


def test_adding_a_mask_then_clicking_same_place_again_deletes_it():
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm'
    press(explorer, 'bank_0', 0, 1)
    assert np.array_equal(
        explorer.masks['bank_0']["m"]["layer", 0].values, [False, True, False]
    )
    press(explorer, 'bank_0', 0, 1)
    assert np.array_equal(
        explorer.masks['bank_0']["m"]["layer", 0].values, [False, False, False]
    )


def test_deleting_mask_by_click_affects_all_visible_masks():
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_x_y_masks(det_banks))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm_xy'
    press(explorer, 'bank_0', 0.2, 1)
    explorer.mask_control.selector.value = 'm_y'
    press(explorer, 'bank_0', 0.2, 1)
    assert not explorer.masks['bank_0']["m_xy"]["layer", 0].any()


def test_add_mask_with_small_coord():
    dg = make_banks()
    dg['bank_0'].coords['x'] = sc.arange('x', 4) * 0.1
    explorer = DetectorExplorer(dg, masks=make_x_y_masks(dg))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm_xy'
    press(explorer, 'bank_0', 0.3, 1)
    assert sc.identical(
        explorer.masks['bank_0']['m_xy']['layer', 0]['y', 1],
        sc.array(dims=['x'], values=[False, False, False, True], dtype=bool),
    )


def test_add_mask_with_small_coord_and_bin_edges():
    dg = make_banks()
    dg['bank_0'].coords['x'] = sc.arange('x', 5) * 0.1
    explorer = DetectorExplorer(dg, masks=make_x_y_masks(dg))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm_xy'
    press(explorer, 'bank_0', 0.201, 1)  # Just above the 3rd bin edge
    mask = explorer.masks['bank_0']['m_xy']['layer', 0]['y', 1]
    assert sc.identical(
        mask, sc.array(dims=['x'], values=[False, False, True, False], dtype=bool)
    )
    press(explorer, 'bank_0', 0.198, 1)  # Just below the 3rd bin edge
    assert sc.identical(
        mask, sc.array(dims=['x'], values=[False, True, True, False], dtype=bool)
    )


def test_add_mask_with_large_coord():
    dg = make_banks()
    dg['bank_0'].coords['x'] = sc.arange('x', 1000, 1004)
    explorer = DetectorExplorer(dg, masks=make_x_y_masks(dg))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm_xy'
    press(explorer, 'bank_0', 1001, 1)
    mask = explorer.masks['bank_0']['m_xy']['layer', 0]['y', 1]
    assert sc.identical(
        mask, sc.array(dims=['x'], values=[False, True, False, False], dtype=bool)
    )


def test_add_mask_with_large_coord_and_bin_edges():
    dg = make_banks()
    dg['bank_0'].coords['x'] = sc.arange('x', 1000, 1005)
    explorer = DetectorExplorer(dg, masks=make_x_y_masks(dg))
    select(explorer, 'bank_0')
    explorer.mask_control.selector.value = 'm_xy'
    press(explorer, 'bank_0', 1001.1, 1)
    mask = explorer.masks['bank_0']['m_xy']['layer', 0]['y', 1]
    assert sc.identical(
        mask, sc.array(dims=['x'], values=[False, True, False, False], dtype=bool)
    )
    press(explorer, 'bank_0', 1000.9, 1)  # Just below the 2nd bin edge
    assert sc.identical(
        mask, sc.array(dims=['x'], values=[True, True, False, False], dtype=bool)
    )


# ---------------------------------------------------------------------------
# 5. Mouse navigation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('bank', ['bank_0', 'bank_1', 'bank_2'])
def test_clicking_overview_subplot_opens_corresponding_bank(bank):
    explorer = DetectorExplorer(make_banks())
    ax = explorer.subplots[bank].ax
    fire(explorer.main_figure.canvas, ax, 'button_press_event', 1, 1)
    assert explorer.current_key == bank


def test_clicking_outside_axes_or_on_colorbar_does_nothing():
    explorer = DetectorExplorer(make_banks())
    canvas = explorer.main_figure.canvas
    fire(canvas, None, 'button_press_event')
    assert explorer.current_tab == 0

    subplot_axes = {f.ax for f in explorer.subplots.values()}
    colorbar_axes = [ax for ax in explorer.main_figure.axes if ax not in subplot_axes]
    assert colorbar_axes
    for ax in colorbar_axes:
        fire(canvas, ax, 'button_press_event', 0.5, 0.5)
        assert explorer.current_tab == 0


def test_pan_zoom_mode_suppresses_navigation_and_masking():
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    ax = explorer.subplots['bank_1'].ax
    explorer.main_figure.canvas.toolbar.pan()
    canvas = explorer.main_figure.canvas
    fire(canvas, ax, 'button_press_event', 1, 1)
    assert explorer.current_tab == 0

    select(explorer, 'bank_1')
    explorer.mask_control.selector.value = 'm'
    explorer.tab_figs['bank_1'].fig.canvas.toolbar.zoom()
    press(explorer, 'bank_1', 1, 1)
    assert not explorer.masks['bank_1']['m'].any()


def test_overview_clicks_do_not_change_masks():
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    explorer.mask_control.selector.value = 'm'
    fire(
        explorer.main_figure.canvas,
        explorer.subplots['bank_1'].ax,
        'button_press_event',
        1,
        1,
    )
    assert explorer.current_key == 'bank_1'
    assert not any(m['m'].any() for m in explorer.masks.values())


# ---------------------------------------------------------------------------
# 7. Clearing and saving
# ---------------------------------------------------------------------------


def test_clear_on_bank_affects_only_that_banks_selected_slice():
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    select(explorer, 'bank_1')
    explorer.mask_control.selector.value = 'm'
    press(explorer, 'bank_1', 1, 1)
    explorer.bank_dim_sliders['bank_1'][0].slider.value = 1
    press(explorer, 'bank_1', 1, 1)

    explorer.mask_control.clear_button.click()

    assert explorer.masks['bank_1']['m']['layer', 1].values.sum() == 0
    assert explorer.masks['bank_1']['m']['layer', 0].values.sum() == 1


def test_clear_button_is_disabled_on_overview_tab():
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    select(explorer, 'bank_1')
    explorer.mask_control.selector.value = 'm'
    assert not explorer.mask_control.clear_button.disabled
    press(explorer, 'bank_1', 1, 1)
    select(explorer, 'All')
    assert explorer.mask_control.clear_button.disabled


def test_save_with_empty_filename_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    control = explorer.mask_control
    select(explorer, 'bank_1')
    control.selector.value = 'm'
    press(explorer, 'bank_1', 1, 1)
    assert control.save_button.disabled
    control.filename_input.value = 'masks.h5'
    assert not control.save_button.disabled
    control.filename_input.value = ''
    assert control.save_button.disabled

    control.save_button.click()
    assert list(tmp_path.iterdir()) == []


def test_save_masks_round_trip(tmp_path):
    det_banks = make_banks()
    explorer = DetectorExplorer(det_banks, masks=make_y_masks(det_banks))
    explorer.mask_control.selector.value = 'm'
    select(explorer, 'bank_2')
    press(explorer, 'bank_2', 1, 0)
    select(explorer, 'bank_1')
    press(explorer, 'bank_1', 1, 1)

    path = tmp_path / 'masks.h5'
    explorer.mask_control.filename_input.value = str(path)
    explorer.mask_control.save_button.click()

    loaded = read_pixel_masks([path]).by_bank
    for bank, masks in explorer.masks.items():
        for k, m in masks.items():
            assert_identical(loaded[bank][k], m)


def test_save_does_not_include_empty_masks_not_in_original(tmp_path):
    det_banks = make_banks()
    # No mask on bank 1
    masks = {
        "bank_0": {
            "m": sc.zeros(
                sizes={dim: det_banks.sizes[dim] for dim in ("layer", "y")}, dtype=bool
            )
        }
    }
    explorer = DetectorExplorer(det_banks, masks=masks)
    explorer.mask_control.selector.value = 'm'
    select(explorer, 'bank_0')
    press(explorer, 'bank_0', 0, 0)
    path = tmp_path / 'masks.h5'
    explorer.mask_control.filename_input.value = str(path)
    explorer.mask_control.save_button.click()
    loaded = read_pixel_masks([path]).by_bank
    assert "bank_0" in loaded
    assert "m" in loaded["bank_0"]
    assert "bank_1" not in loaded


def test_save_does_not_include_empty_masks_not_in_original_after_undo_click(tmp_path):
    det_banks = make_banks()
    # No mask on bank 1
    masks = {
        "bank_0": {
            "m": sc.zeros(
                sizes={dim: det_banks.sizes[dim] for dim in ("layer", "y")}, dtype=bool
            )
        }
    }
    explorer = DetectorExplorer(det_banks, masks=masks)
    explorer.mask_control.selector.value = 'm'
    select(explorer, 'bank_0')
    press(explorer, 'bank_0', 0, 0)
    press(explorer, 'bank_0', 0, 0)  # Undo the previous press
    select(explorer, 'bank_1')
    press(explorer, 'bank_1', 1, 1)
    press(explorer, 'bank_1', 1, 1)  # Undo the previous press
    path = tmp_path / 'masks.h5'
    explorer.mask_control.filename_input.value = str(path)
    explorer.mask_control.save_button.click()
    loaded = read_pixel_masks([path]).by_bank
    # Bank 0 should still be included because it was in the original masks
    assert "bank_0" in loaded
    assert "m" in loaded["bank_0"]
    assert "bank_1" not in loaded


def test_save_does_not_include_empty_masks_not_in_original_after_clear_button(tmp_path):
    det_banks = make_banks()
    # No mask on bank 1
    masks = {
        "bank_0": {
            "m": sc.zeros(
                sizes={dim: det_banks.sizes[dim] for dim in ("layer", "y")}, dtype=bool
            )
        }
    }
    explorer = DetectorExplorer(det_banks, masks=masks)
    explorer.mask_control.selector.value = 'm'
    select(explorer, 'bank_0')
    press(explorer, 'bank_0', 0, 0)
    explorer.mask_control.clear_button.click()  # Clear the mask on bank 0
    select(explorer, 'bank_1')
    press(explorer, 'bank_1', 1, 1)
    explorer.mask_control.clear_button.click()  # Clear the mask on bank 1
    path = tmp_path / 'masks.h5'
    explorer.mask_control.filename_input.value = str(path)
    explorer.mask_control.save_button.click()
    loaded = read_pixel_masks([path]).by_bank
    # Bank 0 should still be included because it was in the original masks
    assert "bank_0" in loaded
    assert "m" in loaded["bank_0"]
    assert "bank_1" not in loaded


# ---------------------------------------------------------------------------
# 8. Colour controls
# ---------------------------------------------------------------------------


def test_min_and_max_work_independently_and_only_update_active_figures():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    mapper = explorer.tab_figs['bank_1'].view.colormapper

    explorer.color_control.vmin = 3
    assert (mapper.cmin, mapper.cmax) == (3, mapper.cmax)
    explorer.color_control.vmax = 8
    assert (mapper.cmin, mapper.cmax) == (3, 8)

    for other in (explorer.tab_figs['bank_0'], *explorer.subplots.values()):
        assert (other.view.colormapper.cmin, other.view.colormapper.cmax) != (3, 8)

    select(explorer, 'bank_0')
    mapper0 = explorer.tab_figs['bank_0'].view.colormapper
    # After selecting bank_0, the colormapper should update to the values in the boxes
    assert (mapper0.cmin, mapper0.cmax) == (3, 8)


def test_fit_to_displayed_data():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    explorer.color_control.vmin = 2
    explorer.color_control.vmax = 5
    explorer.bank_dim_sliders['bank_1'][0].slider.value = 1
    mapper = explorer.tab_figs['bank_1'].view.colormapper
    assert (mapper.cmin, mapper.cmax) == (2, 5)

    explorer.color_control.fit_button.click()
    expected = explorer.data['bank_1']['layer', 1].data
    assert mapper.cmin == expected.min().value
    assert mapper.cmax == expected.max().value


def test_log_toggle_updates_normalization():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    mapper = explorer.tab_figs['bank_1'].view.colormapper

    explorer.color_control.log_button.value = True
    assert mapper.norm == 'log'
    explorer.color_control.log_button.value = False
    assert mapper.norm == 'linear'


@pytest.mark.parametrize(
    'values',
    [
        [0.0, 0.0],
        [-3.0, -1.0],
        [5.0, 5.0],
        [float('nan'), float('nan')],
        [-2.0, 4.0],
    ],
)
@pytest.mark.parametrize('log', [False, True])
def test_fit_and_tab_switching_survive_degenerate_data(values, log):
    data = make_banks({'y': 2, 'x': 2}, count=2)
    for index, array in enumerate(data.values()):
        array.values[...] = values[index]
    explorer = DetectorExplorer(data)
    explorer.color_control.log_button.value = log

    explorer.color_control.fit_button.click()
    select(explorer, 'bank_0')
    explorer.color_control.fit_button.click()
    select(explorer, 'bank_1')
    select(explorer, 'All')
