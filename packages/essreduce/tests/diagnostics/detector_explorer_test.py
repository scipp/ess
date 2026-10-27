# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.reduce.diagnostics.detector_explorer import DetectorExplorer


@pytest.fixture(autouse=True)
def widget_backend():
    backend = plt.get_backend()
    plt.switch_backend('module://ipympl.backend_nbagg')
    yield
    plt.close('all')
    plt.switch_backend(backend)


def make_banks(sizes=None, count=3):
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
        expected = expected.assign_masks(
            {'mask': sc.zeros(sizes=expected.sizes, dtype=bool)}
        ).transpose(visible_dims)
        assert_identical(explorer.apply_masks_nodes[bank](), expected)
        assert_identical(explorer.all_apply_masks_nodes[bank](), expected)
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


@pytest.mark.parametrize(
    'sizes',
    [
        {'y': 3, 'x': 4},
        {'layer': 2, 'y': 3, 'x': 4},
        {'time': 3, 'layer': 2, 'y': 3, 'x': 4},
    ],
)
def test_one_initial_slider_per_hidden_dimension(sizes):
    explorer = DetectorExplorer(make_banks(sizes))
    hidden_dims = list(sizes)[:-2]

    for sliders in [explorer.all_dim_sliders, *explorer.bank_dim_sliders.values()]:
        assert [slider.dim for slider in sliders] == hidden_dims
        for slider in sliders:
            assert slider.slider.min == 0
            assert slider.slider.max == sizes[slider.dim] - 1
            assert slider.slider.value == 0
            assert slider.sum.value is False
            assert slider.value == (slider.dim, 0)


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
        explorer.apply_masks_nodes['bank_0']().data,
        explorer.data['bank_0']['layer', 1].data,
    )


def test_single_bank_clear_uses_bank_slider_slice():
    explorer = DetectorExplorer(make_banks(count=1))
    explorer.bank_dim_sliders['bank_0'][0].slider.value = 1
    explorer.masks['bank_0'].values[...] = True

    explorer.mask_control.clear_button.click()

    assert explorer.masks['bank_0']['layer', 0].values.all()
    assert not explorer.masks['bank_0']['layer', 1].values.any()


def test_single_bank_color_controls_update_bank_figure():
    explorer = DetectorExplorer(make_banks(count=1))

    explorer.color_control.vmin.value = '1'
    explorer.color_control.vmax.value = '10'

    colormapper = explorer.tab_figs['bank_0'].view.colormapper
    assert colormapper.cmin == 1
    assert colormapper.cmax == 10


def test_initial_masks_are_false_and_independent_between_banks():
    data = make_banks()
    explorer = DetectorExplorer(data)

    for bank, array in data.items():
        assert_identical(explorer.masks[bank], sc.zeros(sizes=array.sizes, dtype=bool))
    explorer.masks['bank_0'].values[...] = True
    for bank in ('bank_1', 'bank_2'):
        assert not explorer.masks[bank].values.any()


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
    out = explorer.data[bank]
    mask = explorer.masks[bank]
    for dim, ind in indices.items():
        if ind is None:
            out = out.sum(dim)
            mask = mask.all(dim)
        else:
            out = out[dim, ind]
            mask = mask[dim, ind]
    return out, mask


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


def toggle_model(mask, mode, x, y):
    """Reference for a mask click on a (y, x) numpy array."""
    region = {
        'single': (y, x),
        'horizontal': (y, slice(None)),
        'vertical': (slice(None), x),
    }[mode]
    mask[region] = not mask[region].all()


def layer0(explorer, bank='bank_0'):
    return explorer.masks[bank]['layer', 0].values


# ---------------------------------------------------------------------------
# 2. Slicing and summing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('view', ['bank', 'all'])
def test_slider_displays_selected_slice_including_first_and_last(index, view):
    explorer = DetectorExplorer(make_banks())
    sliders = sliders_of(explorer, 'bank_1' if view == 'bank' else 'All')
    nodes = (
        explorer.apply_masks_nodes
        if view == 'bank'
        else (explorer.all_apply_masks_nodes)
    )
    if view == 'all':
        select(explorer, 'All')
    sliders['layer'].slider.value = index

    result = nodes['bank_1']()
    assert_identical(result.data, explorer.data['bank_1']['layer', index].data)
    assert not result.masks['mask'].values.any()


@pytest.mark.parametrize('view', ['bank', 'all'])
def test_sum_displays_reduction_and_disabling_restores_slice(view):
    explorer = DetectorExplorer(make_banks())
    name = 'bank_1' if view == 'bank' else 'All'
    nodes = (
        explorer.apply_masks_nodes
        if view == 'bank'
        else (explorer.all_apply_masks_nodes)
    )
    select(explorer, name)
    slider = sliders_of(explorer, name)['layer']
    slider.slider.value = 1

    slider.sum.value = True
    assert_identical(nodes['bank_1']().data, explorer.data['bank_1'].sum('layer').data)
    slider.sum.value = False
    assert_identical(nodes['bank_1']().data, explorer.data['bank_1']['layer', 1].data)


def test_multiple_hidden_dimensions_support_mixed_slicing_and_summing():
    explorer = DetectorExplorer(make_banks({'time': 3, 'layer': 2, 'y': 3, 'x': 4}))
    sliders = sliders_of(explorer, 'bank_2')
    sliders['time'].sum.value = True
    sliders['layer'].slider.value = 1
    expected = explorer.data['bank_2']['layer', 1].sum('time')
    assert_identical(explorer.apply_masks_nodes['bank_2']().data, expected.data)

    sliders['time'].sum.value = False
    sliders['time'].slider.value = 2
    sliders['layer'].sum.value = True
    expected = explorer.data['bank_2']['time', 2].sum('layer')
    assert_identical(explorer.apply_masks_nodes['bank_2']().data, expected.data)


@pytest.mark.parametrize('view', ['bank', 'all'])
def test_summed_masks_require_all_contributing_elements(view):
    explorer = DetectorExplorer(make_banks())
    nodes = (
        explorer.apply_masks_nodes
        if view == 'bank'
        else (explorer.all_apply_masks_nodes)
    )
    sliders = sliders_of(explorer, 'bank_1' if view == 'bank' else 'All')
    sliders['layer'].sum.value = True
    explorer.masks['bank_1']['layer', 0]['y', 1]['x', 2].value = True
    explorer.masks['bank_1']['layer', 1]['y', 1]['x', 2].value = True
    explorer.masks['bank_1']['layer', 0]['y', 0]['x', 0].value = True
    explorer.mask_selection_nodes['bank_1'].notify_children(message=None)
    explorer.all_mask_selection_nodes['bank_1'].notify_children(message=None)

    mask = nodes['bank_1']().masks['mask'].values
    expected = sc.zeros(sizes={'y': 3, 'x': 4}, dtype=bool).values
    expected[1, 2] = True
    assert (mask == expected).all()


def test_all_index_beyond_smaller_bank_is_blank_and_unmasked():
    explorer = DetectorExplorer(make_uneven_banks())
    sliders = sliders_of(explorer, 'All')
    sliders['layer'].slider.value = 3

    result = explorer.all_apply_masks_nodes['bank_1']()
    assert sc.isnan(result.data).values.all()
    assert not result.masks['mask'].values.any()
    expected = explorer.data['bank_0']['time', 0]['layer', 3]
    assert_identical(explorer.all_apply_masks_nodes['bank_0']().data, expected.data)

    sliders['layer'].slider.value = 0
    sliders['time'].slider.value = 2
    result = explorer.all_apply_masks_nodes['bank_2']()
    assert sc.isnan(result.data).values.all()
    assert not result.masks['mask'].values.any()


def test_slider_updates_preserve_manual_color_limits():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    explorer.color_control.vmin.value = '2'
    explorer.color_control.vmax.value = '9'
    explorer.bank_dim_sliders['bank_1'][0].slider.value = 1

    colormapper = explorer.tab_figs['bank_1'].view.colormapper
    assert (colormapper.cmin, colormapper.cmax) == (2, 9)
    assert explorer.color_control.vmin.value == '2'


# ---------------------------------------------------------------------------
# 3. Tab switching
# ---------------------------------------------------------------------------

TRANSITIONS = [
    ('All', 'bank_1'),
    ('bank_1', 'All'),
    ('bank_0', 'bank_1'),
    ('bank_1', 'bank_0'),
]


def test_current_bank_and_figures_match_selected_tab():
    explorer = DetectorExplorer(make_banks())
    for name in ['bank_1', 'All', 'bank_2', 'bank_0', 'All']:
        select(explorer, name)
        assert explorer.current_key == name
        if name == 'All':
            assert explorer.current_figs == list(explorer.subplots.values())
        else:
            assert explorer.current_figs == [explorer.tab_figs[name]]


@pytest.mark.parametrize(('start', 'end'), TRANSITIONS)
def test_mixed_sum_settings_transfer(start, end):
    explorer = DetectorExplorer(make_uneven_banks())
    select(explorer, start)
    sliders_of(explorer, start)['time'].sum.value = True
    sliders_of(explorer, start)['layer'].sum.value = False
    select(explorer, end)
    assert sliders_of(explorer, end)['time'].sum.value is True
    assert sliders_of(explorer, end)['layer'].sum.value is False

    sliders_of(explorer, end)['time'].sum.value = False
    sliders_of(explorer, end)['layer'].sum.value = True
    select(explorer, start)
    assert sliders_of(explorer, start)['time'].sum.value is False
    assert sliders_of(explorer, start)['layer'].sum.value is True


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


def test_summed_dimensions_do_not_overwrite_stored_indices():
    explorer = DetectorExplorer(make_uneven_banks())
    all_layer = sliders_of(explorer, 'All')['layer']
    all_layer.slider.value = 2
    select(explorer, 'bank_1')
    bank_layer = sliders_of(explorer, 'bank_1')['layer']
    bank_layer.slider.value = 1
    bank_layer.sum.value = True

    select(explorer, 'All')
    assert all_layer.sum.value is True
    stored = all_layer.slider.value
    select(explorer, 'bank_1')
    assert bank_layer.slider.value == 1
    assert all_layer.slider.value == stored
    bank_layer.sum.value = False
    select(explorer, 'All')
    assert all_layer.sum.value is False
    assert all_layer.slider.value == 1


@pytest.mark.parametrize(('start', 'end'), TRANSITIONS)
def test_destination_plots_show_correct_data_and_masks(start, end):
    explorer = DetectorExplorer(make_banks())
    select(explorer, start)
    explorer.masks['bank_1']['layer', 0]['y', 1]['x', 2].value = True
    explorer.mask_selection_nodes['bank_1'].notify_children(message=None)
    explorer.all_mask_selection_nodes['bank_1'].notify_children(message=None)
    select(explorer, end)

    for nodes in (explorer.apply_masks_nodes, explorer.all_apply_masks_nodes):
        result = nodes['bank_1']()
        expected, mask = expected_display(explorer, 'bank_1', layer=0)
        assert_identical(result.data, expected.data)
        assert_identical(result.masks['mask'], mask)
        assert result.masks['mask'].values[1, 2]


def test_color_limits_and_log_apply_to_newly_selected_figures():
    explorer = DetectorExplorer(make_banks())
    explorer.color_control.log_button.value = True
    explorer.color_control.vmin.value = '1'
    explorer.color_control.vmax.value = '10'
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


def test_repeated_tab_switches_do_not_duplicate_callbacks_or_actions():
    explorer = DetectorExplorer(make_banks())
    canvas, _ = bank_canvas(explorer, 'bank_1')
    main = explorer.main_figure.canvas
    counts = (
        len(canvas.callbacks.callbacks.get('button_press_event', {})),
        len(main.callbacks.callbacks.get('button_press_event', {})),
    )
    for _ in range(5):
        for name in ('bank_1', 'All', 'bank_2', 'bank_1'):
            select(explorer, name)
    assert counts == (
        len(canvas.callbacks.callbacks.get('button_press_event', {})),
        len(main.callbacks.callbacks.get('button_press_event', {})),
    )

    explorer.mask_control.mode = 'single'
    press(explorer, 'bank_1', 1, 1)
    assert explorer.masks['bank_1'].values.sum() == 1


# ---------------------------------------------------------------------------
# 4. Mouse navigation
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
    explorer = DetectorExplorer(make_banks())
    ax = explorer.subplots['bank_1'].ax
    explorer.main_figure.canvas.toolbar.pan()
    canvas = explorer.main_figure.canvas
    fire(canvas, ax, 'button_press_event', 1, 1)
    assert explorer.current_tab == 0

    select(explorer, 'bank_1')
    explorer.mask_control.mode = 'single'
    explorer.tab_figs['bank_1'].fig.canvas.toolbar.zoom()
    press(explorer, 'bank_1', 1, 1)
    assert not explorer.masks['bank_1'].values.any()


def test_overview_clicks_do_not_change_masks():
    explorer = DetectorExplorer(make_banks())
    explorer.mask_control.mode = 'single'
    fire(
        explorer.main_figure.canvas,
        explorer.subplots['bank_1'].ax,
        'button_press_event',
        1,
        1,
    )
    assert explorer.current_key == 'bank_1'
    assert not any(m.values.any() for m in explorer.masks.values())


# ---------------------------------------------------------------------------
# 5. Masking
# ---------------------------------------------------------------------------

MODES = ['single', 'horizontal', 'vertical']
ALL_TRANSITIONS = [(a, b) for a in MODES for b in MODES if a != b]


def test_mode_buttons_are_mutually_exclusive():
    explorer = DetectorExplorer(make_banks())
    control = explorer.mask_control
    assert control.mode is None
    for mode in MODES:
        control.mode_buttons[mode].value = True
        assert control.mode == mode
        assert [b.value for b in control.mode_buttons.values()].count(True) == 1
    control.mode_buttons['vertical'].value = False
    assert control.mode is None

    select(explorer, 'bank_0')
    press(explorer, 'bank_0', 1, 1)
    assert not explorer.masks['bank_0'].values.any()


@pytest.mark.parametrize('mode', MODES)
def test_mode_toggles_exactly_the_intended_elements(mode):
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.mask_control.mode = mode
    press(explorer, 'bank_0', 2, 1)

    expected = sc.zeros(sizes=explorer.data['bank_0'].sizes, dtype=bool).values
    toggle_model(expected[0], mode, 2, 1)
    assert (explorer.masks['bank_0'].values == expected).all()


@pytest.mark.parametrize(('first', 'second'), ALL_TRANSITIONS)
def test_mode_transitions_use_new_mode_even_on_same_pixel(first, second):
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    expected = sc.zeros(sizes=explorer.data['bank_0'].sizes, dtype=bool).values
    explorer.masks['bank_0']['layer', 0]['y', 0]['x', 3].value = True
    expected[0, 0, 3] = True

    explorer.mask_control.mode = first
    for x, y in [(2, 1), (2, 1)]:
        press(explorer, 'bank_0', x, y)
        toggle_model(expected[0], first, x, y)
        assert (explorer.masks['bank_0'].values == expected).all()

    explorer.mask_control.mode = second
    for x, y in [(2, 1), (2, 1), (1, 2), (2, 1)]:
        press(explorer, 'bank_0', x, y)
        toggle_model(expected[0], second, x, y)
        assert (explorer.masks['bank_0'].values == expected).all()


def test_partially_masked_selection_becomes_masked_and_full_becomes_unmasked():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.masks['bank_0']['layer', 0]['y', 1]['x', 0].value = True
    explorer.mask_control.mode = 'horizontal'

    press(explorer, 'bank_0', 2, 1)
    assert layer0(explorer)[1].all()
    press(explorer, 'bank_0', 2, 1)
    assert not layer0(explorer)[1].any()


def test_hidden_sliced_dimension_restricts_edits():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.bank_dim_sliders['bank_0'][0].slider.value = 1
    explorer.mask_control.mode = 'horizontal'
    press(explorer, 'bank_0', 0, 2)

    assert explorer.masks['bank_0']['layer', 1].values[2].all()
    assert explorer.masks['bank_0'].values.sum() == 4


def test_summed_dimension_applies_edits_across_full_extent():
    explorer = DetectorExplorer(make_banks({'time': 3, 'layer': 2, 'y': 3, 'x': 4}))
    select(explorer, 'bank_0')
    sliders = sliders_of(explorer, 'bank_0')
    sliders['time'].sum.value = True
    sliders['layer'].slider.value = 1
    explorer.mask_control.mode = 'vertical'
    press(explorer, 'bank_0', 3, 0)

    mask = explorer.masks['bank_0']
    assert mask['layer', 1]['x', 3].values.all()
    assert mask.values.sum() == 3 * 3
    assert not mask['layer', 0].values.any()


def test_masking_affects_only_selected_bank_and_leaves_data_unchanged():
    data = make_banks()
    original = data.copy(deep=True)
    explorer = DetectorExplorer(data)
    select(explorer, 'bank_1')
    explorer.mask_control.mode = 'vertical'
    press(explorer, 'bank_1', 1, 1)

    assert explorer.masks['bank_1'].values.any()
    for bank in ('bank_0', 'bank_2'):
        assert not explorer.masks[bank].values.any()
    for bank in data:
        assert_identical(explorer.data[bank].data, original[bank].data)
        assert not explorer.data[bank].masks


@pytest.mark.parametrize(
    ('x', 'y', 'ix', 'iy'),
    [(2.4, 0.6, 2, 1), (2.6, 1.4, 3, 1), (0.2, 1.6, 0, 2)],
)
def test_click_coordinates_are_rounded_to_nearest_pixel(x, y, ix, iy):
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'single'
    press(explorer, 'bank_0', x, y)
    expected = sc.zeros(sizes={'y': 3, 'x': 4}, dtype=bool).values
    expected[iy, ix] = True
    assert (layer0(explorer) == expected).all()


@pytest.mark.parametrize('mode', MODES)
def test_reversed_visible_dimensions_target_correct_pixels(mode):
    explorer = DetectorExplorer(make_banks(), show_dims=['x', 'y'])
    select(explorer, 'bank_0')
    explorer.mask_control.mode = mode
    # The horizontal axis now shows 'y' (size 3), the vertical axis 'x' (size 4).
    press(explorer, 'bank_0', 2, 3)

    expected = sc.zeros(sizes={'y': 3, 'x': 4}, dtype=bool).values
    toggle_model(expected.T, mode, 2, 3)
    assert (layer0(explorer) == expected).all()


def test_masks_refresh_in_both_views_after_switching_tabs():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    explorer.mask_control.mode = 'single'
    press(explorer, 'bank_1', 2, 1)
    select(explorer, 'All')
    select(explorer, 'bank_1')
    select(explorer, 'bank_0')
    select(explorer, 'bank_1')

    for nodes in (explorer.apply_masks_nodes, explorer.all_apply_masks_nodes):
        mask = nodes['bank_1']().masks['mask'].values
        assert mask[1, 2]
        assert mask.sum() == 1
        assert not nodes['bank_0']().masks['mask'].values.any()


# ---------------------------------------------------------------------------
# 6. Continuous masking
# ---------------------------------------------------------------------------


def handler_counts(explorer, bank):
    canvas, _ = bank_canvas(explorer, bank)
    return {
        name: len(canvas.callbacks.callbacks.get(name, {}))
        for name in (
            'button_press_event',
            'motion_notify_event',
            'button_release_event',
        )
    }


def test_without_continuous_mode_only_initial_click_changes_masks():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'single'
    press(explorer, 'bank_0', 1, 1)
    move(explorer, 'bank_0', 2, 1)
    move(explorer, 'bank_0', 3, 2)
    assert layer0(explorer).sum() == 1


@pytest.mark.parametrize('mode', MODES)
def test_continuous_masking_follows_movement_without_repeated_toggles(mode):
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.mask_control.mode = mode
    explorer.mask_control.continuous_button.value = True
    expected = sc.zeros(sizes={'y': 3, 'x': 4}, dtype=bool).values

    press(explorer, 'bank_0', 1, 1)
    toggle_model(expected, mode, 1, 1)
    assert (layer0(explorer) == expected).all()

    # Stays in the same pixel/row/column: nothing changes.
    same = {'single': [(1, 1)], 'horizontal': [(3, 1)], 'vertical': [(1, 2)]}[mode]
    for x, y in same * 2:
        move(explorer, 'bank_0', x, y)
        assert (layer0(explorer) == expected).all()

    # New target: changes exactly once, even when moving repeatedly within it.
    for x, y in [(3, 2), (3, 2)]:
        move(explorer, 'bank_0', x, y)
    toggle_model(expected, mode, 3, 2)
    assert (layer0(explorer) == expected).all()


def test_leaving_axes_has_no_effect_and_release_stops_masking():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'single'
    explorer.mask_control.continuous_button.value = True
    before = handler_counts(explorer, 'bank_0')

    press(explorer, 'bank_0', 1, 1)
    move_outside(explorer, 'bank_0')
    assert layer0(explorer).sum() == 1
    release(explorer, 'bank_0', 1, 1)
    assert handler_counts(explorer, 'bank_0') == before
    move(explorer, 'bank_0', 2, 2)
    assert layer0(explorer).sum() == 1


def test_release_outside_axes_stops_masking_and_new_gesture_starts_cleanly():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'single'
    explorer.mask_control.continuous_button.value = True
    before = handler_counts(explorer, 'bank_0')

    for _ in range(3):
        press(explorer, 'bank_0', 0, 0)
        move(explorer, 'bank_0', 1, 0)
        release(explorer, 'bank_0', outside=True)
        assert handler_counts(explorer, 'bank_0') == before
        move(explorer, 'bank_0', 2, 0)
        # Gesture toggled (0,0) and (1,0) once each; (2,0) is untouched.
        assert layer0(explorer)[0].tolist() == [True, True, False, False]
        press(explorer, 'bank_0', 0, 0)  # untoggle (0, 0)
        press(explorer, 'bank_0', 1, 0)  # untoggle (1, 0)
        release(explorer, 'bank_0', outside=True)
        assert not layer0(explorer).any()
        assert handler_counts(explorer, 'bank_0') == before


def test_continuous_state_and_icon_stay_consistent():
    explorer = DetectorExplorer(make_banks())
    button = explorer.mask_control.continuous_button
    assert (button.value, button.icon) == (False, 'unlock')
    button.value = True
    assert button.icon == 'lock'
    button.value = False
    assert button.icon == 'unlock'


# ---------------------------------------------------------------------------
# 7. Clearing and saving
# ---------------------------------------------------------------------------


def fill_all_masks(explorer):
    for mask in explorer.masks.values():
        mask.values[...] = True


def test_clear_on_bank_affects_only_that_banks_selected_slice():
    explorer = DetectorExplorer(make_banks())
    fill_all_masks(explorer)
    select(explorer, 'bank_1')
    explorer.bank_dim_sliders['bank_1'][0].slider.value = 1
    explorer.mask_control.clear_button.click()

    assert not explorer.masks['bank_1']['layer', 1].values.any()
    assert explorer.masks['bank_1']['layer', 0].values.all()
    for bank in ('bank_0', 'bank_2'):
        assert explorer.masks[bank].values.all()


def test_clear_on_all_affects_selected_slice_in_every_bank():
    explorer = DetectorExplorer(make_banks())
    fill_all_masks(explorer)
    sliders_of(explorer, 'All')['layer'].slider.value = 1
    explorer.mask_control.clear_button.click()

    for mask in explorer.masks.values():
        assert not mask['layer', 1].values.any()
        assert mask['layer', 0].values.all()


def test_clear_with_summed_dimension_clears_full_extent_only_in_selection():
    explorer = DetectorExplorer(make_banks({'time': 3, 'layer': 2, 'y': 3, 'x': 4}))
    fill_all_masks(explorer)
    select(explorer, 'bank_0')
    sliders = sliders_of(explorer, 'bank_0')
    sliders['time'].sum.value = True
    sliders['layer'].slider.value = 1
    explorer.mask_control.clear_button.click()

    mask = explorer.masks['bank_0']
    assert not mask['layer', 1].values.any()
    assert mask['layer', 0].values.all()
    assert explorer.masks['bank_1'].values.all()


def test_clear_refreshes_both_views_and_empty_clear_is_harmless():
    explorer = DetectorExplorer(make_banks())
    fill_all_masks(explorer)
    for node in explorer.mask_selection_nodes.values():
        node.notify_children(message=None)
    for node in explorer.all_mask_selection_nodes.values():
        node.notify_children(message=None)
    select(explorer, 'bank_1')
    explorer.mask_control.clear_button.click()
    for nodes in (explorer.apply_masks_nodes, explorer.all_apply_masks_nodes):
        assert not nodes['bank_1']().masks['mask'].values.any()
        assert nodes['bank_0']().masks['mask'].values.all()

    explorer.mask_control.clear_button.click()
    assert not explorer.masks['bank_1']['layer', 0].values.any()
    assert explorer.masks['bank_1']['layer', 1].values.all()


def test_filename_controls_save_button_and_empty_filename_writes_nothing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    explorer = DetectorExplorer(make_banks())
    control = explorer.mask_control
    assert control.save_button.disabled
    control.filename_input.value = 'masks.h5'
    assert not control.save_button.disabled
    control.filename_input.value = ''
    assert control.save_button.disabled

    control.save_button.click()
    assert list(tmp_path.iterdir()) == []


def test_saved_hdf5_round_trips_all_banks_regardless_of_selection(tmp_path):
    explorer = DetectorExplorer(make_uneven_banks())
    rng = np.random.default_rng(1)
    for mask in explorer.masks.values():
        mask.values[...] = rng.random(mask.shape) > 0.5
    select(explorer, 'bank_2')
    sliders_of(explorer, 'bank_2')['layer'].slider.value = 2
    sliders_of(explorer, 'bank_2')['time'].sum.value = True

    path = tmp_path / 'masks.h5'
    explorer.mask_control.filename_input.value = str(path)
    explorer.mask_control.save_button.click()

    loaded = sc.io.load_hdf5(path)
    assert list(loaded) == list(explorer.masks)
    for bank, mask in explorer.masks.items():
        assert_identical(loaded[bank], mask)


# ---------------------------------------------------------------------------
# 8. Colour controls
# ---------------------------------------------------------------------------


def test_min_and_max_work_independently_and_only_update_active_figures():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    mapper = explorer.tab_figs['bank_1'].view.colormapper
    auto_min, auto_max = mapper.cmin, mapper.cmax

    explorer.color_control.vmin.value = '3'
    assert (mapper.cmin, mapper.cmax) == (3, auto_max)
    explorer.color_control.vmin.value = ''
    explorer.color_control.vmax.value = '8'
    assert (mapper.cmin, mapper.cmax) == (auto_min, 8)
    explorer.color_control.vmin.value = '2'
    assert (mapper.cmin, mapper.cmax) == (2, 8)

    for other in (explorer.tab_figs['bank_0'], *explorer.subplots.values()):
        assert (other.view.colormapper.cmin, other.view.colormapper.cmax) != (2, 8)


def test_emptying_fields_restores_automatic_bounds():
    explorer = DetectorExplorer(make_banks())
    mapper = explorer.subplots['bank_0'].view.colormapper
    auto = (mapper.cmin, mapper.cmax)
    explorer.color_control.vmin.value = '3'
    explorer.color_control.vmax.value = '5'
    explorer.color_control.vmin.value = ''
    explorer.color_control.vmax.value = ''
    assert (mapper.cmin, mapper.cmax) == auto


def test_fit_clears_bounds_and_fits_displayed_data():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    explorer.color_control.vmin.value = '2'
    explorer.color_control.vmax.value = '5'
    explorer.bank_dim_sliders['bank_1'][0].slider.value = 1
    mapper = explorer.tab_figs['bank_1'].view.colormapper
    assert (mapper.cmin, mapper.cmax) == (2, 5)

    explorer.color_control.fit_button.click()
    expected = explorer.data['bank_1']['layer', 1].data
    assert explorer.color_control.vmin.value == ''
    assert explorer.color_control.vmax.value == ''
    assert mapper.cmin == expected.min().value
    assert mapper.cmax == expected.max().value


def test_log_toggle_updates_normalization_and_resets_bounds():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    mapper = explorer.tab_figs['bank_1'].view.colormapper
    explorer.color_control.vmin.value = '2'
    explorer.color_control.vmax.value = '5'

    explorer.color_control.log_button.value = True
    assert mapper.norm == 'log'
    assert explorer.color_control.vmin.value == ''
    assert explorer.color_control.vmax.value == ''
    assert 'user' not in mapper._cmin
    assert 'user' not in mapper._cmax
    explorer.color_control.log_button.value = False
    assert mapper.norm == 'linear'


@pytest.mark.parametrize('field', ['vmin', 'vmax'])
@pytest.mark.parametrize('bad', ['0', '-1'])
def test_nonpositive_bounds_in_log_mode_leave_range_unchanged(field, bad):
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    explorer.color_control.log_button.value = True
    explorer.color_control.vmin.value = '2'
    explorer.color_control.vmax.value = '9'
    mapper = explorer.tab_figs['bank_1'].view.colormapper

    getattr(explorer.color_control, field).value = bad
    assert (mapper.cmin, mapper.cmax) == (2, 9)


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


def test_fit_and_tab_switching_with_masked_data():
    explorer = DetectorExplorer(make_banks({'y': 2, 'x': 2}, count=2))
    explorer.masks['bank_0'].values[...] = True
    explorer.mask_selection_nodes['bank_0'].notify_children(message=None)
    explorer.all_mask_selection_nodes['bank_0'].notify_children(message=None)
    explorer.color_control.fit_button.click()
    select(explorer, 'bank_0')
    explorer.color_control.fit_button.click()
    select(explorer, 'All')
