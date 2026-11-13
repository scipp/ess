# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import math

import matplotlib.pyplot as plt
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


def test_single_bank_clear_uses_bank_slider_slice():
    explorer = DetectorExplorer(make_banks(count=1))
    explorer.bank_dim_sliders['bank_0'][0].slider.value = 1
    explorer.masks['bank_0']['new'] = sc.ones(
        sizes=explorer.data['bank_0'].sizes, dtype=bool
    )

    explorer.mask_control.clear_button.click()

    assert explorer.masks['bank_0']['new']['layer', 0].values.all()
    assert not explorer.masks['bank_0']['new']['layer', 1].values.any()


def test_single_bank_color_controls_update_bank_figure():
    explorer = DetectorExplorer(make_banks(count=1))

    explorer.color_control.vmin.value = '1'
    explorer.color_control.vmax.value = '10'

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
    assert not explorer.bank_dim_sliders['bank_0'][0].sum.disabled
    assert explorer.bank_dim_sliders['bank_1'][0].slider.disabled
    assert explorer.bank_dim_sliders['bank_1'][0].sum.disabled
    assert not explorer.bank_dim_sliders['bank_2'][0].slider.disabled
    assert not explorer.bank_dim_sliders['bank_2'][0].sum.disabled

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
# 2. Slicing and summing
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


@pytest.mark.parametrize('view', ['bank', 'all'])
def test_sum_displays_reduction_and_disabling_restores_slice(view):
    explorer = DetectorExplorer(make_banks())
    name = 'bank_1' if view == 'bank' else 'All'
    nodes = explorer.to_plot_nodes if view == 'bank' else (explorer.all_to_plot_nodes)
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
    assert_identical(explorer.to_plot_nodes['bank_2']().data, expected.data)

    sliders['time'].sum.value = False
    sliders['time'].slider.value = 2
    sliders['layer'].sum.value = True
    expected = explorer.data['bank_2']['time', 2].sum('layer')
    assert_identical(explorer.to_plot_nodes['bank_2']().data, expected.data)


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
    explorer.masks['bank_1']['mask'] = sc.zeros(sizes=explorer.data.sizes, dtype=bool)
    explorer.masks['bank_1']['mask']['layer', 0]['y', 1]['x', 2].value = True
    explorer.apply_masks_nodes['bank_1'].notify_children(message=None)
    select(explorer, end)

    for nodes in (explorer.to_plot_nodes, explorer.all_to_plot_nodes):
        result = nodes['bank_1']()
        expected = expected_display(explorer, 'bank_1', layer=0)
        assert_identical(result, expected)
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
    keys = list(explorer.masks['bank_1'].keys())
    assert len(keys) == 1
    assert explorer.masks['bank_1'][keys[0]].values.sum() == 1


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
    assert not explorer.masks['bank_1']


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
    assert not any(m for m in explorer.masks.values())


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
    assert not explorer.masks['bank_0']


@pytest.mark.parametrize('mode', MODES)
def test_mode_toggles_exactly_the_intended_elements(mode):
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')
    explorer.mask_control.mode = mode
    press(explorer, 'bank_0', 2, 1)

    expected = mask_at(explorer.data['bank_0'].sizes, mode, 0, 2, 1)
    assert_identical(next(iter(explorer.masks['bank_0'].values())), expected)


@pytest.mark.parametrize(('first', 'second'), ALL_TRANSITIONS)
def test_mode_transitions_use_new_mode_even_on_same_pixel(first, second):
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_0')

    explorer.mask_control.mode = first
    press(explorer, 'bank_0', 2, 1)
    expected_first = mask_at(explorer.data['bank_0'].sizes, first, 0, 2, 1)
    keys = list(explorer.masks['bank_0'].keys())
    assert_identical(explorer.masks['bank_0'][keys[0]], expected_first)

    explorer.mask_control.mode = second
    press(explorer, 'bank_0', 2, 1)
    expected_second = mask_at(explorer.data['bank_0'].sizes, second, 0, 2, 1)
    keys = list(explorer.masks['bank_0'].keys())
    assert len(keys) == 2
    assert_identical(explorer.masks['bank_0'][keys[1]], expected_second)
    assert (
        explorer.masks['bank_0'][keys[0]].dims != explorer.masks['bank_0'][keys[1]].dims
    )

    press(explorer, 'bank_0', 2, 1)
    # The second mask should have disappeared
    assert len(explorer.masks['bank_0']) == 1


def test_summed_dimension_applies_edits_across_full_extent():
    explorer = DetectorExplorer(
        make_banks({'time': 3, 'layer': 2, 'y': 3, 'x': 4}), show_dims=['y', 'x']
    )
    select(explorer, 'bank_0')
    sliders = sliders_of(explorer, 'bank_0')
    sliders['time'].sum.value = True
    sliders['layer'].slider.value = 1
    explorer.mask_control.mode = 'vertical'
    press(explorer, 'bank_0', 3, 0)

    mask = next(iter(explorer.masks['bank_0'].values()))
    assert set(mask.dims) == {'x', 'layer'}


def test_masking_affects_only_selected_bank_and_leaves_data_unchanged():
    data = make_banks()
    original = data.copy(deep=True)
    explorer = DetectorExplorer(data)
    select(explorer, 'bank_1')
    explorer.mask_control.mode = 'vertical'
    press(explorer, 'bank_1', 1, 1)

    assert explorer.masks['bank_1']
    for bank in ('bank_0', 'bank_2'):
        assert not explorer.masks[bank]
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
    expected = mask_at(explorer.data['bank_0'].sizes, 'single', 0, ix, iy)
    assert_identical(next(iter(explorer.masks['bank_0'].values())), expected)


@pytest.mark.parametrize('mode', MODES)
def test_reversed_visible_dimensions_target_correct_pixels(mode):
    explorer = DetectorExplorer(make_banks(), show_dims=['x', 'y'])
    select(explorer, 'bank_0')
    explorer.mask_control.mode = mode
    # The horizontal axis now shows 'y' (size 3), the vertical axis 'x' (size 4).
    press(explorer, 'bank_0', 2, 3)
    sizes = explorer.data['bank_0'].sizes

    if mode == 'single':
        expected = sc.zeros(sizes=sizes, dtype=bool)
        expected['layer', 0]['x', 3]['y', 2].value = True
    elif mode == 'horizontal':
        sizes = sizes.copy()
        del sizes['y']
        expected = sc.zeros(sizes=sizes, dtype=bool)
        expected['layer', 0]['x', 3].value = True
    elif mode == 'vertical':
        sizes = sizes.copy()
        del sizes['x']
        expected = sc.zeros(sizes=sizes, dtype=bool)
        expected['layer', 0]['y', 2].value = True
    assert_identical(next(iter(explorer.masks['bank_0'].values())), expected)


def test_adding_a_mask_then_clicking_same_place_again_deletes_it():
    explorer = DetectorExplorer(make_banks(count=1))
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'horizontal'
    press(explorer, 'bank_0', 0, 1)
    assert explorer.masks['bank_0']  # Ensure the mask was added
    press(explorer, 'bank_0', 0, 1)
    assert not explorer.masks['bank_0']  # Ensure the mask was deleted (not zeroed)


def test_deleting_mask_by_click_does_not_remove_from_dict_if_other_parts_are_masked():
    explorer = DetectorExplorer(make_banks(count=1))
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'horizontal'
    press(explorer, 'bank_0', 1, 1)
    assert len(explorer.masks['bank_0']) == 1  # Ensure the mask was added
    (key,) = explorer.masks['bank_0'].keys()
    n = explorer.masks['bank_0'][key].sum().value
    sliders_of(explorer, 'bank_0')['layer'].slider.value = 1
    press(explorer, 'bank_0', 0, 0)
    # We should still have just one mask, as the dimensions of the mask should be the
    # same, there should just be more values masked
    assert len(explorer.masks['bank_0']) == 1
    assert explorer.masks['bank_0'][key].sum().value > n
    # Go back to first layer
    sliders_of(explorer, 'bank_0')['layer'].slider.value = 0
    # Click to remove mask
    press(explorer, 'bank_0', 1, 1)
    assert key in explorer.masks['bank_0']  # Ensure the mask still exists
    # Ensure there are still masked values
    assert explorer.masks['bank_0'][key].sum().value > 0
    # Ensure the first layer has no masked values
    assert not explorer.masks['bank_0'][key]['layer', 0].values.max()


def test_add_mask_with_small_coord():
    dg = make_banks()
    dg['bank_0'].coords['x'] = sc.arange('x', 4) * 0.1
    explorer = DetectorExplorer(dg)
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'vertical'
    press(explorer, 'bank_0', 0.3, 1)
    mask = next(iter(explorer.masks['bank_0'].values()))
    assert sc.identical(
        mask['layer', 0],
        sc.array(dims=['x'], values=[False, False, False, True], dtype=bool),
    )


def test_add_mask_with_small_coord_and_bin_edges():
    dg = make_banks()
    dg['bank_0'].coords['x'] = sc.arange('x', 5) * 0.1
    explorer = DetectorExplorer(dg)
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'vertical'
    press(explorer, 'bank_0', 0.201, 1)  # Just above the 3rd bin edge
    mask = next(iter(explorer.masks['bank_0'].values()))
    assert sc.identical(
        mask['layer', 0],
        sc.array(dims=['x'], values=[False, False, True, False], dtype=bool),
    )
    press(explorer, 'bank_0', 0.198, 1)  # Just below the 3rd bin edge
    assert sc.identical(
        mask['layer', 0],
        sc.array(dims=['x'], values=[False, True, True, False], dtype=bool),
    )


def test_add_mask_with_large_coord():
    dg = make_banks()
    dg['bank_0'].coords['x'] = sc.arange('x', 1000, 1004)
    explorer = DetectorExplorer(dg)
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'vertical'
    press(explorer, 'bank_0', 1001, 1)
    mask = next(iter(explorer.masks['bank_0'].values()))
    assert sc.identical(
        mask['layer', 0],
        sc.array(dims=['x'], values=[False, True, False, False], dtype=bool),
    )


def test_add_mask_with_large_coord_and_bin_edges():
    dg = make_banks()
    dg['bank_0'].coords['x'] = sc.arange('x', 1000, 1005)
    explorer = DetectorExplorer(dg)
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'vertical'
    press(explorer, 'bank_0', 1001.1, 1)
    mask = next(iter(explorer.masks['bank_0'].values()))
    assert sc.identical(
        mask['layer', 0],
        sc.array(dims=['x'], values=[False, True, False, False], dtype=bool),
    )
    press(explorer, 'bank_0', 1000.9, 1)  # Just below the 2nd bin edge
    assert sc.identical(
        mask['layer', 0],
        sc.array(dims=['x'], values=[True, True, False, False], dtype=bool),
    )


# ---------------------------------------------------------------------------
# 7. Clearing and saving
# ---------------------------------------------------------------------------


def test_clear_on_bank_affects_only_that_banks_selected_slice():
    explorer = DetectorExplorer(make_banks())
    select(explorer, 'bank_1')
    explorer.mask_control.mode = 'horizontal'
    press(explorer, 'bank_1', 1, 1)
    explorer.bank_dim_sliders['bank_1'][0].slider.value = 1
    press(explorer, 'bank_1', 1, 1)

    explorer.mask_control.clear_button.click()

    (key,) = explorer.masks['bank_1'].keys()

    assert explorer.masks['bank_1'][key]['layer', 1].values.sum() == 0
    assert explorer.masks['bank_1'][key]['layer', 0].values.sum() == 1


def test_clear_on_all_affects_selected_slice_in_every_bank():
    explorer = DetectorExplorer(make_banks())
    explorer.mask_control.mode = 'horizontal'
    select(explorer, 'bank_1')
    press(explorer, 'bank_1', 1, 1)
    select(explorer, 'bank_2')
    press(explorer, 'bank_2', 1, 0)
    assert len(explorer.masks['bank_1']) == 1
    assert len(explorer.masks['bank_2']) == 1

    select(explorer, 'All')
    explorer.mask_control.clear_button.click()
    assert len(explorer.masks['bank_1']) == 0
    assert len(explorer.masks['bank_2']) == 0


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
    explorer.mask_control.mode = 'horizontal'
    select(explorer, 'bank_2')
    press(explorer, 'bank_2', 1, 0)
    select(explorer, 'bank_1')
    press(explorer, 'bank_1', 1, 1)

    path = tmp_path / 'masks.h5'
    explorer.mask_control.filename_input.value = str(path)
    explorer.mask_control.save_button.click()

    loaded = sc.io.load_hdf5(path)
    for bank, masks in explorer.masks.items():
        for k, m in masks.items():
            assert_identical(loaded[bank][k], m)


def test_clear_masks_removes_from_dict_if_all_false():
    explorer = DetectorExplorer(make_banks(count=1))
    select(explorer, 'bank_0')
    explorer.mask_control.mode = 'horizontal'
    press(explorer, 'bank_0', 0, 1)
    assert explorer.masks['bank_0']  # Ensure the mask was added
    explorer.mask_control.clear_button.click()
    assert not explorer.masks['bank_0']  # Ensure the mask was removed from the dict


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
