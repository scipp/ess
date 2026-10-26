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
