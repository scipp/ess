# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import ess.loki.data  # noqa: F401
import matplotlib
import pytest
import scipp as sc
from ess import loki
from ess.loki.diagnostics import LokiBankViewer


@pytest.fixture(scope='module', autouse=True)
def setup_matplotlib():
    matplotlib.use('module://ipympl.backend_nbagg')


def test_create_loki_bank_viewer(histogrammed_loki_data):
    viewer = loki.LokiBankViewer(histogrammed_loki_data)
    assert len(viewer.tabs.children) == 9 + 1  # 9 banks + all banks tab


def test_loki_bank_viewer_plotting_args(histogrammed_loki_data):
    viewer = LokiBankViewer(histogrammed_loki_data, norm='log', cmap='jet')
    for fig in viewer.subplots.values():
        mapper = fig.view.colormapper
        assert mapper.norm == 'log'
        assert mapper.cmap.name == 'jet'


def test_loki_bank_viewer_toggle_log_scale(histogrammed_loki_data):
    viewer = LokiBankViewer(histogrammed_loki_data)
    for fig in viewer.subplots.values():
        mapper = fig.view.colormapper
        assert mapper.norm == 'linear'
    viewer.color_control.log_button.value = True
    for fig in viewer.subplots.values():
        mapper = fig.view.colormapper
        assert mapper.norm == 'log'


@pytest.mark.parametrize("dims", [["layer"], ["straw"], ["layer", "straw"]])
def test_loki_bank_viewer_sum_other_dims(histogrammed_loki_data, dims):
    viewer = LokiBankViewer(histogrammed_loki_data.sum(dims))
    assert len(viewer.all_dim_sliders) == 2 - len(dims)


def test_loki_bank_viewer_change_bank(histogrammed_loki_data):
    viewer = LokiBankViewer(histogrammed_loki_data)
    # For now, just check no error occurs when changing tab
    viewer.tabs.selected_index = 2
    # Change back to all banks
    viewer.tabs.selected_index = 0


def test_loki_bank_viewer_tube_masks(histogrammed_loki_data):
    masks = {
        key: {
            "m": sc.zeros(
                sizes={dim: bank.sizes[dim] for dim in ('layer', 'tube')}, dtype=bool
            )
        }
        for key, bank in histogrammed_loki_data.items()
    }
    viewer = LokiBankViewer(histogrammed_loki_data, masks=masks)
    viewer.tabs.selected_index = 2
    viewer.mask_control.selector.value = 'm'
    # Simulate a click event to toggle the mask
    from matplotlib.backend_bases import MouseEvent

    ax = viewer.tab_figs[viewer.current_key].ax
    event = MouseEvent(name='button_press_event', canvas=ax.figure.canvas, x=250, y=6)
    event.inaxes = ax
    event.xdata = event.x
    event.ydata = event.y
    viewer.toggle_mask(event)
    # Check that the mask has been toggled
    assert viewer.masks["loki_detector_1"]["m"]["layer", 0]["tube", 6]
    assert not viewer.masks["loki_detector_1"]["m"]["layer", 0]["tube", 5]
    assert not viewer.masks["loki_detector_1"]["m"]["layer", 0]["tube", 7]
    assert not viewer.masks["loki_detector_1"]["m"]["layer", 1:]["tube", 6].all()
