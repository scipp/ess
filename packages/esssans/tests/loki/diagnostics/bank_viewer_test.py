# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import ess.loki.data  # noqa: F401
import matplotlib
import pytest
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
    for fig in viewer.subplots:
        mapper = fig.view.colormapper
        assert mapper.norm == 'log'
        assert mapper.cmap.name == 'jet'


def test_loki_bank_viewer_toggle_log_scale(histogrammed_loki_data):
    viewer = LokiBankViewer(histogrammed_loki_data)
    for fig in viewer.subplots:
        mapper = fig.view.colormapper
        assert mapper.norm == 'linear'
    viewer.log_button.value = True
    for fig in viewer.subplots:
        mapper = fig.view.colormapper
        assert mapper.norm == 'log'


def test_loki_bank_viewer_sum_all_layers(histogrammed_loki_data):
    viewer = LokiBankViewer(histogrammed_loki_data)
    old_max = [fig.view.colormapper.vmax for fig in viewer.subplots]
    viewer.layer_sum.value = True
    for i, fig in enumerate(viewer.subplots):
        assert fig.view.colormapper.vmax >= old_max[i]


def test_loki_bank_viewer_sum_all_straws(histogrammed_loki_data):
    viewer = LokiBankViewer(histogrammed_loki_data)
    old_max = [fig.view.colormapper.vmax for fig in viewer.subplots]
    viewer.straw_sum.value = True
    for i, fig in enumerate(viewer.subplots):
        assert fig.view.colormapper.vmax >= old_max[i]


def test_loki_bank_viewer_change_bank(histogrammed_loki_data):
    viewer = LokiBankViewer(histogrammed_loki_data)
    # For now, just check no error occurs when changing tab
    viewer.tabs.selected_index = 2
    # Change back to all banks
    viewer.tabs.selected_index = 0
