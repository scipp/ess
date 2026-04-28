# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import ess.loki.data  # noqa: F401
import matplotlib
import pytest
import scipp as sc
from ess import loki
from ess.loki.diagnostics import InstrumentView, LokiBankViewer
from ess.sans.types import (
    BeamCenter,
    Filename,
    NeXusDetectorName,
    RawDetector,
    SampleRun,
)


@pytest.fixture(scope='module')
def loki_data():
    wf = loki.LokiWorkflow()
    wf[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    wf[Filename[SampleRun]] = loki.data.loki_coda_file()

    data = sc.DataGroup()
    for i in range(9):
        key = f"loki_detector_{i}"
        wf[NeXusDetectorName] = key
        data[key] = wf.compute(RawDetector[SampleRun])

    return data


@pytest.fixture(scope='module')
def histogrammed_loki_data(loki_data):
    return loki_data.hist()


def test_create_loki_bank_viewer(histogrammed_loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(histogrammed_loki_data)
    assert len(viewer.tabs.children) == 9 + 1  # 9 banks + all banks tab


def test_loki_bank_viewer_plotting_args(histogrammed_loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(histogrammed_loki_data, norm='log', cmap='jet')
    for fig in viewer.subplots:
        mapper = fig.view.colormapper
        assert mapper.norm == 'log'
        assert mapper.cmap.name == 'jet'


def test_loki_bank_viewer_toggle_log_scale(histogrammed_loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(histogrammed_loki_data)
    for fig in viewer.subplots:
        mapper = fig.view.colormapper
        assert mapper.norm == 'linear'
    viewer.log_button.value = True
    for fig in viewer.subplots:
        mapper = fig.view.colormapper
        assert mapper.norm == 'log'


def test_loki_bank_viewer_sum_all_layers(histogrammed_loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(histogrammed_loki_data)
    old_max = [fig.view.colormapper.vmax for fig in viewer.subplots]
    viewer.layer_sum.value = True
    for i, fig in enumerate(viewer.subplots):
        assert fig.view.colormapper.vmax >= old_max[i]


def test_loki_bank_viewer_sum_all_straws(histogrammed_loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(histogrammed_loki_data)
    old_max = [fig.view.colormapper.vmax for fig in viewer.subplots]
    viewer.straw_sum.value = True
    for i, fig in enumerate(viewer.subplots):
        assert fig.view.colormapper.vmax >= old_max[i]


def test_loki_bank_viewer_change_bank(histogrammed_loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(histogrammed_loki_data)
    # For now, just check no error occurs when changing tab
    viewer.tabs.selected_index = 2
    # Change back to all banks
    viewer.tabs.selected_index = 0


def test_creat_loki_instrument_view(histogrammed_loki_data):
    InstrumentView(histogrammed_loki_data)


def test_creat_loki_instrument_view_with_dim_slider(loki_data):
    InstrumentView(loki_data.hist(event_time_offset=10), dim='event_time_offset')
