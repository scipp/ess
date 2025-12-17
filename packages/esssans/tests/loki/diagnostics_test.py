# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import matplotlib
import pytest
import scipp as sc

import ess.loki.data  # noqa: F401
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
    wf[Filename[SampleRun]] = loki.data.loki_coda_file_small()

    data = sc.DataGroup()
    for i in range(9):
        key = f"loki_detector_{i}"
        wf[NeXusDetectorName] = key
        data[key] = wf.compute(RawDetector[SampleRun])

    return data


def test_create_loki_bank_viewer(loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(loki_data.hist())
    assert len(viewer.figs) == 9


def test_loki_bank_viewer_plotting_args(loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(loki_data.hist(), norm='log', cmap='jet')
    for fig in viewer.figs:
        mapper = fig.view.colormapper
        assert mapper.norm == 'log'
        assert mapper.cmap.name == 'jet'


def test_loki_bank_viewer_toggle_log_scale(loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(loki_data.hist())
    for fig in viewer.figs:
        mapper = fig.view.colormapper
        assert mapper.norm == 'linear'
    viewer.log_button.value = True
    for fig in viewer.figs:
        mapper = fig.view.colormapper
        assert mapper.norm == 'log'


def test_loki_bank_viewer_sum_all_layers(loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(loki_data.hist())
    old_max = [fig.view.colormapper.vmax for fig in viewer.figs]
    viewer.layer_sum.value = True
    for i, fig in enumerate(viewer.figs):
        assert fig.view.colormapper.vmax >= old_max[i]


def test_loki_bank_viewer_sum_all_straws(loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = LokiBankViewer(loki_data.hist())
    old_max = [fig.view.colormapper.vmax for fig in viewer.figs]
    viewer.straw_sum.value = True
    for i, fig in enumerate(viewer.figs):
        assert fig.view.colormapper.vmax >= old_max[i]


def test_creat_loki_instrument_view(loki_data):
    InstrumentView(loki_data.hist())


def test_creat_loki_instrument_view_with_dim_slider(loki_data):
    InstrumentView(loki_data.hist(event_time_offset=200), dim='event_time_offset')
