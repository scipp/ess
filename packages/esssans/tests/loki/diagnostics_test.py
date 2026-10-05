# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import ess.loki.data  # noqa: F401
import matplotlib
import pytest
import scipp as sc
import scippneutron as scn
from ess import loki
from ess.loki import diagnostics
from ess.loki.diagnostics import LokiBankViewer
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


def test_diagnostics_available_at_top_level():
    assert loki.LokiBankViewer is LokiBankViewer
    assert loki.instrument_view is scn.instrument_view
    assert loki.InstrumentView is diagnostics.InstrumentView
    assert {'LokiBankViewer', 'instrument_view', 'InstrumentView'} <= set(loki.__all__)


@pytest.mark.parametrize('use_positional_args', [False, True])
def test_deprecated_instrument_view_forwards_arguments(
    monkeypatch, use_positional_args
):
    data = sc.DataGroup()
    pixel_size = sc.scalar(2.0, unit='cm')
    result = object()
    calls = []

    def instrument_view(data, **kwargs):
        calls.append((data, kwargs))
        return result

    monkeypatch.setattr(diagnostics, 'instrument_view', instrument_view)
    if use_positional_args:
        entry_point = loki.InstrumentView
        args = (data, 'tof', pixel_size)
        kwargs = {'cmap': 'jet'}
    else:
        entry_point = diagnostics.InstrumentView
        args = (data,)
        kwargs = {'dim': 'tof', 'pixel_size': pixel_size, 'cmap': 'jet'}
    with pytest.warns(DeprecationWarning, match='use ess.loki.instrument_view') as w:
        viewer = entry_point(*args, **kwargs)

    assert viewer is result
    assert len(calls) == 1
    assert calls[0][0] is data
    assert calls[0][1] == {'dim': 'tof', 'pixel_size': pixel_size, 'cmap': 'jet'}
    assert w[0].filename == __file__


@pytest.mark.parametrize('dim', [None, 'tof'])
def test_deprecated_instrument_view_creates_figure(dim):
    data = sc.DataArray(
        sc.ones(dims=['pixel', 'tof'], shape=[2, 3], unit='counts'),
        coords={
            'position': sc.vectors(
                dims=['pixel'], values=[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], unit='m'
            ),
            'tof': sc.arange('tof', 4.0, unit='us'),
        },
    )
    with pytest.warns(DeprecationWarning, match='use ess.loki.instrument_view'):
        fig = loki.InstrumentView(data, dim=dim)
    assert len(fig.artists) == 1


def test_create_loki_bank_viewer(histogrammed_loki_data):
    matplotlib.use('module://ipympl.backend_nbagg')
    viewer = loki.LokiBankViewer(histogrammed_loki_data)
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
