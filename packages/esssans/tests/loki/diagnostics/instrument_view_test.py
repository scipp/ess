# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
from ess import loki


@pytest.mark.parametrize('use_deprecated', [False, True])
def test_instrument_view_creates_figure(histogrammed_loki_data, use_deprecated):
    if use_deprecated:
        with pytest.warns(DeprecationWarning, match='use ess.loki.instrument_view'):
            fig = loki.InstrumentView(histogrammed_loki_data)
    else:
        fig = loki.instrument_view(histogrammed_loki_data)

    assert len(fig.artists) == 1


@pytest.mark.parametrize('use_deprecated', [False, True])
def test_instrument_view_with_slider_dim(loki_data, use_deprecated):
    data = loki_data.hist(
        event_time_offset=sc.linspace('event_time_offset', 0, 71, 6, unit='ms').to(
            unit='ns'
        )
    )
    if use_deprecated:
        with pytest.warns(DeprecationWarning, match='use ess.loki.instrument_view'):
            fig = loki.InstrumentView(data, dim='event_time_offset')
    else:
        fig = loki.instrument_view(data, dim='event_time_offset')

    assert len(fig.artists) == 1


@pytest.mark.parametrize('use_deprecated', [False, True])
def test_instrument_view_forwards_arguments(histogrammed_loki_data, use_deprecated):

    if use_deprecated:
        with pytest.warns(DeprecationWarning, match='use ess.loki.instrument_view'):
            fig = loki.InstrumentView(histogrammed_loki_data, cmap="jet")
    else:
        fig = loki.instrument_view(histogrammed_loki_data, cmap="jet")

    assert fig.view.colormapper.cmap.name == "jet"
