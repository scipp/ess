# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
from ess import loki
from ess.loki import data

FILES = [data.loki_coda_file(), data.loki_broken_file()]


@pytest.mark.parametrize("file", FILES)
def test_load_detectors_all_banks(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="depends_on chain"):
            dg = loki.load_detectors(file)
    else:
        dg = loki.load_detectors(file)
    assert set(dg.keys()) == {f"loki_detector_{i}" for i in range(9)}
    # Banks should have been re-shaped
    assert "detector_number" not in dg.dims


@pytest.mark.parametrize("file", FILES)
@pytest.mark.parametrize(
    "banks",
    [["loki_detector_0"], ["loki_detector_0", "loki_detector_4"]],
)
def test_load_detectors_selected_banks(file, banks):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="depends_on chain"):
            dg = loki.load_detectors(file, detectors=banks)
    else:
        dg = loki.load_detectors(file, detectors=banks)
    assert set(dg.keys()) == set(banks)


@pytest.mark.parametrize("file", FILES)
def test_load_monitors_all_monitors(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match=r"(?:Falling back|depends_on chain)"):
            mons = loki.load_monitors(file)
    else:
        mons = loki.load_monitors(file)
    assert set(mons.keys()) == {f"beam_monitor_m{i}" for i in range(5)}


@pytest.mark.parametrize("file", FILES)
@pytest.mark.parametrize(
    "monitors",
    [["beam_monitor_m0"], ["beam_monitor_m1", "beam_monitor_m3"]],
)
def test_load_monitors_selected_monitor(file, monitors):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="Falling back"):
            mons = loki.load_monitors(file, monitors=monitors)
    else:
        mons = loki.load_monitors(file, monitors=monitors)
    assert set(mons.keys()) == set(monitors)
