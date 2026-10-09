# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
from ess import dream
from ess.dream import data


def test_load_detectors_all_banks():
    dg = dream.load_detectors(data.dream_coda_file())
    assert set(dg.keys()) == {
        "endcap_backward_detector",
        "endcap_forward_detector",
        "mantle_detector",
        "high_resolution_detector",
        "sans_detector",
    }
    # Banks should have been re-shaped
    assert "detector_number" not in dg.dims


@pytest.mark.parametrize(
    "banks",
    [["mantle_detector"], ["endcap_backward_detector"], ["endcap_forward_detector"]],
)
def test_load_detectors_selected_banks(banks):
    dg = dream.load_detectors(data.dream_coda_file(), detectors=banks)
    assert set(dg.keys()) == set(banks)
    # Banks should have been re-shaped
    assert "detector_number" not in dg.dims


def test_load_monitors_all_monitors():
    mons = dream.load_monitors(data.dream_coda_file())
    assert set(mons.keys()) == {"monitor_bunker", "monitor_cave"}


@pytest.mark.parametrize(
    "monitors",
    [["monitor_bunker"], ["monitor_bunker", "monitor_cave"]],
)
def test_load_monitors_selected_monitors(monitors):
    mons = dream.load_monitors(data.dream_coda_file(), monitors=monitors)
    assert set(mons.keys()) == set(monitors)
