# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
from ess import dream
from ess.dream import data

TEST_BANK_SIZES = {
    "endcap_backward_detector": {
        "strip": 1,
        "wire": 16,
        "module": 11,
        "segment": 28,
        "counter": 2,
    },
    "endcap_forward_detector": {
        "strip": 1,
        "wire": 16,
        "module": 5,
        "segment": 28,
        "counter": 2,
    },
    "mantle_detector": {
        "wire": 2,
        "module": 5,
        "segment": 6,
        "strip": 256,
        "counter": 2,
    },
    "high_resolution_detector": {"strip": 2, "other": -1},
    "sans_detector": {"strip": 2, "other": -1},
}


def test_load_detectors_all_banks():
    dg = dream.load_detectors(data.dream_coda_file(), fold=TEST_BANK_SIZES)
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
    dg = dream.load_detectors(
        data.dream_coda_file(), detectors=banks, fold=TEST_BANK_SIZES
    )
    assert set(dg.keys()) == set(banks)
    # Banks should have been re-shaped
    assert "detector_number" not in dg.dims


def test_load_monitors_all_monitors():
    with pytest.warns(UserWarning, match="Failed to load"):
        mons = dream.load_monitors(data.dream_coda_file())
    assert set(mons.keys()) == {"monitor_bunker", "monitor_cave"}


@pytest.mark.parametrize(
    "monitors",
    [["monitor_bunker"], ["monitor_bunker", "monitor_cave"]],
)
def test_load_monitors_selected_monitors(monitors):
    with pytest.warns(UserWarning, match="Failed to load"):
        mons = dream.load_monitors(data.dream_coda_file(), monitors=monitors)
    assert set(mons.keys()) == set(monitors)
