# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest

from ess.reduce.nexus import load_detectors, load_monitors


def test_load_detectors_all_banks(loki_coda_file):
    dg = load_detectors(loki_coda_file)
    assert set(dg.keys()) == {f"loki_detector_{i}" for i in range(9)}


@pytest.mark.parametrize("bank_name", [["loki_detector_0"], "loki_detector_0"])
def test_load_detectors_one_bank_good_file(loki_coda_file, bank_name):
    dg = load_detectors(loki_coda_file, detectors=bank_name)
    assert set(dg.keys()) == {"loki_detector_0"}
    # Data should NOT have been reshaped
    assert dg.dims == ("detector_number",)


@pytest.mark.parametrize("bank_name", [["loki_detector_0"], "loki_detector_0"])
def test_load_detectors_one_bank_bad_file(loki_broken_file, bank_name):
    with pytest.warns(UserWarning, match="depends_on chain"):
        dg = load_detectors(loki_broken_file, detectors=bank_name)
    assert set(dg.keys()) == {"loki_detector_0"}
    # Data should NOT have been reshaped
    assert dg.dims == ("detector_number",)


def test_load_detectors_two_banks_good_file(loki_coda_file):
    dg = load_detectors(
        loki_coda_file, detectors=["loki_detector_0", "loki_detector_4"]
    )
    assert set(dg.keys()) == {"loki_detector_0", "loki_detector_4"}
    # Data should NOT have been reshaped
    assert dg.dims == ("detector_number",)


def test_load_detectors_two_banks_bad_file(loki_broken_file):
    with pytest.warns(UserWarning, match="depends_on chain"):
        dg = load_detectors(
            loki_broken_file, detectors=["loki_detector_0", "loki_detector_4"]
        )
    assert set(dg.keys()) == {"loki_detector_0", "loki_detector_4"}
    # Data should NOT have been reshaped
    assert dg.dims == ("detector_number",)


def test_load_detectors_one_bank_with_fold(loki_coda_file):
    fold = {'loki_detector_0': {'layer': 4, 'tube': 56, 'straw': 7, 'pixel': -1}}
    dg = load_detectors(loki_coda_file, detectors=["loki_detector_0"], fold=fold)
    assert set(dg.keys()) == {"loki_detector_0"}
    # Data should have been reshaped according to the fold
    assert set(dg.dims) == set(fold['loki_detector_0'].keys())


def test_load_detectors_two_banks_with_fold(loki_coda_file):
    fold = {
        'loki_detector_1': {'layer': 4, 'tube': 16, 'straw': 7, 'pixel': -1},
        'loki_detector_3': {'layer': 4, 'tube': 16, 'straw': 7, 'pixel': -1},
    }
    dg = load_detectors(
        loki_coda_file, detectors=["loki_detector_1", "loki_detector_3"], fold=fold
    )
    assert set(dg.keys()) == {"loki_detector_1", "loki_detector_3"}
    # Data should have been reshaped according to the fold
    for bank in ["loki_detector_1", "loki_detector_3"]:
        assert set(dg[bank].dims) == set(fold[bank].keys())


def test_load_monitors_all_monitors(loki_coda_file):
    mons = load_monitors(loki_coda_file)
    assert set(mons.keys()) == {f"beam_monitor_m{i}" for i in range(5)}


@pytest.mark.parametrize("monitor_name", [["beam_monitor_m0"], "beam_monitor_m0"])
def test_load_monitors_one_monitor_good_file(loki_coda_file, monitor_name):
    mons = load_monitors(loki_coda_file, monitors=monitor_name)
    assert set(mons.keys()) == {"beam_monitor_m0"}


@pytest.mark.parametrize("monitor_name", [["beam_monitor_m0"], "beam_monitor_m0"])
def test_load_monitors_one_monitor_bad_file(loki_broken_file, monitor_name):
    with pytest.warns(UserWarning, match="Falling back"):
        mons = load_monitors(loki_broken_file, monitors=monitor_name)
    assert set(mons.keys()) == {"beam_monitor_m0"}


def test_load_monitors_two_monitors_good_file(loki_coda_file):
    mons = load_monitors(
        loki_coda_file, monitors=["beam_monitor_m1", "beam_monitor_m3"]
    )
    assert set(mons.keys()) == {"beam_monitor_m1", "beam_monitor_m3"}


def test_load_monitors_two_monitors_bad_file(loki_broken_file):
    with pytest.warns(UserWarning, match="Falling back"):
        mons = load_monitors(
            loki_broken_file, monitors=["beam_monitor_m1", "beam_monitor_m3"]
        )
    assert set(mons.keys()) == {"beam_monitor_m1", "beam_monitor_m3"}
