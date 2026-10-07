# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from ess import loki
from ess.loki import data


def test_load_detectors_all_banks():
    file = data.loki_coda_file()

    dg = loki.load_detectors(file)
    assert set(dg.keys()) == {f"loki_detector_{i}" for i in range(9)}
    # Banks should have been re-shaped
    assert "detector_number" not in dg.dims


def test_load_detectors_one_bank():
    file = data.loki_coda_file()

    dg = loki.load_detectors(file, banks=["loki_detector_0"])
    assert set(dg.keys()) == {"loki_detector_0"}


def test_load_detectors_one_bank_from_str():
    file = data.loki_coda_file()

    dg = loki.load_detectors(file, banks="loki_detector_2")
    assert set(dg.keys()) == {"loki_detector_2"}


def test_load_detectors_two_banks():
    file = data.loki_coda_file()

    dg = loki.load_detectors(file, banks=["loki_detector_1", "loki_detector_4"])
    assert set(dg.keys()) == {"loki_detector_1", "loki_detector_4"}


def test_load_monitors_all_monitors():
    file = data.loki_coda_file()

    mons = loki.load_monitors(file)
    assert set(mons.keys()) == {f"beam_monitor_m{i}" for i in range(5)}


def test_load_monitors_one_monitor():
    file = data.loki_coda_file()

    mons = loki.load_monitors(file, monitors=["beam_monitor_m0"])
    assert set(mons.keys()) == {"beam_monitor_m0"}


def test_load_monitors_two_monitors():
    file = data.loki_coda_file()

    mons = loki.load_monitors(file, monitors=["beam_monitor_m1", "beam_monitor_m3"])
    assert set(mons.keys()) == {"beam_monitor_m1", "beam_monitor_m3"}


def test_load_monitors_one_monitor_from_str():
    file = data.loki_coda_file()

    mons = loki.load_monitors(file, monitors="beam_monitor_m2")
    assert set(mons.keys()) == {"beam_monitor_m2"}
