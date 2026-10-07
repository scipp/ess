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
def test_load_detectors_one_bank(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="depends_on chain"):
            dg = loki.load_detectors(file, banks=["loki_detector_0"])
    else:
        dg = loki.load_detectors(file, banks=["loki_detector_0"])
    assert set(dg.keys()) == {"loki_detector_0"}


@pytest.mark.parametrize("file", FILES)
def test_load_detectors_one_bank_from_str(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="depends_on chain"):
            dg = loki.load_detectors(file, banks="loki_detector_0")
    else:
        dg = loki.load_detectors(file, banks="loki_detector_0")
    assert set(dg.keys()) == {"loki_detector_0"}


@pytest.mark.parametrize("file", FILES)
def test_load_detectors_two_banks(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="depends_on chain"):
            dg = loki.load_detectors(file, banks=["loki_detector_0", "loki_detector_4"])
    else:
        dg = loki.load_detectors(file, banks=["loki_detector_0", "loki_detector_4"])
    assert set(dg.keys()) == {"loki_detector_0", "loki_detector_4"}


@pytest.mark.parametrize("file", FILES)
def test_load_monitors_all_monitors(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match=r"(?:Falling back|depends_on chain)"):
            mons = loki.load_monitors(file)
    else:
        mons = loki.load_monitors(file)
    assert set(mons.keys()) == {f"beam_monitor_m{i}" for i in range(5)}


@pytest.mark.parametrize("file", FILES)
def test_load_monitors_one_monitor(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="Falling back"):
            mons = loki.load_monitors(file, monitors=["beam_monitor_m0"])
    else:
        mons = loki.load_monitors(file, monitors=["beam_monitor_m0"])
    assert set(mons.keys()) == {"beam_monitor_m0"}


@pytest.mark.parametrize("file", FILES)
def test_load_monitors_two_monitors(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="Falling back"):
            mons = loki.load_monitors(
                file, monitors=["beam_monitor_m1", "beam_monitor_m3"]
            )
    else:
        mons = loki.load_monitors(file, monitors=["beam_monitor_m1", "beam_monitor_m3"])
    assert set(mons.keys()) == {"beam_monitor_m1", "beam_monitor_m3"}


@pytest.mark.parametrize("file", FILES)
def test_load_monitors_one_monitor_from_str(file):

    if file == FILES[1]:
        with pytest.warns(UserWarning, match="Falling back"):
            mons = loki.load_monitors(file, monitors="beam_monitor_m1")
    else:
        mons = loki.load_monitors(file, monitors="beam_monitor_m1")
    assert set(mons.keys()) == {"beam_monitor_m1"}
