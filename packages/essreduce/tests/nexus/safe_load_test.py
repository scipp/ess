# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest

from ess.reduce.nexus import load_detectors, load_monitors

# def test_load_detectors_all_banks_good_file(loki_coda_file):
#     dg = load_detectors(
#         loki_coda_file,
#     )
#     assert set(dg.keys()) == {f"loki_detector_{i}" for i in range(9)}
#     # Banks should NOT have been re-shaped
#     assert "detector_number" not in dg.dims


# def test_load_detectors_all_banks_bad_file(loki_broken_file):
#     with pytest.warns(UserWarning, match="depends_on chain"):
#         dg = load_detectors(file)
#     assert set(dg.keys()) == {f"loki_detector_{i}" for i in range(9)}
#     # Banks should have been re-shaped
#     assert "detector_number" not in dg.dims


def test_load_detectors_needs_banks_arg(loki_coda_file):
    with pytest.raises(TypeError, match="required positional argument"):
        load_detectors(loki_coda_file)


def test_load_detectors_one_bank_good_file(loki_coda_file):
    dg = load_detectors(loki_coda_file, banks=["loki_detector_0"])
    assert set(dg.keys()) == {"loki_detector_0"}
    # Data should NOT have been reshaped
    assert dg.dims == ("detector_number",)


def test_load_detectors_one_bank_bad_file(loki_broken_file):
    with pytest.warns(UserWarning, match="depends_on chain"):
        dg = load_detectors(loki_broken_file, banks=["loki_detector_0"])
    assert set(dg.keys()) == {"loki_detector_0"}
    # Data should NOT have been reshaped
    assert dg.dims == ("detector_number",)


def test_load_detectors_one_bank_good_file_from_str(loki_coda_file):
    dg = load_detectors(loki_coda_file, banks="loki_detector_0")
    assert set(dg.keys()) == {"loki_detector_0"}
    assert dg.dims == ("detector_number",)


def test_load_detectors_one_bank_bad_file_from_str(loki_broken_file):
    with pytest.warns(UserWarning, match="depends_on chain"):
        dg = load_detectors(loki_broken_file, banks="loki_detector_0")
    assert set(dg.keys()) == {"loki_detector_0"}
    assert dg.dims == ("detector_number",)


def test_load_detectors_two_banks_good_file(loki_coda_file):
    dg = load_detectors(loki_coda_file, banks=["loki_detector_0", "loki_detector_4"])
    assert set(dg.keys()) == {"loki_detector_0", "loki_detector_4"}
    # Data should NOT have been reshaped
    assert dg.dims == ("detector_number",)


def test_load_detectors_two_banks_bad_file(loki_broken_file):
    with pytest.warns(UserWarning, match="depends_on chain"):
        dg = load_detectors(
            loki_broken_file, banks=["loki_detector_0", "loki_detector_4"]
        )
    assert set(dg.keys()) == {"loki_detector_0", "loki_detector_4"}
    # Data should NOT have been reshaped
    assert dg.dims == ("detector_number",)


# def test_load_detectors_two_banks(file):
#     if file == FILES[1]:
#         with pytest.warns(UserWarning, match="depends_on chain"):
#             dg = load_detectors(file, banks=["loki_detector_0", "loki_detector_4"])
#     else:
#         dg = load_detectors(file, banks=["loki_detector_0", "loki_detector_4"])
#     assert set(dg.keys()) == {"loki_detector_0", "loki_detector_4"}


# def test_load_monitors_all_monitors(file):
#     if file == FILES[1]:
#         with pytest.warns(UserWarning, match=r"(?:Falling back|depends_on chain)"):
#             mons = load_monitors(file)
#     else:
#         mons = load_monitors(file)
#     assert set(mons.keys()) == {f"beam_monitor_m{i}" for i in range(5)}


# def test_load_monitors_one_monitor(file):
#     if file == FILES[1]:
#         with pytest.warns(UserWarning, match="Falling back"):
#             mons = load_monitors(file, monitors=["beam_monitor_m0"])
#     else:
#         mons = load_monitors(file, monitors=["beam_monitor_m0"])
#     assert set(mons.keys()) == {"beam_monitor_m0"}


# def test_load_monitors_two_monitors(file):
#     if file == FILES[1]:
#         with pytest.warns(UserWarning, match="Falling back"):
#             mons = load_monitors(file, monitors=["beam_monitor_m1", "beam_monitor_m3"])
#     else:
#         mons = load_monitors(file, monitors=["beam_monitor_m1", "beam_monitor_m3"])
#     assert set(mons.keys()) == {"beam_monitor_m1", "beam_monitor_m3"}


# def test_load_monitors_one_monitor_from_str(file):
#     if file == FILES[1]:
#         with pytest.warns(UserWarning, match="Falling back"):
#             mons = load_monitors(file, monitors="beam_monitor_m1")
#     else:
#         mons = load_monitors(file, monitors="beam_monitor_m1")
#     assert set(mons.keys()) == {"beam_monitor_m1"}
