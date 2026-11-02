# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
from ess.loki.diagnostics import load_detectors, load_monitors


def test_load_detectors_all_banks_good_file(loki_coda_file):

    dg = load_detectors(loki_coda_file)
    assert set(dg.keys()) == {f"loki_detector_{i}" for i in range(9)}
    # Banks should have been re-shaped
    assert "detector_number" not in dg.dims
    assert dg.ndim > 0


def test_load_detectors_all_banks_bad_file(loki_broken_file):

    with pytest.warns(UserWarning):  # noqa: PT030
        dg = load_detectors(loki_broken_file)
    assert set(dg.keys()) == {f"loki_detector_{i}" for i in range(9)}
    # Banks should have been re-shaped
    assert "detector_number" not in dg.dims
    assert dg.ndim > 0


@pytest.mark.parametrize(
    "banks",
    [["loki_detector_0"], ["loki_detector_0", "loki_detector_4"]],
)
def test_load_detectors_selected_banks_good_file(loki_coda_file, banks):

    dg = load_detectors(loki_coda_file, detectors=banks)
    assert set(dg.keys()) == set(banks)
    assert dg.ndim > 0


@pytest.mark.parametrize(
    "banks",
    [["loki_detector_0"], ["loki_detector_0", "loki_detector_4"]],
)
def test_load_detectors_selected_banks_bad_file(loki_broken_file, banks):

    with pytest.warns(UserWarning):  # noqa: PT030
        dg = load_detectors(loki_broken_file, detectors=banks)
    assert set(dg.keys()) == set(banks)
    assert dg.ndim > 0


def test_load_monitors_all_monitors_good_file(loki_coda_file):
    mons = load_monitors(loki_coda_file)
    assert set(mons.keys()) == {f"beam_monitor_m{i}" for i in range(5)}


def test_load_monitors_all_monitors_bad_file(loki_broken_file):

    with pytest.warns(UserWarning):  # noqa: PT030
        mons = load_monitors(loki_broken_file)
    assert set(mons.keys()) == {f"beam_monitor_m{i}" for i in range(5)}


@pytest.mark.parametrize(
    "monitors",
    [["beam_monitor_m4"], ["beam_monitor_m1", "beam_monitor_m4"]],
)
def test_load_monitors_selected_monitor_good_file(loki_coda_file, monitors):

    mons = load_monitors(loki_coda_file, monitors=monitors)
    assert set(mons.keys()) == set(monitors)


@pytest.mark.parametrize(
    "monitors",
    [["beam_monitor_m4"], ["beam_monitor_m1", "beam_monitor_m4"]],
)
def test_load_monitors_selected_monitor_bad_file(loki_broken_file, monitors):

    with pytest.warns(UserWarning):  # noqa: PT030
        mons = load_monitors(loki_broken_file, monitors=monitors)
    assert set(mons.keys()) == set(monitors)
