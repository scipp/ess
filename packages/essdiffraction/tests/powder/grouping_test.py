# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from ess.powder.grouping import (
    focus_data_dspacing_and_two_theta,
    group_two_theta,
    integrate_two_theta,
)
from ess.powder.types import KeepEvents, SampleRun

DSPACING_BINS = sc.linspace('dspacing', 1.0, 2.0, num=11, unit='angstrom')
"""Coarse d-spacing bins; the tests are only concerned with two-theta."""


@pytest.fixture
def detector() -> sc.DataArray:
    """Events spread smoothly over the full two-theta range of a wide detector."""
    rng = np.random.default_rng(90210)
    n_pixel = 5000
    n_event = 20
    two_theta = sc.array(
        dims=['pixel'],
        values=np.deg2rad(np.linspace(10.0, 170.0, n_pixel)),
        unit='rad',
    )
    events = sc.DataArray(
        sc.ones(dims=['event'], shape=[n_pixel * n_event], unit='counts'),
        coords={
            'dspacing': sc.array(
                dims=['event'],
                values=rng.uniform(1.0, 2.0, n_pixel * n_event),
                unit='angstrom',
            ),
            'pixel': sc.array(
                dims=['event'], values=np.repeat(np.arange(n_pixel), n_event)
            ),
        },
    )
    return events.group('pixel').drop_coords('pixel').assign_coords(two_theta=two_theta)


def _focus_and_group(
    detector: sc.DataArray, two_theta_bins: sc.Variable, *, keep_events: bool
) -> sc.DataArray:
    focussed = focus_data_dspacing_and_two_theta(
        detector,
        DSPACING_BINS,
        two_theta_bins,
        KeepEvents[SampleRun](keep_events),
    )
    grouped = group_two_theta(focussed, two_theta_bins)
    return grouped if grouped.bins is None else grouped.hist()


@pytest.mark.parametrize('keep_events', [True, False])
@pytest.mark.parametrize(
    'two_theta_bins',
    [
        sc.linspace('two_theta', 75.0, 105.0, num=180, unit='deg'),
        sc.linspace('two_theta', 0.8, 2.4, num=17, unit='rad'),
        sc.array(dims=['two_theta'], values=[0.5, 0.6, 1.5, 1.55, 2.9], unit='rad'),
    ],
    ids=['many-narrow-deg', 'few-wide-rad', 'non-uniform-rad'],
)
def test_group_two_theta_matches_direct_histogram(
    detector, two_theta_bins, keep_events
):
    """Focussing must not redistribute counts between requested two-theta bins.

    Focussing bins that are not aligned with the requested bins get assigned to
    whichever requested bin contains their center. The number of bins per group
    then varies periodically, producing large spikes.
    """
    result = _focus_and_group(detector, two_theta_bins, keep_events=keep_events)
    expected = detector.hist(
        two_theta=two_theta_bins.to(unit=detector.coords['two_theta'].unit),
        dspacing=DSPACING_BINS,
    )
    assert sc.allclose(result.data, expected.data)


def test_group_two_theta_preserves_requested_bins(detector):
    two_theta_bins = sc.linspace('two_theta', 75.0, 105.0, num=180, unit='deg')
    result = _focus_and_group(detector, two_theta_bins, keep_events=False)
    assert sc.identical(result.coords['two_theta'], two_theta_bins)


@pytest.mark.parametrize('keep_events', [True, False])
def test_integrate_two_theta_covers_full_detector_range(detector, keep_events):
    """The requested bins may cover only part of the detector."""
    two_theta_bins = sc.linspace('two_theta', 75.0, 105.0, num=180, unit='deg')
    focussed = focus_data_dspacing_and_two_theta(
        detector, DSPACING_BINS, two_theta_bins, KeepEvents[SampleRun](keep_events)
    )
    result = integrate_two_theta(focussed)
    if result.bins is not None:
        result = result.hist()
    expected = detector.hist(dspacing=DSPACING_BINS).sum('pixel')
    assert sc.allclose(result.data, expected.data)


def test_focussing_without_requested_bins_keeps_all_counts(detector):
    focussed = focus_data_dspacing_and_two_theta(
        detector, DSPACING_BINS, None, KeepEvents[SampleRun](False)
    )
    assert sc.allclose(
        focussed.sum().data, detector.hist(dspacing=DSPACING_BINS).sum().data
    )


def test_group_two_theta_without_requested_bins_raises(detector):
    focussed = focus_data_dspacing_and_two_theta(
        detector, DSPACING_BINS, None, KeepEvents[SampleRun](False)
    )
    with pytest.raises(ValueError, match='TwoThetaBins'):
        group_two_theta(focussed, None)
