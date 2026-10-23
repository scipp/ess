# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable

import numpy as np
import pytest
import scipp as sc
import scipp.testing
from ess.loki import straw_calibration as cal

ACTIVE_HALF_LENGTH = 467.0  # mm
PEAK_WIDTH = 6.0  # mm
SLITS = sc.arange('slit', -450.0, 451.0, 100.0, unit='mm')
# Nominal pixel positions decrease with the pixel index, as in LoKI files.
PIXELS = sc.linspace('pixel', 499.0, -499.0, num=512, unit='mm')


def response(scale: float, offset: float, cubic: float) -> Callable:
    """Nominal position as a function of the true position along a straw (in mm)."""

    def f(true: np.ndarray) -> np.ndarray:
        return offset + scale * true + cubic * true**3 / ACTIVE_HALF_LENGTH**2

    return f


def straw_counts(
    rng: np.random.Generator, slit_images: np.ndarray, to_nominal: Callable
) -> np.ndarray:
    """Counts per pixel for slit images on a flat background."""
    background = rng.uniform(-ACTIVE_HALF_LENGTH, ACTIVE_HALF_LENGTH, 400_000)
    peaks = rng.normal(np.repeat(slit_images, 20_000), PEAK_WIDTH)
    true = np.concatenate([background, peaks])
    true = true[np.abs(true) < ACTIVE_HALF_LENGTH]
    pitch = abs((PIXELS[1] - PIXELS[0]).value)
    edges = np.r_[PIXELS.values[::-1] - pitch / 2, PIXELS.values[0] + pitch / 2]
    return np.histogram(to_nominal(true), bins=edges)[0][::-1]


def make_counts(straws: list[list[np.ndarray]]) -> sc.DataArray:
    return sc.DataArray(
        sc.array(dims=['tube', 'straw', 'pixel'], values=straws, unit='counts'),
        coords={'pixel': PIXELS},
    )


def pixel_index(nominal: float) -> int:
    return int(np.argmin(np.abs(PIXELS.values - nominal)))


RESPONSES = [
    response(0.9, 5.0, 0.0),
    response(1.0, -20.0, -0.02),
    response(0.85, 25.0, 0.01),
]


@pytest.fixture(scope='module')
def unmagnified() -> tuple[sc.DataArray, sc.DataArray]:
    rng = np.random.default_rng(1234)
    counts = make_counts([[straw_counts(rng, SLITS.values, r) for r in RESPONSES]])
    return counts, cal.find_slit_peaks(counts, SLITS)


def test_projected_slit_positions_without_distance_returns_slits() -> None:
    result = cal.projected_slit_positions(
        SLITS,
        source_to_mask=sc.scalar(1.0, unit='m'),
        mask_to_straw=sc.scalar(0.0, unit='mm'),
    )
    sc.testing.assert_allclose(result, SLITS)


def test_projected_slit_positions_magnifies_about_source() -> None:
    result = cal.projected_slit_positions(
        SLITS,
        source_to_mask=sc.scalar(0.4, unit='m'),
        mask_to_straw=sc.array(dims=['straw'], values=[0.0, 40.0], unit='mm'),
        source_offset=sc.scalar(50.0, unit='mm'),
    )
    assert result.dims == ('straw', 'slit')
    sc.testing.assert_allclose(result['straw', 0], SLITS)
    sc.testing.assert_allclose(
        result['straw', 1],
        sc.scalar(50.0, unit='mm') + (SLITS - 50.0 * SLITS.unit) * 1.1,
    )


def test_find_slit_peaks_finds_all_peaks(
    unmagnified: tuple[sc.DataArray, sc.DataArray],
) -> None:
    _, peaks = unmagnified
    assert peaks.dims == ('tube', 'straw', 'slit')
    assert not peaks.masks['not_found'].values.any()
    for straw, r in enumerate(RESPONSES):
        np.testing.assert_allclose(
            peaks['tube', 0]['straw', straw].values, r(SLITS.values), atol=0.5
        )
    sc.testing.assert_identical(
        peaks.coords['expected'],
        SLITS.broadcast(sizes={'tube': 1, 'straw': 3, 'slit': 10}),
    )


def test_find_slit_peaks_reports_active_region_and_sigma(
    unmagnified: tuple[sc.DataArray, sc.DataArray],
) -> None:
    _, peaks = unmagnified
    for straw, r in enumerate(RESPONSES):
        p = peaks['tube', 0]['straw', straw]
        assert abs(p.coords['active_begin'].value - r(-ACTIVE_HALF_LENGTH)) < 3.0
        assert abs(p.coords['active_end'].value - r(ACTIVE_HALF_LENGTH)) < 3.0
        np.testing.assert_allclose(
            p.coords['sigma'].values, PEAK_WIDTH * (r(1.0) - r(0.0)), rtol=0.2
        )


def test_find_slit_peaks_does_not_shift_slits_when_outer_images_are_cut_off() -> None:
    # Magnified images of the outermost slits fall outside the active region.
    # A comb shifted by one slit then matches the visible peaks equally well.
    rng = np.random.default_rng(42)
    images = SLITS * 1.08
    r = response(0.9, 25.0, 0.0)
    counts = make_counts([[straw_counts(rng, images.values, r)]])
    peaks = cal.find_slit_peaks(counts, images)
    found = ~peaks.masks['not_found'].values[0, 0]
    np.testing.assert_array_equal(found, [False] + [True] * 8 + [False])
    np.testing.assert_allclose(
        peaks.values[0, 0][found], r(images.values[found]), atol=0.5
    )


@pytest.mark.parametrize('end', [-1, 1])
def test_find_slit_peaks_ignores_hot_pixel_outside_active_region(end: int) -> None:
    rng = np.random.default_rng(3)
    images = SLITS * 1.08
    r = response(0.9, 25.0, 0.0)
    y = straw_counts(rng, images.values, r)
    y[pixel_index(end * 495.0)] = 500
    peaks = cal.find_slit_peaks(make_counts([[y]]), images)
    found = ~peaks.masks['not_found'].values[0, 0]
    np.testing.assert_array_equal(found, [False] + [True] * 8 + [False])
    np.testing.assert_allclose(
        peaks.values[0, 0][found], r(images.values[found]), atol=0.5
    )


def test_find_slit_peaks_does_not_report_spike_as_peak() -> None:
    rng = np.random.default_rng(5)
    r = response(0.9, 0.0, 0.0)
    background = rng.poisson(800, size=512)
    background[(PIXELS.values < r(-ACTIVE_HALF_LENGTH))] = 0
    background[(PIXELS.values > r(ACTIVE_HALF_LENGTH))] = 0
    background[pixel_index(r(150.0))] = 5000
    peaks = cal.find_slit_peaks(make_counts([[background]]), SLITS)
    assert peaks.masks['not_found'].values.all()


def test_find_slit_peaks_masks_all_peaks_of_dead_straw() -> None:
    rng = np.random.default_rng(1)
    counts = make_counts(
        [[straw_counts(rng, SLITS.values, RESPONSES[0]), np.zeros(512)]]
    )
    peaks = cal.find_slit_peaks(counts, SLITS)
    assert not peaks.masks['not_found']['straw', 0].values.any()
    assert peaks.masks['not_found']['straw', 1].values.all()
    assert np.isnan(peaks.coords['active_begin']['tube', 0]['straw', 1].value)


def test_find_slit_peaks_accepts_increasing_positions(
    unmagnified: tuple[sc.DataArray, sc.DataArray],
) -> None:
    counts, peaks = unmagnified
    flipped = counts.copy()
    flipped.data.values = counts.data.values[..., ::-1]
    flipped.coords['pixel'].values = counts.coords['pixel'].values[::-1]
    sc.testing.assert_identical(cal.find_slit_peaks(flipped, SLITS), peaks)


def test_find_slit_peaks_accepts_decreasing_expected_positions(
    unmagnified: tuple[sc.DataArray, sc.DataArray],
) -> None:
    counts, peaks = unmagnified
    reversed_slits = sc.array(dims=['slit'], values=SLITS.values[::-1], unit='mm')
    result = cal.find_slit_peaks(counts, reversed_slits)
    np.testing.assert_array_equal(result.values[..., ::-1], peaks.values)


def test_find_slit_peaks_raises_if_counts_have_masks(
    unmagnified: tuple[sc.DataArray, sc.DataArray],
) -> None:
    counts, _ = unmagnified
    masked = counts.copy(deep=False)
    masked.masks['bad'] = sc.zeros(sizes={'pixel': 512}, dtype=bool)
    with pytest.raises(ValueError, match='Masks'):
        cal.find_slit_peaks(masked, SLITS)


def test_fit_straw_calibration_recovers_true_positions(
    unmagnified: tuple[sc.DataArray, sc.DataArray],
) -> None:
    _, peaks = unmagnified
    coefficients = cal.fit_straw_calibration(peaks)
    assert coefficients.dims == ('tube', 'straw', 'power')
    assert (coefficients.coords['degree'] == sc.index(3)).all()
    assert (coefficients.coords['n_peaks'] == sc.index(10)).all()
    assert (coefficients.coords['residual_stddev'] < sc.scalar(0.5, unit='mm')).all()
    true = np.linspace(-450.0, 450.0, 50)
    for straw, r in enumerate(RESPONSES):
        nominal = sc.array(dims=['x'], values=r(true), unit='mm')
        result = cal.calibrate_positions(
            coefficients['tube', 0]['straw', straw], nominal
        )
        np.testing.assert_allclose(result.values, true, atol=1.0)


def make_peaks(n_valid: int) -> sc.DataArray:
    return sc.DataArray(
        SLITS * 0.9 + sc.scalar(3.0, unit='mm'),
        coords={'expected': SLITS},
        masks={'not_found': sc.arange('slit', 10, unit=None) >= sc.index(n_valid)},
    )


@pytest.mark.parametrize(('n_valid', 'degree'), [(10, 3), (6, 3), (5, 2), (4, 1)])
def test_fit_straw_calibration_reduces_degree_for_few_peaks(
    n_valid: int, degree: int
) -> None:
    coefficients = cal.fit_straw_calibration(make_peaks(n_valid))
    assert coefficients.coords['degree'].value == degree
    assert coefficients.coords['n_peaks'].value == n_valid
    assert not coefficients.masks['uncalibrated'].value
    np.testing.assert_allclose(coefficients.values[degree + 1 :], 0.0)


def test_fit_straw_calibration_masks_straws_with_too_few_peaks() -> None:
    coefficients = cal.fit_straw_calibration(make_peaks(3))
    assert coefficients.masks['uncalibrated'].value
    assert np.isnan(coefficients.values).all()


def test_fit_straw_calibration_excludes_peaks_with_any_mask() -> None:
    peaks = make_peaks(10)
    peaks.masks['bad_slit'] = sc.arange('slit', 10, unit=None) < sc.index(5)
    coefficients = cal.fit_straw_calibration(peaks)
    assert coefficients.coords['n_peaks'].value == 5
    sc.testing.assert_identical(coefficients.coords['fit_begin'], peaks.data['slit', 5])
    sc.testing.assert_identical(coefficients.coords['fit_end'], peaks.data['slit', 9])


def test_calibrate_positions_converts_units() -> None:
    coefficients = cal.fit_straw_calibration(make_peaks(10))
    x = sc.array(dims=['pixel'], values=[-0.3, 0.0, 0.3], unit='m')
    result = cal.calibrate_positions(coefficients, x)
    assert result.unit == 'mm'
    np.testing.assert_allclose(result.values, (x.to(unit='mm').values - 3.0) / 0.9)
    assert 'uncalibrated' in result.masks
    in_metres = cal.calibrate_positions(coefficients.to(unit='m'), x)
    sc.testing.assert_allclose(in_metres.data, result.data.to(unit='m'))


@pytest.fixture
def magnified_peaks() -> tuple[sc.DataArray, sc.Variable]:
    """Peaks for a source 445 mm in front of the first layer, without noise."""
    sizes = {'layer': 3, 'tube': 20, 'straw': 7}
    depth = sc.array(
        dims=['layer', 'straw'],
        values=np.arange(3)[:, None] * 26.26 + np.linspace(-7.7, 7.7, 7),
        unit='mm',
    ).broadcast(sizes=sizes)
    readout_scale = sc.array(
        dims=['straw'], values=[1.0, 0.92, 0.85, 0.84, 0.87, 0.91, 1.0]
    )
    images = cal.projected_slit_positions(
        SLITS, source_to_mask=sc.scalar(445.0, unit='mm'), mask_to_straw=depth
    )
    peaks = sc.DataArray(
        images * readout_scale,
        masks={'not_found': sc.zeros(sizes={**sizes, 'slit': 10}, dtype=bool)},
    )
    return peaks, depth


def test_estimate_source_distance_recovers_distance(
    magnified_peaks: tuple[sc.DataArray, sc.Variable],
) -> None:
    peaks, depth = magnified_peaks
    rng = np.random.default_rng(7)
    # Readout scale varying between tubes, noise, and a few broken straws.
    peaks.data *= sc.array(dims=['tube'], values=rng.normal(1.0, 0.01, 20))
    peaks.values += rng.normal(0.0, 0.3, peaks.shape)
    peaks.values[0, 0, :3] *= 1.3
    result = cal.estimate_source_distance(peaks, slits=SLITS, depth=depth)
    assert result.unit == 'mm'
    assert abs(result.value - 445.0) < 3 * np.sqrt(result.variance)
    assert np.sqrt(result.variance) < 0.05 * 445.0


def test_estimate_source_distance_raises_for_single_layer(
    magnified_peaks: tuple[sc.DataArray, sc.Variable],
) -> None:
    peaks, depth = magnified_peaks
    with pytest.raises(ValueError, match='more than one depth'):
        cal.estimate_source_distance(
            peaks['layer', 0:1], slits=SLITS, depth=depth['layer', 0:1]
        )
