# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
import scipp.testing

from ess.powder.correction import (
    apply_lorentz_correction,
    merge_calibration,
    normalize_by_monitor_histogram,
    normalize_by_monitor_integrated,
    normalize_by_vanadium_dspacing,
    normalize_by_vanadium_dspacing_and_two_theta,
)
from ess.powder.types import (
    CaveMonitor,
    CorrectedDspacing,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    NormalizedDspacing,
    SampleRun,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthMonitor,
)


@pytest.fixture
def calibration():
    rng = np.random.default_rng(789236)
    n = 30
    ds = sc.Dataset(
        data={
            'difa': sc.array(
                dims=['spectrum'],
                values=rng.uniform(1.0e2, 1.0e3, n),
                unit='us / angstrom**2',
            ),
            'difc': sc.array(
                dims=['spectrum'],
                values=rng.uniform(1.0e3, 1.0e4, n),
                unit='us / angstrom',
            ),
            'tzero': sc.array(
                dims=['spectrum'], values=rng.uniform(-1e2, 1e2, n), unit='us'
            ),
            'mask': sc.full(dims=['spectrum'], shape=[n], value=False, unit=None),
        },
        coords={'spectrum': sc.arange('spectrum', n, unit=None)},
    )
    return ds


def test_merge_calibration_add_all_parameters(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None)
        },
    )
    with_cal = merge_calibration(into=da, calibration=calibration)

    assert sc.identical(with_cal.coords['difa'], calibration['difa'].data)
    assert sc.identical(with_cal.coords['difc'], calibration['difc'].data)
    assert sc.identical(with_cal.coords['tzero'], calibration['tzero'].data)
    assert sc.identical(with_cal.masks['calibration'], calibration['mask'].data)


def test_merge_calibration_raises_if_spectrum_mismatch(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.zeros(
                sizes={'spectrum': calibration.sizes['spectrum']}, unit=None
            )
        },
    )
    with pytest.raises(ValueError, match='spectrum of calibration and target'):
        merge_calibration(into=da, calibration=calibration)


def test_merge_calibration_raises_if_difa_exists(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None),
            'difa': sc.ones(sizes={'spectrum': calibration.sizes['spectrum']}),
        },
    )
    with pytest.raises(
        ValueError, match='there already is metadata with the same name'
    ):
        merge_calibration(into=da, calibration=calibration)


def test_merge_calibration_raises_if_difc_exists(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None),
            'difc': sc.ones(sizes={'spectrum': calibration.sizes['spectrum']}),
        },
    )
    with pytest.raises(
        ValueError, match='there already is metadata with the same name'
    ):
        merge_calibration(into=da, calibration=calibration)


def test_merge_calibration_raises_if_tzero_exists(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None),
            'tzero': sc.ones(sizes={'spectrum': calibration.sizes['spectrum']}),
        },
    )
    with pytest.raises(
        ValueError, match='there already is metadata with the same name'
    ):
        merge_calibration(into=da, calibration=calibration)


def test_merge_calibration_raises_if_mask_exists(calibration):
    da = sc.DataArray(
        sc.ones(sizes=calibration.sizes),
        coords={
            'spectrum': sc.arange('spectrum', calibration.sizes['spectrum'], unit=None)
        },
        masks={
            'calibration': sc.ones(sizes={'spectrum': calibration.sizes['spectrum']})
        },
    )
    with pytest.raises(ValueError, match='there already is a mask with the same name'):
        merge_calibration(into=da, calibration=calibration)


@pytest.mark.parametrize('data_dtype', ['float32', 'float64'])
@pytest.mark.parametrize('dspacing_dtype', ['float32', 'float64'])
@pytest.mark.parametrize('two_theta_dtype', ['float32', 'float64'])
def test_lorentz_correction_dense_1d_coords(
    data_dtype, dspacing_dtype, two_theta_dtype
):
    da = sc.DataArray(
        sc.full(
            value=2.1,
            sizes={'detector_number': 3, 'dspacing': 4},
            unit='counts',
            dtype=data_dtype,
        ),
        coords={
            'dspacing': sc.array(
                dims=['dspacing'],
                values=[0.1, 0.4, 0.7, 1.1],
                unit='angstrom',
                dtype=dspacing_dtype,
            ),
            'two_theta': sc.array(
                dims=['detector_number'],
                values=[0.8, 0.9, 1.3],
                unit='rad',
                dtype=two_theta_dtype,
            ),
            'detector_number': sc.array(
                dims=['detector_number'], values=[0, 1, 2], unit=None
            ),
        },
    )
    original = da.copy(deep=True)
    corrected = apply_lorentz_correction(da)

    assert corrected.sizes == {'detector_number': 3, 'dspacing': 4}
    assert corrected.unit == 'angstrom**4 * counts'
    assert corrected.dtype == original.dtype
    assert not corrected.variances
    assert not corrected.bins

    d = original.coords['dspacing'].broadcast(sizes=corrected.sizes).values
    two_theta = original.coords['two_theta'].broadcast(sizes=corrected.sizes).values
    if any(dt == 'float32' for dt in (data_dtype, dspacing_dtype, two_theta_dtype)):
        rtol = 1e-6
    else:
        rtol = 1e-15
    np.testing.assert_allclose(
        corrected.data.values, 2.1 * d**4 * np.sin(two_theta / 2), rtol=rtol
    )

    assert set(corrected.coords.keys()) == {'two_theta', 'dspacing', 'detector_number'}
    for key, coord in corrected.coords.items():
        sc.testing.assert_identical(coord, original.coords[key])
        sc.testing.assert_identical(da.coords[key], original.coords[key])


def test_apply_lorentz_correction_dense_2d_coord():
    da = sc.DataArray(
        sc.full(value=0.7, sizes={'detector_number': 3, 'dspacing': 4}),
        coords={
            'dspacing': sc.array(
                dims=['dspacing'], values=[0.1, 0.4, 0.7, 1.1], unit='angstrom'
            ).broadcast(sizes={'detector_number': 3, 'dspacing': 4}),
            'two_theta': sc.array(
                dims=['detector_number'], values=[0.8, 0.9, 1.3], unit='rad'
            ),
            'detector_number': sc.array(
                dims=['detector_number'], values=[0, 1, 2], unit=None
            ),
        },
    )
    original = da.copy(deep=True)
    corrected = apply_lorentz_correction(da)

    assert corrected.sizes == {'detector_number': 3, 'dspacing': 4}
    assert corrected.unit == 'angstrom**4'
    assert corrected.dtype == original.dtype
    assert not corrected.variances
    assert not corrected.bins

    d = original.coords['dspacing'].values
    two_theta = original.coords['two_theta'].broadcast(sizes=corrected.sizes).values
    np.testing.assert_allclose(
        corrected.data.values, 0.7 * d**4 * np.sin(two_theta / 2)
    )

    assert set(corrected.coords.keys()) == {'two_theta', 'dspacing', 'detector_number'}
    for key, coord in corrected.coords.items():
        sc.testing.assert_identical(coord, original.coords[key])
        sc.testing.assert_identical(da.coords[key], original.coords[key])


@pytest.mark.parametrize('data_dtype', ['float32', 'float64'])
@pytest.mark.parametrize('dspacing_dtype', ['float32', 'float64'])
@pytest.mark.parametrize('two_theta_dtype', ['float32', 'float64'])
def test_apply_lorentz_correction_event_coords(
    data_dtype, dspacing_dtype, two_theta_dtype
):
    buffer = sc.DataArray(
        sc.full(value=1.5, sizes={'event': 6}, unit='counts', dtype=data_dtype),
        coords={
            'detector_number': sc.array(dims=['event'], values=[0, 3, 2, 2, 0, 4]),
            'dspacing': sc.array(
                dims=['event'],
                values=[0.1, 0.4, 0.2, 1.0, 1.3, 0.7],
                unit='angstrom',
                dtype=dspacing_dtype,
            ),
        },
    )
    da = buffer.group('detector_number').bin(dspacing=2)
    da.coords['two_theta'] = sc.array(
        dims=['detector_number'],
        values=[0.4, 1.2, 1.5, 1.6],
        unit='rad',
        dtype=two_theta_dtype,
    )
    original = da.copy(deep=True)
    corrected = apply_lorentz_correction(da)

    assert corrected.sizes == {'detector_number': 4, 'dspacing': 2}
    assert corrected.bins.unit == 'angstrom**4 * counts'
    assert corrected.bins.dtype == data_dtype

    d = original.bins.coords['dspacing']
    two_theta = sc.bins_like(original, original.coords['two_theta'])
    expected = (1.5 * d**4 * sc.sin(two_theta / 2)).to(dtype=data_dtype)
    if any(dt == 'float32' for dt in (data_dtype, dspacing_dtype, two_theta_dtype)):
        rtol = 1e-6
    else:
        rtol = 1e-15
    np.testing.assert_allclose(
        corrected.bins.concat().value.values,
        expected.bins.concat().value.values,
        rtol=rtol,
    )

    assert set(corrected.coords.keys()) == {'detector_number', 'two_theta', 'dspacing'}
    for key, coord in corrected.coords.items():
        sc.testing.assert_identical(coord, original.coords[key])
        sc.testing.assert_identical(da.coords[key], original.coords[key])
    sc.testing.assert_identical(
        corrected.bins.coords['dspacing'], original.bins.coords['dspacing']
    )
    sc.testing.assert_identical(
        da.bins.coords['dspacing'], original.bins.coords['dspacing']
    )


def test_apply_lorentz_correction_favors_event_coords():
    buffer = sc.DataArray(
        sc.full(value=1.5, sizes={'event': 6}, unit='counts'),
        coords={
            'detector_number': sc.array(dims=['event'], values=[0, 3, 2, 2, 0, 4]),
            'dspacing': sc.array(
                dims=['event'],
                values=[0.1, 0.4, 0.2, 1.0, 1.3, 0.7],
                unit='angstrom',
            ),
        },
    )
    da = buffer.group('detector_number').bin(dspacing=2)
    da.coords['two_theta'] = sc.array(
        dims=['detector_number'],
        values=[0.4, 1.2, 1.5, 1.6],
        unit='rad',
    )
    da.coords['dspacing'][-1] = 10.0  # this should not affect the correction
    corrected = apply_lorentz_correction(da)

    d = da.bins.coords['dspacing']  # event-coord, not the modified bin-coord
    two_theta = sc.bins_like(da, da.coords['two_theta'])
    expected = 1.5 * d**4 * sc.sin(two_theta / 2)
    np.testing.assert_allclose(
        corrected.bins.concat().value.values,
        expected.bins.concat().value.values,
        rtol=1e-15,
    )

    for key, coord in corrected.coords.items():
        sc.testing.assert_identical(coord, da.coords[key])
        sc.testing.assert_identical(da.coords[key], da.coords[key])
    sc.testing.assert_identical(
        corrected.bins.coords['dspacing'], da.bins.coords['dspacing']
    )


def test_apply_lorentz_correction_needs_coords():
    da = sc.DataArray(
        sc.ones(sizes={'detector_number': 3, 'dspacing': 4}),
        coords={
            'detector_number': sc.array(
                dims=['detector_number'], values=[0, 1, 2], unit=None
            )
        },
    )
    with pytest.raises(KeyError):
        apply_lorentz_correction(da)


def test_normalize_by_monitor_histogram_expected_results():
    detector = sc.DataArray(
        sc.arange('wavelength', 3, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 3.0, unit='Å')},
    ).bin(wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[5.0, 6.0], unit='counts'),
        coords={
            'wavelength': sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å')
        },
    )
    normalized = normalize_by_monitor_histogram(
        CorrectedDspacing[SampleRun](detector),
        monitor=WavelengthMonitor[SampleRun, CaveMonitor](monitor),
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    mon_coord = monitor.coords['wavelength']
    expected = NormalizedDspacing[SampleRun](
        detector / monitor.data * (mon_coord[1:] - mon_coord[:-1])
    )
    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_ignores_monitor_values_out_of_range():
    detector = sc.DataArray(
        sc.arange('wavelength', 3, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 3.0, unit='Å')},
    ).bin(wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[4.0, 10.0], unit='counts'),
        coords={
            'wavelength': sc.array(dims=['wavelength'], values=[0.0, 3, 4], unit='Å')
        },
    )
    normalized = normalize_by_monitor_histogram(
        CorrectedDspacing[SampleRun](detector),
        monitor=WavelengthMonitor[SampleRun, CaveMonitor](monitor),
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )
    expected = NormalizedDspacing[SampleRun](
        detector / sc.scalar(4.0, unit='counts') * sc.scalar(3.0, unit='Å')
    )
    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_integrated_expected_results():
    detector = sc.DataArray(
        sc.arange('wavelength', 1, 4, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 3.0, unit='Å')},
    ).bin(wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[4.0, 5.0, 6.0], unit='counts'),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[0.0, 0.5, 2, 3], unit='Å'
            )
        },
    )
    normalized = normalize_by_monitor_integrated(
        CorrectedDspacing[SampleRun](detector),
        monitor=WavelengthMonitor[SampleRun, CaveMonitor](monitor),
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )
    expected = NormalizedDspacing[SampleRun](
        detector / sc.scalar(4 + 5 + 6, unit='counts * Å')
    )
    sc.testing.assert_identical(normalized, expected)


@pytest.mark.parametrize('event_coord', [True, False])
def test_normalize_by_monitor_integrated_uses_monitor_values_at_boundary(
    event_coord: bool,
):
    detector = sc.DataArray(
        sc.arange('wavelength', 4, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 4.0, unit='Å')},
    )
    if event_coord:
        # Make sure event at 3 is included
        detector = detector.bin(
            wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3.1], unit='Å')
        )
        del detector.coords['wavelength']
    else:
        detector = detector.bin(
            wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å')
        )
        del detector.bins.coords['wavelength']
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[4.0, 10.0], unit='counts'),
        coords={
            'wavelength': sc.array(dims=['wavelength'], values=[0.0, 2, 4], unit='Å')
        },
    )
    normalized = normalize_by_monitor_integrated(
        CorrectedDspacing[SampleRun](detector),
        monitor=WavelengthMonitor[SampleRun, CaveMonitor](monitor),
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )
    expected = NormalizedDspacing[SampleRun](
        detector / sc.scalar(4.0 + 10.0, unit='counts')
    )
    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_integrated_assigns_mask_if_monitor_range_too_narrow():
    detector = sc.DataArray(
        sc.arange('wavelength', 3, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 3.0, unit='Å')},
    ).bin(wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[4.0, 10.0], unit='counts'),
        coords={
            'wavelength': sc.array(dims=['wavelength'], values=[1.0, 3, 4], unit='Å')
        },
    )
    with pytest.raises(ValueError, match="smaller than the range of the detector"):
        normalize_by_monitor_integrated(
            CorrectedDspacing[SampleRun](detector),
            monitor=WavelengthMonitor[SampleRun, CaveMonitor](monitor),
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )


class TestNormalizeByVanadium:
    def random_variable(
        self,
        rng: np.random.Generator,
        dim: str,
        n: int,
        unit: str,
        with_variances: bool = False,
    ) -> sc.Variable:
        values = rng.uniform(0.1, 2.0, n)
        variances = values * rng.uniform(0.1, 0.5, n) if with_variances else None
        return sc.array(dims=[dim], values=values, variances=variances, unit=unit)

    def random_binned_data(
        self,
        rng: np.random.Generator,
        n_events: int,
        unit: str,
        with_variances: bool = False,
        *coords: tuple[str, int, str],
    ) -> sc.DataArray:
        return sc.DataArray(
            self.random_variable(
                rng, 'event', n_events, unit, with_variances=with_variances
            ),
            coords={
                dim: self.random_variable(rng, 'event', n_events, coord_unit)
                for (dim, _, coord_unit) in coords
            },
        ).bin({dim: n_bins for (dim, n_bins, _) in coords})

    def make_sample_and_vanadium_1d(self) -> tuple[sc.DataArray, sc.DataArray]:
        rng = np.random.default_rng(seed=495)
        sample = self.random_binned_data(rng, 84, 'count', True, ('dspacing', 35, 'Å'))
        vanadium = self.random_binned_data(
            rng, 146, 'count', True, ('dspacing', 79, 'Å')
        )
        return sample, vanadium

    def make_sample_and_vanadium_2d(self) -> tuple[sc.DataArray, sc.DataArray]:
        rng = np.random.default_rng(seed=3193)
        sample = self.random_binned_data(
            rng, 138, 'count', True, ('dspacing', 35, 'Å'), ('two_theta', 13, 'rad')
        )
        vanadium = self.random_binned_data(
            rng, 170, 'count', True, ('dspacing', 79, 'Å'), ('two_theta', 14, 'rad')
        )
        return sample, vanadium

    def test_1d_binned_vanadium(self) -> None:
        sample, vanadium = self.make_sample_and_vanadium_1d()
        normed = normalize_by_vanadium_dspacing(
            FocussedDataDspacing[SampleRun](sample),
            FocussedDataDspacing[VanadiumRun](vanadium),
            UncertaintyBroadcastMode.drop,
        )
        # we test masks separately
        normed = normed.drop_masks(list(normed.masks.keys()))

        norm = vanadium.hist(dspacing=sample.coords['dspacing'])
        expected = sample / sc.values(norm)
        sc.testing.assert_allclose(normed, expected)

    def test_1d_histogrammed_vanadium(self) -> None:
        sample, vanadium = self.make_sample_and_vanadium_1d()
        vanadium = vanadium.hist()
        normed = normalize_by_vanadium_dspacing(
            FocussedDataDspacing[SampleRun](sample),
            FocussedDataDspacing[VanadiumRun](vanadium),
            UncertaintyBroadcastMode.drop,
        )
        # we test masks separately
        normed = normed.drop_masks(list(normed.masks.keys()))

        norm = vanadium.rebin(dspacing=sample.coords['dspacing'])
        expected = sample / sc.values(norm)
        sc.testing.assert_allclose(normed, expected)

    def test_1d_binned_vanadium_binning_has_no_effect(self) -> None:
        sample, vanadium = self.make_sample_and_vanadium_1d()
        vana_binned_like_sample = vanadium.bin(dspacing=sample.coords['dspacing'])
        normed_a = normalize_by_vanadium_dspacing(
            FocussedDataDspacing[SampleRun](sample),
            FocussedDataDspacing[VanadiumRun](vanadium),
            UncertaintyBroadcastMode.drop,
        )
        normed_b = normalize_by_vanadium_dspacing(
            FocussedDataDspacing[SampleRun](sample),
            FocussedDataDspacing[VanadiumRun](vana_binned_like_sample),
            UncertaintyBroadcastMode.drop,
        )
        sc.testing.assert_allclose(normed_a, normed_b)

    def test_1d_masks_zero_vanadium_bins(self) -> None:
        sample, vanadium = self.make_sample_and_vanadium_1d()
        vanadium['dspacing', 5] = sc.scalar(0.0, variance=0.0, unit='counts')
        normed = normalize_by_vanadium_dspacing(
            FocussedDataDspacing[SampleRun](sample),
            FocussedDataDspacing[VanadiumRun](vanadium),
            UncertaintyBroadcastMode.drop,
        )

        norm = vanadium.hist(dspacing=sample.coords['dspacing'])

        sc.testing.assert_allclose(
            normed.masks['zero_vanadium'], norm.data == sc.scalar(0.0, unit='counts')
        )

    def test_2d_binned_vanadium(self) -> None:
        sample, vanadium = self.make_sample_and_vanadium_2d()
        normed = normalize_by_vanadium_dspacing_and_two_theta(
            FocussedDataDspacingTwoTheta[SampleRun](sample),
            FocussedDataDspacingTwoTheta[VanadiumRun](vanadium),
            UncertaintyBroadcastMode.drop,
        )
        # we test masks separately
        normed = normed.drop_masks(list(normed.masks.keys()))

        norm = vanadium.hist(
            dspacing=sample.coords['dspacing'], two_theta=sample.coords['two_theta']
        )
        expected = sample / sc.values(norm)
        sc.testing.assert_allclose(normed, expected)

    def test_2d_histogrammed_vanadium(self) -> None:
        sample, vanadium = self.make_sample_and_vanadium_2d()
        vanadium = vanadium.hist()
        normed = normalize_by_vanadium_dspacing_and_two_theta(
            FocussedDataDspacingTwoTheta[SampleRun](sample),
            FocussedDataDspacingTwoTheta[VanadiumRun](vanadium),
            UncertaintyBroadcastMode.drop,
        )
        # we test masks separately
        normed = normed.drop_masks(list(normed.masks.keys()))

        norm = vanadium.rebin(
            dspacing=sample.coords['dspacing'], two_theta=sample.coords['two_theta']
        )
        expected = sample / sc.values(norm)
        sc.testing.assert_allclose(normed, expected)

    def test_2d_binned_vanadium_binning_has_no_effect(self) -> None:
        sample, vanadium = self.make_sample_and_vanadium_2d()
        vana_binned_like_sample = vanadium.bin(
            dspacing=sample.coords['dspacing'], two_theta=sample.coords['two_theta']
        )
        normed_a = normalize_by_vanadium_dspacing_and_two_theta(
            FocussedDataDspacingTwoTheta[SampleRun](sample),
            FocussedDataDspacingTwoTheta[VanadiumRun](vanadium),
            UncertaintyBroadcastMode.drop,
        )
        normed_b = normalize_by_vanadium_dspacing_and_two_theta(
            FocussedDataDspacingTwoTheta[SampleRun](sample),
            FocussedDataDspacingTwoTheta[VanadiumRun](vana_binned_like_sample),
            UncertaintyBroadcastMode.drop,
        )
        sc.testing.assert_allclose(normed_a, normed_b)

    def test_2d_masks_zero_vanadium_bins(self) -> None:
        sample, vanadium = self.make_sample_and_vanadium_2d()
        vanadium['dspacing', 5]['two_theta', 7] = sc.scalar(
            0.0, variance=0.0, unit='counts'
        )
        normed = normalize_by_vanadium_dspacing_and_two_theta(
            FocussedDataDspacingTwoTheta[SampleRun](sample),
            FocussedDataDspacingTwoTheta[VanadiumRun](vanadium),
            UncertaintyBroadcastMode.drop,
        )

        norm = vanadium.hist(
            dspacing=sample.coords['dspacing'], two_theta=sample.coords['two_theta']
        )

        sc.testing.assert_allclose(
            normed.masks['zero_vanadium'], norm.data == sc.scalar(0.0, unit='counts')
        )
