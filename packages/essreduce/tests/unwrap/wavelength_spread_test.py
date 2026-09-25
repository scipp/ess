# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.reduce.nexus.types import AnyRun
from ess.reduce.unwrap import LookupTable, wavelength_spread

DISTANCES = np.array([10.0, 12.0, 14.0])


def _stddev(distance: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    # Linear in both arguments, so linear interpolation reproduces it exactly.
    return 0.01 + 0.02 * wavelength + 0.003 * distance


def _make_table(*, roll: int = 0) -> LookupTable:
    # Mean wavelength is proportional to event_time_offset / distance. `roll` shifts
    # the table along event_time_offset such that the mean wavelength is no longer
    # sorted, as in a table that wraps around at the pulse period.
    eto = np.linspace(0.0, 70.0, 71)
    mean = 4.0 * eto[np.newaxis, :] / DISTANCES[:, np.newaxis]
    mean = np.roll(mean, roll, axis=1)
    stddev = _stddev(DISTANCES[:, np.newaxis], mean)
    array = sc.DataArray(
        sc.array(
            dims=['distance', 'event_time_offset'],
            values=mean,
            variances=stddev**2,
            unit='angstrom',
        ),
        coords={
            'distance': sc.array(dims=['distance'], values=DISTANCES, unit='m'),
            'event_time_offset': sc.array(
                dims=['event_time_offset'], values=eto, unit='ms'
            ),
        },
    )
    return LookupTable[AnyRun, str](
        array=array,
        pulse_period=sc.scalar(70.0, unit='ms'),
        pulse_stride=1,
        distance_resolution=sc.scalar(2.0, unit='m'),
        time_resolution=sc.scalar(1.0, unit='ms'),
    )


@pytest.fixture
def ltotal() -> sc.Variable:
    return sc.array(dims=['x', 'y'], values=[[10.0, 11.3], [12.0, 14.0]], unit='m')


@pytest.fixture
def wavelength() -> sc.Variable:
    return sc.linspace('wavelength', 3.0, 19.0, num=9, unit='angstrom')


def _expected(ltotal: sc.Variable, wavelength: sc.Variable) -> np.ndarray:
    return _stddev(
        ltotal.values[..., np.newaxis], wavelength.values[np.newaxis, np.newaxis, :]
    )


@pytest.mark.parametrize('roll', [0, 17])
def test_wavelength_spread_interpolates_in_wavelength_and_distance(
    ltotal: sc.Variable, wavelength: sc.Variable, roll: int
) -> None:
    result = wavelength_spread(_make_table(roll=roll), ltotal, wavelength)
    assert result.dims == ('x', 'y', 'wavelength')
    assert result.unit == 'angstrom'
    np.testing.assert_allclose(result.values, _expected(ltotal, wavelength))


def test_wavelength_spread_converts_units(
    ltotal: sc.Variable, wavelength: sc.Variable
) -> None:
    result = wavelength_spread(
        _make_table(), ltotal.to(unit='mm'), wavelength.to(unit='nm')
    )
    assert result.unit == 'nm'
    np.testing.assert_allclose(
        result.to(unit='angstrom').values, _expected(ltotal, wavelength)
    )


def test_wavelength_spread_ignores_nan_entries(
    ltotal: sc.Variable, wavelength: sc.Variable
) -> None:
    table = _make_table()
    table.array.values[:, 50:] = np.nan
    table.array.variances[:, :3] = np.nan
    result = wavelength_spread(table, ltotal, wavelength)
    assert np.isfinite(result.values).all()
    # Wavelengths inside the finite range of all rows are unaffected
    inside = (wavelength > sc.scalar(1.2, unit='angstrom')) & (
        wavelength < sc.scalar(14.0, unit='angstrom')
    )
    np.testing.assert_allclose(
        result['wavelength', inside].values,
        _expected(ltotal, wavelength['wavelength', inside]),
    )


def test_wavelength_spread_uses_closest_wavelength_outside_row_range() -> None:
    ltotal = sc.array(dims=['x'], values=[10.0], unit='m')
    # Row at 10 m covers wavelengths [0, 28] angstrom
    wavelength = sc.array(dims=['wavelength'], values=[28.0, 50.0], unit='angstrom')
    result = wavelength_spread(_make_table(), ltotal, wavelength)
    assert result.values[0, 0] == result.values[0, 1]


@pytest.mark.parametrize('value', [9.9, 14.1])
def test_wavelength_spread_raises_if_ltotal_outside_table(
    wavelength: sc.Variable, value: float
) -> None:
    ltotal = sc.array(dims=['x'], values=[12.0, value], unit='m')
    with pytest.raises(ValueError, match='outside the distance range'):
        wavelength_spread(_make_table(), ltotal, wavelength)


def test_wavelength_spread_raises_without_variances(
    ltotal: sc.Variable, wavelength: sc.Variable
) -> None:
    table = _make_table()
    table.array.variances = None
    with pytest.raises(ValueError, match='no variances'):
        wavelength_spread(table, ltotal, wavelength)
