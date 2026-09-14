# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.reduce.spec import ArraySpec
from ess.reduce.spec.conversions import (
    check_array,
    edges_to_variable,
    range_to_variables,
)
from ess.reduce.spec.parameters import (
    Scale,
    TOARange,
    WavelengthEdges,
)


class TestCheckArray:
    @pytest.fixture
    def iofq(self) -> sc.DataArray:
        return sc.DataArray(
            sc.ones(dims=['Q'], shape=[3], unit='counts'),
            coords={'Q': sc.linspace('Q', 0.0, 1.0, 4, unit='1/Å')},
        )

    def test_matching_array_passes(self, iofq: sc.DataArray) -> None:
        check_array(iofq, ArraySpec(dims=('Q',), unit='counts', coords={'Q': '1/Å'}))

    def test_variable_matches_spec_without_coords(self) -> None:
        check_array(sc.scalar(1.0, unit='m'), ArraySpec(dims=(), unit='m'))

    def test_no_unit_is_distinct_from_dimensionless(self) -> None:
        check_array(sc.scalar('a', unit=None), ArraySpec(dims=()))
        check_array(sc.scalar(1.0), ArraySpec(dims=(), unit='dimensionless'))
        with pytest.raises(ValueError, match='unit'):
            check_array(sc.scalar(1.0), ArraySpec(dims=()))

    @pytest.mark.parametrize(
        'spec',
        [
            ArraySpec(dims=('x',), unit='counts'),
            ArraySpec(dims=('Q',), unit='m'),
            ArraySpec(dims=('Q',), unit='counts', coords={'wavelength': 'Å'}),
            ArraySpec(dims=('Q',), unit='counts', coords={'Q': 'nm'}),
            ArraySpec(dims=('Q',), unit='counts', binned=True),
        ],
    )
    def test_mismatch_raises(self, iofq: sc.DataArray, spec: ArraySpec) -> None:
        with pytest.raises(ValueError, match='does not match'):
            check_array(iofq, spec)

    def test_binned_data_matches_binned_spec(self) -> None:
        events = sc.data.binned_x(nevent=10, nbin=2)
        check_array(events, ArraySpec(dims=('x',), unit='K', binned=True))
        with pytest.raises(ValueError, match='binned'):
            check_array(events, ArraySpec(dims=('x',), unit='K'))


def test_linear_edges() -> None:
    edges = WavelengthEdges(start=1.0, stop=10.0, num_bins=9)
    var = edges_to_variable(edges, dim='wavelength')
    assert sc.identical(
        var, sc.linspace('wavelength', start=1.0, stop=10.0, num=10, unit='Å')
    )


def test_log_edges() -> None:
    edges = WavelengthEdges(start=1.0, stop=100.0, num_bins=2, scale=Scale.LOG)
    var = edges_to_variable(edges, dim='wavelength')
    assert sc.identical(
        var, sc.geomspace('wavelength', start=1.0, stop=100.0, num=3, unit='Å')
    )


def test_range_to_variables() -> None:
    low, high = range_to_variables(TOARange(start=10.0, stop=20.0))
    assert sc.identical(low, sc.scalar(10.0, unit='µs'))
    assert sc.identical(high, sc.scalar(20.0, unit='µs'))
