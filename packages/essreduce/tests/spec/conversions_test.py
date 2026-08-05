# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.reduce.spec.conversions import edges_to_variable, range_to_variables
from ess.reduce.spec.parameters import (
    Scale,
    TOARange,
    WavelengthEdges,
)


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
