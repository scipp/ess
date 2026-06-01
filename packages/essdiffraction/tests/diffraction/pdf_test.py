# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scipp.testing
from ess.diffraction.pdf import (
    _covariance_of_matrix_vector_product,
    linearized_radial_distribution_function,
    pair_correlation_function,
    pair_distribution_function,
    radial_distribution_function,
    running_coordination_number,
)


def test_pdf_structure_factor_needs_q_coord() -> None:
    da = sc.DataArray(sc.ones(sizes={'Q': 3}))
    r = sc.array(dims='r', values=[2, 3, 4, 5.0])

    with pytest.raises(KeyError):
        pair_correlation_function(
            da,
            r,
        )


def test_pair_correlation_function() -> None:
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')

    G: sc.DataArray = pair_correlation_function(  # type: ignore[assignment]
        da,
        r,
    )
    assert G.data.unit == '1/angstrom^2'
    sc.testing.assert_identical(G.coords['r'], r)


def test_pair_correlation_function_can_return_covariances() -> None:
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')

    [G, cov] = pair_correlation_function(da, r, return_covariances=True)
    assert G.data.unit == '1/angstrom^2'
    assert cov.data.unit == '1/angstrom^4'
    assert cov.ndim == 2


def test_pair_correlation_function_result_unchanged() -> None:
    # Note: bogus data
    da = sc.DataArray(
        sc.array(dims='Q', values=[1, 2, 4.0]),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')

    G: sc.DataArray = pair_correlation_function(  # type: ignore[assignment]
        da,
        r,
    )
    sc.testing.assert_allclose(
        G.data,
        sc.array(dims='r', values=[-0.616322, 1.51907, -3.11757], unit='1/angstrom^2'),
        rtol=sc.scalar(1e-5),
    )


def test_pair_distribution_function() -> None:
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    atomic_density = sc.scalar(1.0, unit='1/angstrom^3')

    G: sc.DataArray = pair_correlation_function(da, r)  # type: ignore[assignment]
    g = pair_distribution_function(G, atomic_density=atomic_density)
    assert g.data.unit == '1'
    sc.testing.assert_identical(G.coords['r'], r)


def test_radial_distribution_function() -> None:
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    atomic_density = sc.scalar(1.0, unit='1/angstrom^3')

    G: sc.DataArray = pair_correlation_function(da, r)  # type: ignore[assignment]
    g = pair_distribution_function(G, atomic_density=atomic_density)
    rdf = radial_distribution_function(g, atomic_density=atomic_density)
    assert rdf.data.unit == '1/angstrom'
    sc.testing.assert_identical(G.coords['r'], r)


def test_linearized_radial_distribution_function() -> None:
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    atomic_density = sc.scalar(1.0, unit='1/angstrom^3')

    G: sc.DataArray = pair_correlation_function(da, r)  # type: ignore[assignment]
    g = pair_distribution_function(G, atomic_density=atomic_density)
    rdf = radial_distribution_function(g, atomic_density=atomic_density)
    T = linearized_radial_distribution_function(rdf)
    assert T.data.unit == '1/angstrom^2'
    sc.testing.assert_identical(T.coords['r'], r)


def test_running_coordination_number() -> None:
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 4, 5, 6.0], unit='angstrom')
    atomic_density = sc.scalar(1.0, unit='1/angstrom^3')

    G: sc.DataArray = pair_correlation_function(da, r)  # type: ignore[assignment]
    g = pair_distribution_function(G, atomic_density=atomic_density)
    rdf = radial_distribution_function(g, atomic_density=atomic_density)
    C = running_coordination_number(rdf)
    assert C.data.unit == '1'
    sc.testing.assert_identical(C.coords['r'], r)
    sc.testing.assert_identical(C[0].value, rdf[0].value * 2)  # *2 from bin size
    sc.testing.assert_identical(C[1].value, rdf[0].value * 2 + rdf[1].value * 1)


def test_matrix_vector_covariance() -> None:
    A = sc.array(dims='ab', values=[[1, 2], [3, 4]])
    v = sc.array(dims='b', values=[0, 0.0], variances=[1.0, 2.0])
    C = _covariance_of_matrix_vector_product(A, v)
    a00, a01, a10, a11 = A.values.ravel()
    v0, v1 = v.variances
    sc.testing.assert_allclose(
        C,
        sc.array(
            dims=['a', 'a_2'],
            values=[
                [
                    v0 * a00 * a00 + v1 * a01 * a01,
                    v0 * a00 * a10 + v1 * a01 * a11,
                ],
                [
                    v0 * a10 * a00 + v1 * a11 * a01,
                    v0 * a10 * a10 + v1 * a11 * a11,
                ],
            ],
        ),
    )
