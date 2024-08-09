# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scipp.testing

from ess.powder.transform import (
    _covariance_of_matrix_vector_product,
    compute_pdf_from_structure_factor,
)


def test_pdf_structure_factor_needs_q_coord():
    da = sc.DataArray(sc.ones(sizes={'Q': 3}))
    r = sc.array(dims='r', values=[2, 3, 4, 5.0])
    with pytest.raises(KeyError):
        compute_pdf_from_structure_factor(
            da,
            r,
        )


def test_pdf_structure_factor():
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    v = compute_pdf_from_structure_factor(
        da,
        r,
    )
    assert v.data.unit == '1/angstrom^2'
    sc.testing.assert_identical(v.coords['r'], r)


def test_pdf_structure_factor_can_return_covariances():
    da = sc.DataArray(
        sc.ones(sizes={'Q': 3}),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    da.variances = da.data.values.copy()
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    _, cov = compute_pdf_from_structure_factor(da, r, return_covariances=True)


def test_pdf_structure_factor_result_unchanged():
    # Note: bogus data
    da = sc.DataArray(
        sc.array(dims='Q', values=[1, 2, 4.0]),
        coords={'Q': sc.array(dims='Q', values=[0, 1, 2, 3.0], unit='1/angstrom')},
    )
    r = sc.array(dims='r', values=[2, 3, 4, 5.0], unit='angstrom')
    v = compute_pdf_from_structure_factor(
        da,
        r,
    )
    sc.testing.assert_allclose(
        v.data,
        sc.array(dims='r', values=[-0.616322, 1.51907, -3.11757], unit='1/angstrom^2'),
        rtol=sc.scalar(1e-5),
    )


def test_matrix_vector_covariance():
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
