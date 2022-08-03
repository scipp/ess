# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import warnings
import scipp as sc
from ess.reflectometry import corrections
from orsopy import fileio


def test_normalize_by_counts():
    """
    Tests the corrections.normalize_by_counts function without
    a orsopy object present.
    """
    N = 50
    values = [1.] * N
    data = sc.Variable(dims=['position'],
                       unit=sc.units.counts,
                       values=values,
                       variances=values)
    array = sc.DataArray(data=data)
    array_normalized = corrections.normalize_by_counts(array)
    result = sc.DataArray(data=sc.Variable(dims=['position'],
                                           unit=sc.units.dimensionless,
                                           values=[1 / N] * N,
                                           variances=[1 / (N * N) + 1 / (N * N * N)] *
                                           N))
    assert sc.allclose(array_normalized.data, result.data)


def test_normalize_by_counts_orso():
    """
    Tests the corrections.normalize_by_counts function
    with a orsopy object present.
    """
    N = 50
    values = [1.] * N
    data = sc.Variable(dims=['position'],
                       unit=sc.units.counts,
                       values=values,
                       variances=values)
    array = sc.DataArray(data=data, attrs={'orso': sc.scalar(fileio.orso.Orso.empty())})
    array.attrs['orso'].value.reduction.corrections = []
    array_normalized = corrections.normalize_by_counts(array)
    result = sc.DataArray(data=sc.Variable(dims=['position'],
                                           unit=sc.units.dimensionless,
                                           values=[1 / N] * N,
                                           variances=[1 / (N * N) + 1 / (N * N * N)] *
                                           N))
    assert sc.allclose(array_normalized.data, result.data)
    assert 'total counts' in array.attrs['orso'].value.reduction.corrections


def test_beam_on_sample():
    """
    Tests the corrections.beam_on_sample function.
    """
    beam_size = sc.scalar(1., unit=sc.units.mm)
    theta = sc.scalar(0.1, unit=sc.units.rad)
    expected_result = sc.scalar(10.01668613, unit=sc.units.mm)
    assert sc.allclose(expected_result, corrections.beam_on_sample(beam_size, theta))


def test_beam_on_sample_array():
    """
    Tests the corrections.beam_on_sample function with an array of theta.
    """
    beam_size = sc.scalar(1., unit=sc.units.mm)
    theta = sc.array(dims=['x'], values=[0.1, 0.5], unit=sc.units.rad)
    expected_result = sc.array(dims=['x'],
                               values=[10.01668613, 2.085829643],
                               unit=sc.units.mm)
    assert sc.allclose(expected_result, corrections.beam_on_sample(beam_size, theta))
