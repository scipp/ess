# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ..amor.tools import fwhm_to_std
from . import orso


def footprint_correction(data_array: sc.DataArray) -> sc.DataArray:
    """
    Perform the footprint correction on the data array that has a :code:`beam_size` and
    binned :code:`theta` values.

    Parameters
    ----------
    data_array:
        Data array to perform footprint correction on.

    Returns
    -------
    :
       Footprint corrected data array.
    """
    size_of_beam_on_sample = beam_on_sample(data_array.coords['beam_size'],
                                            data_array.bins.coords['theta'])
    footprint_scale = sc.erf(
        fwhm_to_std(data_array.coords['sample_size'] / size_of_beam_on_sample))
    data_array_fp_correction = data_array / footprint_scale.squeeze()
    try:
        data_array_fp_correction.attrs['orso'].value.reduction.corrections += [
            'footprint correction'
        ]
    except KeyError:
        orso.not_found_warning()
    return data_array_fp_correction


def normalize_by_counts(data_array: sc.DataArray) -> sc.DataArray:
    """
    Normalize the bin-summed data by the total number of counts.

    Parameters
    ----------
    data_array:
        Data array to be normalized.

    Returns
    -------
    :
        Normalized data array.
    """
    # Dividing by ncounts fails because ncounts also has variances, and this introduces
    # correlations. According to Heybrock et al. (2023), we can however safely drop the
    # variances of ncounts if counts_in_bin / ncounts is small everywhere.
    ncounts = sc.values(data_array.sum())
    norm = data_array / ncounts
    if norm.max().value > 0.1:
        ind = np.argmax(data_array.values)
        raise ValueError(
            'One or more bins contain a number of counts of the same order as the '
            'total number of counts. It is not safe to drop the variances of the '
            'denominator when normalizing by the total number of counts in this '
            f'regime. The maximum counts found is {data_array.values[ind]} at '
            f'index {ind}. The total number of counts is {ncounts.value}.')
    try:
        norm.attrs['orso'].value.reduction.corrections += ['total counts']
    except KeyError:
        orso.not_found_warning()
    return norm


def beam_on_sample(beam_size: sc.Variable, theta: sc.Variable) -> sc.Variable:
    """
    Size of the beam on the sample.

    Parameters
    ----------
    beam_size:
        Full width half maximum of the beam.
    theta:
        Angular of incidence with the sample.

    Returns
    -------
    :
        Size of the beam on the sample.
    """
    return beam_size / sc.sin(theta)
