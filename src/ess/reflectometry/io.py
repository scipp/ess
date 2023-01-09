# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc


def save_ort(data_array: sc.DataArray, filename: str, dimension: str = None):
    """
    Save a data array with the ORSO .ort file format.

    Parameters
    ----------
    data_array:
        Scipp-data array to save.
    filename:
        Filename.
    dimension:
        String for dimension to perform mean over, defaults to 'detector_id'.
    """
    if dimension is None:
        dimension = 'detector_id'
    from orsopy import fileio
    if filename[:-4] == '.ort':
        raise ValueError("The expected output file ending is .ort.")
    q = data_array.mean(dimension).coords['Q']
    if data_array.mean(dimension).coords.is_edges('Q'):
        q = sc.midpoints(q)
    R = data_array.mean(dimension).data
    sR = sc.stddevs(data_array.mean(dimension).data)
    sq = data_array.coords['sigma_Q']
    dataset = fileio.orso.OrsoDataset(
        data_array.attrs['orso'].value,
        np.array([q.values, R.values, sR.values, sq.values]).T)
    fileio.orso.save_orso([dataset], filename)
