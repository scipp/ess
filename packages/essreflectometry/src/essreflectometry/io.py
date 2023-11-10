# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional

import numpy as np
import scipp as sc


def save_ort(
    data_array: sc.DataArray, filename: str, dimension: Optional[str] = None
) -> None:
    """
    Save a data array with the ORSO .ort file format.

    Parameters
    ----------
    data_array:
        Scipp-data array to save.
    filename:
        Filename.
    dimension:
        String for dimension to perform mean over.
    """
    from orsopy import fileio

    if filename[:-4] == '.ort':
        raise ValueError("The expected output file ending is .ort.")
    if dimension is not None:
        data_array = data_array.mean(dimension)
    q = data_array.coords['Q']
    if data_array.coords.is_edges('Q'):
        q = sc.midpoints(q)
    R = data_array.data
    sR = sc.stddevs(data_array.data)
    sq = data_array.coords['sigma_Q']
    dataset = fileio.orso.OrsoDataset(
        data_array.attrs['orso'].value,
        np.array([q.values, R.values, sR.values, sq.values]).T,
    )
    fileio.orso.save_orso([dataset], filename)
