# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import os
from typing import Iterable, TextIO, Union

import numpy as np
import scipp as sc
from orsopy.fileio.base import Column
from orsopy.fileio.orso import Orso, OrsoDataset

from .orso import OrsoDataSource, OrsoReduction
from .types import NormalizedIofQ


def save_ort(
    *,
    filename: Union[str, os.PathLike, TextIO],
    data: Iterable[sc.Variable],
    orso: Orso,
) -> None:
    """
    Save a data with the ORSO .ort file format.

    Parameters
    ----------
    filename:
        The file or filename to save to.
    data:
        Columns to save.
        Must match the columns stored in ``orso`.columns``.
    orso:
        ORSO object holding the metadata for ``data``.
    """
    _require_ort_suffix(filename)
    orso = _validate_orso_header(orso)
    dataset = OrsoDataset(
        orso, np.column_stack([_extract_values_array(d) for d in data])
    )
    dataset.save(filename)


def save_iofq_ort(
    filename: Union[str, os.PathLike, TextIO],
    iofq: NormalizedIofQ,
    data_source: OrsoDataSource,
    reduction: OrsoReduction,
) -> None:
    """Save reduced I-of-Q data to an ORSO .ort file.

    Parameters
    ----------
    filename:
        The file or filename to save to.
    iofq:
        Reduced :math:`I(Q)`.
    data_source:
        ORSO ``DataSource`` with metadata about the data that
        ``iofq`` was computed from.
    reduction:
        ORSO ``Reduction`` with metadata about the reduction process.
    """
    orso = Orso(
        data_source=data_source,
        reduction=reduction,
        columns=[
            Column('Qz', '1/angstrom', 'wavevector transfer'),
            Column('R', None, 'reflectivity'),
            Column('sR', None, 'standard deviation of reflectivity'),
            Column(
                'sQz',
                '1/angstrom',
                'standard deviation of wavevector transfer resolution',
            ),
        ],
    )

    qz = iofq.coords['Q'].to(unit='1/angstrom', copy=False)
    if iofq.coords.is_edges('Q'):
        qz = sc.midpoints(qz)
    r = sc.values(iofq.data)
    sr = sc.stddevs(iofq.data)
    sqz = iofq.coords['sigma_Q'].to(unit='1/angstrom', copy=False)

    save_ort(filename=filename, data=(qz, r, sr, sqz), orso=orso)


def _require_ort_suffix(filename: Union[str, os.PathLike, TextIO]) -> None:
    try:
        path = os.fspath(filename)
        if not path.endswith('.ort'):
            raise ValueError("The output file must have the suffix '.ort'")
    except TypeError:
        return  # Cannot check suffix of TextIO object, assume it is correct.


def _validate_orso_header(orso: Orso) -> Orso:
    return Orso(**orso.to_dict())


def _extract_values_array(var: sc.Variable) -> np.ndarray:
    if var.variances is not None:
        raise sc.VariancesError(
            "ORT columns must not have variances. "
            "Store the uncertainties as standard deviations in a separate column."
        )
    if var.ndim != 1:
        raise sc.DimensionError(f"ORT columns must be one-dimensional, got {var.sizes}")
    return var.values
