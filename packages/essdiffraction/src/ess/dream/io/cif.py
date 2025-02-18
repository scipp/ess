# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""CIF writer for DREAM."""

import scipp as sc
from scippneutron.io import cif

from ess.powder.calibration import OutputCalibrationData
from ess.powder.types import (
    Beamline,
    CIFAuthors,
    IofTof,
    ReducedTofCIF,
    ReducerSoftwares,
    Source,
)


def prepare_reduced_tof_cif(
    da: IofTof,
    *,
    authors: CIFAuthors,
    beamline: Beamline,
    source: Source,
    reducers: ReducerSoftwares,
    calibration: OutputCalibrationData,
) -> ReducedTofCIF:
    """Construct a CIF builder with reduced data in d-spacing.

    The object contains the d-spacing coordinate, intensities,
    and some metadata.

    Parameters
    ----------
    da:
        Reduced 1d data with a ``'tof'`` dimension and coordinate.
    authors:
        List of authors to write to the file.
    beamline:
        Information about the beamline that the data was produced at.
    source:
        Information about the neutron source.
    reducers:
        List of software pieces used to reduce the data.
    calibration:
        Coefficients for conversion between d-spacing and final ToF.
        See :meth:`scippneutron.io.cif.CIF.with_powder_calibration`.

    Returns
    -------
    :
        An object that contains the reduced data and metadata.
        Us its ``save`` method to write the CIF file.
    """
    to_save = _prepare_data(da)
    return ReducedTofCIF(
        cif.CIF('reduced_tof')
        .with_reducers(*(reducer.compact_repr for reducer in reducers))
        .with_authors(*authors)
        .with_beamline(beamline, source)
        .with_powder_calibration(calibration.to_cif_format())
        .with_reduced_powder_data(to_save)
    )


def _prepare_data(da: sc.DataArray) -> sc.DataArray:
    hist = da.copy(deep=False) if da.bins is None else da.hist()
    hist.coords[hist.dim] = sc.midpoints(hist.coords[hist.dim])
    return hist
