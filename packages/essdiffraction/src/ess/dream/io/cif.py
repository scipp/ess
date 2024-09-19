# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""CIF writer for DREAM."""

import scipp as sc
from scippneutron.io import cif

from ess.powder.types import CIFAuthors, IofDspacing, ReducedDspacingCIF


def prepare_reduced_dspacing_cif(
    da: IofDspacing, *, authors: CIFAuthors
) -> ReducedDspacingCIF:
    """Construct a CIF builder with reduced data in d-spacing.

    The object contains the d-spacing coordinate, intensities,
    and some metadata.

    Parameters
    ----------
    da:
        Reduced 1d data with a `'dspacing'` dimension and coordinate.
    authors:
        List of authors to write to the file.

    Returns
    -------
    :
        An object that contains the reduced data and metadata.
        Us its ``save`` method to write the CIF file.
    """
    from .. import __version__

    to_save = _prepare_data(da)
    return ReducedDspacingCIF(
        cif.CIF('reduced_dspacing')
        .with_reducers(f'ess.dream v{__version__}')
        .with_authors(*authors)
        .with_beamline(beamline='DREAM', facility='ESS')
        .with_reduced_powder_data(to_save)
    )


def _prepare_data(da: sc.DataArray) -> sc.DataArray:
    hist = da.copy(deep=False) if da.bins is None else da.hist()
    hist.coords[hist.dim] = sc.midpoints(hist.coords[hist.dim])
    return hist
