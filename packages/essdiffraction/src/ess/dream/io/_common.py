# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import scipp as sc
from scipp.core import irreducible_mask


def prepare_reduced_data(da: sc.DataArray) -> sc.DataArray:
    """Prepare reduced data for saving."""
    if da.ndim != 1:
        raise sc.DimensionError(f"Can only save 1D data, got {da.sizes}")

    hist = da.hist() if da.is_binned else da.copy(deep=False)
    hist.coords[hist.dim] = sc.midpoints(hist.coords[hist.dim])

    if (mask := irreducible_mask(hist.masks, hist.dim)) is not None:
        # No file format we use here supports masks, so we remove masked data points
        hist = hist[~mask]
        hist.masks.clear()

    return hist
