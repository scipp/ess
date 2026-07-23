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

    if hist.masks:
        # No file format we use here supports masks, so the next
        # best thing is to zero out masked data:
        hist.data = hist.data.copy()
        hist.values *= irreducible_mask(hist.masks, hist.dim).values
        hist.masks.clear()

    return hist
