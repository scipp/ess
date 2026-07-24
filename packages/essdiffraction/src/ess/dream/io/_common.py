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
        # No file format we use here supports masks, so the next
        # best thing is to zero out masked data:
        if hist.variances is not None:
            replacement = sc.scalar(0.0, variance=0.0, unit=hist.unit, dtype=hist.dtype)
        else:
            replacement = sc.scalar(0.0, unit=hist.unit, dtype=hist.dtype)
        hist.data = sc.where(mask, replacement, hist.data)
        hist.masks.clear()

    return hist
