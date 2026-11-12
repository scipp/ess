# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Flat detector viewer for BIFROST."""

from collections.abc import Mapping
from typing import Any

import scipp as sc

from ess.reduce.diagnostics.flat_bank_viewer import FlatBankViewer


class FlatDetectorViewer(FlatBankViewer):
    """Interactive 2D plot of all detector pixels."""

    def __init__(
        self,
        data: Mapping[str, sc.DataArray] | sc.DataArray,
        *,
        cbar: bool = True,
        rasterized: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new viewer.

        Parameters
        ----------
        data:
            Histogrammed data, one entry per bank.
            cbar:
                Whether to show a colorbar.
        rasterized:
            If ``True``, the figure is rasterized which improves rendering
            speed but reduces resolution.
        **kwargs:
            Additional arguments passed to :func:`plopp.imagefigure`.
        """
        vertical = ("arc", "tube")
        horizontal = ("channel", "length")
        sliced = set(data.dims) - set(vertical) - set(horizontal)
        super().__init__(
            data,
            dims=(*vertical, *horizontal, *sliced),
            sliced=sliced,
            vertical=vertical,
            horizontal=horizontal,
            cbar=cbar,
            rasterized=rasterized,
            **kwargs,
        )
