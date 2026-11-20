# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Flat detector viewer for BIFROST."""

from collections.abc import Iterable, Mapping
from typing import Any

import scipp as sc

from ess.reduce.diagnostics.flat_data_viewer import FlatDataViewer


class FlatDetectorViewer(FlatDataViewer):
    """2D plot of all detector pixels."""

    def __init__(
        self,
        data: Mapping[str, sc.DataArray] | sc.DataArray,
        horizontal: Iterable[str] | dict[str, Iterable[str]] = ('channel', 'length'),
        vertical: Iterable[str] | dict[str, Iterable[str]] = ('arc', 'tube'),
        *,
        cbar: bool = True,
        rasterized: bool = True,
        separators: str | None = None,
        **fig_kwargs: Any,
    ) -> None:
        """Create a new viewer.

        Parameters
        ----------
        data:
            Data to show. A mapping (e.g., a data group) with one entry per bank
            adds buttons for selecting a bank. The banks may have different dims
            but must all have the same unit and no empty dimensions.
        horizontal:
            Dims to display on the horizontal axis.
            Order is outermost to innermost (last dim varies fastest).
            If a dict, the keys are bank names and the values are dims to display
            on the horizontal axis for that bank.
        vertical:
            Dims to display on the vertical axis.
            Order is outermost to innermost (last dim varies fastest).
            If a dict, the keys are bank names and the values are dims to display
            on the vertical axis for that bank.
        cbar:
            Whether to show a colorbar.
        rasterized:
            If ``True``, the figure is rasterized which improves rendering
            speed but reduces resolution.
        separators:
            If not ``None``, draw lines along the bin edges of the slowest
            (first-listed) dim on each axis. The value of this argument is the
            line color.
        **fig_kwargs:
            Additional arguments passed to :func:`plopp.imagefigure`.
        """
        super().__init__(
            data,
            vertical=vertical,
            horizontal=horizontal,
            cbar=cbar,
            rasterized=rasterized,
            separators=separators,
            **fig_kwargs,
        )
