# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Detector diagnostics for DREAM."""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import scipp as sc

from ess.reduce.diagnostics.flat_bank_viewer import FlatBankViewer


class FlatVoxelViewer(FlatBankViewer):
    """Interactive 2D plot of all detector voxels.

    See `DREAM Detector Diagnostics
    <../../user-guide/dream/dream-detector-diagnostics.rst>`_ for explanations
    and an example.
    """

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
        vertical = ('counter', 'segment', 'wire')
        horizontal = ('module', 'strip', 'sumo', 'sector')
        super().__init__(
            _prepare_data(data),
            dims=(*vertical, *horizontal),
            sliced=(),
            vertical=vertical,
            horizontal=horizontal,
            cbar=cbar,
            rasterized=rasterized,
            **kwargs,
        )


def _prepare_data(
    data: Mapping[str, sc.DataArray] | sc.DataArray,
) -> Mapping[str, sc.DataArray] | sc.DataArray:
    if isinstance(data, sc.DataArray):
        return data

    # Sort to get a consistent order
    indices = defaultdict(
        lambda: 10,
        mantle=0,
        endcap_forward=1,
        endcap_backward=2,
        high_resolution=3,
        sans=4,
    )
    return dict(sorted(data.items(), key=lambda t: indices[t[0]]))
