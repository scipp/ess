# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Instrument view for LOKI."""

import warnings

import scipp as sc
from plopp.core.typing import FigureLike
from scippneutron import instrument_view


def InstrumentView(
    data: sc.DataArray | sc.DataGroup | dict,
    dim: str | None = None,
    pixel_size: float | sc.Variable | None = None,
    **kwargs,
) -> FigureLike:
    """Deprecated alias for :func:`ess.loki.instrument_view`.

    .. deprecated:: 26.10.0
        Use :func:`ess.loki.instrument_view` instead.

    Additional arguments are forwarded to :func:`scippneutron.instrument_view`.
    """
    warnings.warn(
        'ess.loki.InstrumentView is deprecated; use ess.loki.instrument_view instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument_view(data, dim=dim, pixel_size=pixel_size, **kwargs)
