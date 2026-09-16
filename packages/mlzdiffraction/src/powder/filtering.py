# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Event filtering helpers for POWTEX powder reduction."""

from typing import cast

from .types import RunType, SpectralDetector, WavelengthDetector


def flatten_to_spectra(data: WavelengthDetector[RunType]) -> SpectralDetector[RunType]:
    """Flatten detector bank dimensions to a single spectrum dimension."""
    if data is None:
        return cast(SpectralDetector[RunType], None)
    dims = [dim for dim in ("wire", "strip", "segment") if dim in data.dims]
    out = data.flatten(dims=dims, to="spectrum") if dims else data
    return SpectralDetector[RunType](out)


providers = (flatten_to_spectra,)
