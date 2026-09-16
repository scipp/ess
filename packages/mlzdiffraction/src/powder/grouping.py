# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Grouping and binning helpers for POWTEX powder reduction."""

from typing import cast

import numpy as np
import scipp as sc

from ..powtex.types import DspacingDetector, KeepEventsMLZ
from .types import (
    DperpBins,
    Dspacing2D,
    DspacingBins,
    RunType,
    SpectralDetector,
)


# TODO: Not used / needed anymore?
def bin_dspacing_dperp_mlz(
    data: SpectralDetector[RunType],
    dspacing_bins: DspacingBins,
    dperp_bins: DperpBins,
    keep_events: KeepEventsMLZ,
) -> DspacingDetector[RunType]:
    """Bin detector data in d-spacing and dperp."""
    if data is None:
        return cast(DspacingDetector[RunType], None)
    args = {dperp_bins.dim: dperp_bins, dspacing_bins.dim: dspacing_bins}
    if keep_events:
        out = data.bin(args)
    else:
        # Reconstructing the wavelength results in an inconsistency if dspacing was
        # computed with a calibration table. Another option would be to use, e.g., the
        # mean wavelength in each bin, but this leads to random wavelength values that
        # break stream processing.
        out = data.hist(args)
        print("needs to be worked out")
        # result = data.hist(args).assign_coords(
        #     wavelength=_reconstruct_wavelength(
        #         dspacing_bins=dspacing_bins, two_theta_bins=twotheta_bins
        #     )
        # )
    if out.bins is not None and "tof" in out.bins.coords:
        out.data = out.data.bins.drop_coords("tof")
    return DspacingDetector[RunType](out)


def focus_to_dspacing_dperp(
    data: SpectralDetector[RunType],
    dspacing_bins: DspacingBins,
    dperp_bins: DperpBins,
    keep_events: KeepEventsMLZ,
) -> Dspacing2D[RunType]:
    """Focus detector spectra to d-spacing and dperp."""
    if data is None:
        return cast(Dspacing2D[RunType], None)
    if data.bins is not None:
        for name in ("tof", "wavelength", "two_theta", "Ltotal"):
            if name in data.bins.coords:
                data = data.bins.drop_coords(name)
        data = data.bins.concat()
    args = {dspacing_bins.dim: dspacing_bins, dperp_bins.dim: dperp_bins}
    out = data.bin(args) if keep_events else data.hist(args)
    out.coords.pop("gd_prtn_chrg", None)
    return Dspacing2D[RunType](out)


def sc_geomarange(dim, start, stop, relative_step, *, endpoint=False, unit=None):
    """Geometric range as a Scipp variable using a relative step."""
    ratio = 1.0 + relative_step
    if ratio <= 0.0:
        raise ValueError("1 + relative_step must be positive")
    if ratio == 1.0:
        raise ValueError("relative_step must not be 0")

    start_var = start if isinstance(start, sc.Variable) else sc.scalar(start, unit=unit)
    stop_var = stop if isinstance(stop, sc.Variable) else sc.scalar(stop, unit=unit)

    if start_var.value <= 0 or stop_var.value <= 0:
        raise ValueError("start and stop must be positive")

    n = int(np.floor(np.log(stop_var.value / start_var.value) / np.log(ratio))) + 1
    values = start_var.value * ratio ** np.arange(n)
    out = sc.array(dims=[dim], values=values, unit=start_var.unit)

    if endpoint and not np.isclose(out.values[-1], stop_var.value):
        out = sc.concat([out, sc.array(dims=[dim], values=[stop_var.value], unit=stop_var.unit)], dim)

    return out


providers = (
    bin_dspacing_dperp_mlz,
    focus_to_dspacing_dperp,
)
