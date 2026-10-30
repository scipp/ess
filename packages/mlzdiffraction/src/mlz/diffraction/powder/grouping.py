# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Grouping and binning helpers for POWTEX powder reduction."""

from typing import cast

import numpy as np
import scipp as sc

from ..powtex.types import DspacingDetector, KeepEvents, WavelengthDetector
from .conversion import lambda_from_d_dperp
from .types import (
    DperpBins,
    Dspacing2D,
    DspacingBins,
    ExtendedData2D,
    NormalisedIntensity2D,
    RunType,
    SpectralDetector,
)


def flatten_to_spectra(data: WavelengthDetector[RunType]) -> SpectralDetector[RunType]:
    """Flatten detector bank dimensions to a single spectrum dimension."""
    if data is None:
        return cast(SpectralDetector[RunType], None)
    dims = [dim for dim in ("wire", "strip", "segment") if dim in data.dims]
    out = data.flatten(dims=dims, to="spectrum") if dims else data
    return SpectralDetector[RunType](out)


# TODO: Not used / needed anymore?
def bin_dspacing_dperp_mlz(
    data: SpectralDetector[RunType],
    dspacing_bins: DspacingBins,
    dperp_bins: DperpBins,
    keep_events: KeepEvents,
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
        # needs to be worked out
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
    keep_events: KeepEvents,
) -> Dspacing2D[RunType]:
    """Focus detector spectra to d-spacing and dperp."""
    if data is None:
        return cast(Dspacing2D[RunType], None)
    if data.bins is not None:
        for name in ("tof", "wavelength", "two_theta", "Ltotal"):
            if name in data.bins.coords:
                data = data.bins.drop_coords(name)
        data = data.bins.concat()
    args = {dperp_bins.dim: dperp_bins, dspacing_bins.dim: dspacing_bins}
    out = data.bin(args) if keep_events else data.hist(args)
    out.coords.pop("gd_prtn_chrg", None)
    return Dspacing2D[RunType](out)


def extend_data(data: NormalisedIntensity2D) -> ExtendedData2D:
    """Append wavelength, two-theta, dspacing, and dperp 2d grids to normalised intensity."""
    lambda_k = sc.scalar(1.0, unit="angstrom")
    d_centers = sc.midpoints(data.coords["dspacing"])
    dp_centers = sc.midpoints(data.coords["dperp"])

    lam_2d = lambda_from_d_dperp(d_centers, dp_centers, lambda_k).transpose()

    dspacing_2d = sc.broadcast(
        d_centers,
        dims=["dperp", "dspacing"],
        shape=[dp_centers.size, d_centers.size],
    )
    dperp_2d = sc.broadcast(
        dp_centers,
        dims=["dperp", "dspacing"],
        shape=[dp_centers.size, d_centers.size],
    )
    two_theta_2d = (2.0 * sc.asin(0.5 * lam_2d / dspacing_2d)).to(unit="deg")

    ds = sc.Dataset(
        coords={
            "dperp": data.coords["dperp"],
            "dspacing": data.coords["dspacing"],
        },
        data={
            "lambda": lam_2d.data if isinstance(lam_2d, sc.DataArray) else lam_2d,
            "two_theta": (
                two_theta_2d.data
                if isinstance(two_theta_2d, sc.DataArray)
                else two_theta_2d
            ),
            "dspacing_2d": (
                dspacing_2d.data
                if isinstance(dspacing_2d, sc.DataArray)
                else dspacing_2d
            ),
            "dperp_2d": (
                dperp_2d.data if isinstance(dperp_2d, sc.DataArray) else dperp_2d
            ),
            "intensity": data
        }
    )

    return ExtendedData2D(ds)


def sc_geomarange(dim, start, stop, relative_step, *, endpoint=False, unit=None):
    """Generates logarithmic bins using start, stop, and a relative step."""
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
        out = sc.concat(
            [
                out,
                sc.array(dims=[dim], values=[stop_var.value], unit=stop_var.unit),
            ],
            dim,
        )

    return out


providers = (
    extend_data,
    bin_dspacing_dperp_mlz,
    flatten_to_spectra,
    focus_to_dspacing_dperp,
)
