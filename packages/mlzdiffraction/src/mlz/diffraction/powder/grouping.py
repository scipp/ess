# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Grouping and binning helpers for POWTEX powder reduction."""

from typing import cast

import scipp as sc

from ..powtex.types import DspacingDetector, KeepEvents, WavelengthDetector
from .conversion import lambda_from_d_dperp
from .types import (
    DperpBins,
    Dspacing2D,
    DspacingBins,
    ExtendedData2D,
    Intensity2D,
    IntensityP2D,
    MaskedData2D,
    NormalisedIntensity2D,
    RunType,
    SpectralDetector,
)


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


def _calculate_combined_mask(ds: sc.Dataset, target_dims: tuple[str, ...]):
    combined = None
    for var_name, da in ds.items():
        for mask_name, mask in da.masks.items():
            if set(mask.dims) != set(target_dims):
                raise ValueError(
                    f"Mask {mask_name!r} from {var_name!r} has dims {mask.dims}, "
                    f"but target has dims {target_dims}."
                )
            if mask.dims != target_dims:
                mask = mask.transpose(target_dims)
            combined = mask.copy() if combined is None else combined | mask
    return combined


def extract_intensity_2d(
    data: MaskedData2D,
) -> Intensity2D:
    """Extract intensity and attach a combined mask."""
    out = data["intensity"].copy()
    combined_mask = _calculate_combined_mask(data, out.dims)
    if combined_mask is not None:
        # remove intensity masks if any, as they are included in the combined mask
        out.masks.clear()
        out.masks["combined_mask"] = combined_mask
    return Intensity2D(out)


def prepare_for_p2d_export(
    data2d: MaskedData2D,
) -> IntensityP2D:
    """Flatten masked 2-D intensity into P2D export rows."""
    target_dims = data2d["intensity"].dims
    combined_mask = _calculate_combined_mask(data2d, target_dims)
    if combined_mask is None:
        combined_mask = sc.zeros(sizes=data2d["intensity"].sizes, dtype="bool")
    out = sc.DataArray(
        data=data2d["intensity"].data,
        coords={
            "dperp": data2d["dperp_2d"].data,
            "dspacing": data2d["dspacing_2d"].data,
            "lambda": data2d["lambda"].data,
            "two_theta": data2d["two_theta"].data,
        },
        masks={"combined_mask": combined_mask},
    )
    out = out.flatten(dims=out.dims, to="d_pixel")
    valid = ~out.masks["combined_mask"]
    # apply mask and drop it from the output afterwards
    out = out[valid]
    out.masks.clear()
    return IntensityP2D(out)


providers = (
    extend_data,
    extract_intensity_2d,
    focus_to_dspacing_dperp,
    prepare_for_p2d_export
)
