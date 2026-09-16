# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Data loading and P2D export helpers for POWTEX powder reduction."""

from pathlib import Path
from typing import cast

import pandas as pd
import scipp as sc
import scippnexus as snx

from .types import (
    AccumulatedProtonCharge,
    CalibrationData,
    CalibrationRun,
    CorrectedDspacing,
    DSpacingMask,
    DataFrameP2D,
    DperpMask,
    FilenameMLZ,
    FilepathMLZ,
    IntensityP2D,
    MaskedExtendedIntensity2D,
    MaskedIntensity2D,
    MaskingData,
    Position,
    P2DOutputFilename,
    RawDataAndMetadata,
    RunType,
    TofDetector,
    TofMask,
    TwoThetaMask,
    WavelengthMask,
)


def load_powgen_mlz(
    filename: FilenameMLZ[RunType],
    datapath: FilepathMLZ[RunType]
) -> TofDetector[RunType]:
    """
    Loads a powgen nxs file.

    The loaded data holds both the events and useful metadata from the file.
    This loader can be applied to already preprocessed POWGEN event files,
    not the raw files from the instrument. To generate an appropriate file,
    the following steps should be taken:
    - use convert_to_scipp_nxs() defined in utils.py to extract appropriate
    fields from the original NeXus files.
    - the appropriate fields (including 'positions') are oextracted from
    the outcome of scn.load_with_mantid().
    """
    if filename is None:
        return cast(TofDetector[RunType], None)
    filepath = Path(datapath) / filename
    dg = sc.io.load_hdf5(filepath)
    out = dg["data"].squeeze()
    out.coords.pop("tof", None)
    out = out.fold(dim="spectrum", sizes={"wire": 40, "strip": 154, "segment": 7})
    if out.bins is not None and "pulse_time" in out.data.bins.coords:
        out.data = out.data.bins.drop_coords("pulse_time")
    return TofDetector[RunType](out)


def load_powgen_calibration_mlz(
    filename: FilenameMLZ[CalibrationRun], datapath: FilepathMLZ[CalibrationRun]
) -> CalibrationData:
    """Load a diamond calibration file."""
    if filename is None:
        return CalibrationData(None)
    filepath = Path(datapath) / filename
    ds = sc.io.load_hdf5(filepath)
    # print(f"Loading calibration data from {filepath}")
    out = sc.Dataset(
        {
            key: da.fold(
                dim="spectrum", sizes={"wire": 40, "strip": 154, "segment": 7}
            )
            for key, da in ds.items()
        }
    )
    return CalibrationData(out)


def load_powgen_tof_mask_mlz() -> MaskingData[TofMask]:
    """Return the time-of-flight mask for POWGEN P2D reduction."""
    mask_function = None
    #mask_function = lambda x: (x < sc.scalar(0.0, unit="us")) | (x > sc.scalar(16666.67, unit="us"))
    return MaskingData[TofMask](mask_function)


def load_powgen_wavelength_mask_mlz() -> MaskingData[WavelengthMask]:
    """Return the wavelength mask for POWGEN P2D reduction."""
    return MaskingData[WavelengthMask](
        lambda x: (
            x <= sc.scalar(0.3, unit="angstrom")
        ) | (
            x >= sc.scalar(1.25, unit="angstrom")
        ) | (
            (x >= sc.scalar(1.03, unit="angstrom")) & (x <= sc.scalar(1.07, unit="angstrom"))
        )
    )


def load_powgen_two_theta_mask_mlz() -> MaskingData[TwoThetaMask]:
    """Return the two-theta mask for POWGEN P2D reduction."""

    def deg(value: float) -> sc.Variable:
        return sc.scalar(value, unit="degree")

    return MaskingData[TwoThetaMask](
        lambda x: (
            x < deg(10.0) # extra cut as in powtex
        ) | (
            x > deg(150.0)
        ) | (
            (x > deg(13.6)) & (x < deg(16.6))
        ) | (
            (x > deg(23.6)) & (x < deg(26.6))
        ) | (
            (x > deg(34.3)) & (x < deg(37.3))
        ) | (
            (x > deg(45.2)) & (x < deg(48.2))
        ) | (
            (x > deg(56.0)) & (x < deg(61.0))
        ) | (
            (x > deg(69.8)) & (x < deg(72.8))
        ) | (
            (x > deg(83.5)) & (x < deg(86.5))
        ) | (
            (x > deg(97.1)) & (x < deg(101.1))
        ) | (
            (x > deg(113.0)) & (x < deg(116.0))
        ) | (
            (x > deg(131.0)) & (x < deg(134.0))
        ) | (
            (x > deg(146.0)) & (x < deg(150.0))
        )
    )


def load_powgen_dspacing_mask_mlz() -> MaskingData[DSpacingMask]:
    """Return the d-spacing mask for POWGEN P2D reduction."""
    return MaskingData[DSpacingMask](
        lambda x: (
            x <= sc.scalar(0.4, unit="angstrom")
        ) | (
            x >= sc.scalar(2.0, unit="angstrom")
        )
    )


def load_powgen_dperp_mask_mlz() -> MaskingData[DperpMask]:
    """Return the dperp mask for POWGEN P2D reduction."""
    return MaskingData[DperpMask](
        lambda x: (
            x <= sc.scalar(0.2, unit="angstrom")
        ) | (
            x >= sc.scalar(4.0, unit="angstrom")
        )
    )


def extract_raw_data_mlz(data: RawDataAndMetadata[RunType]) -> TofDetector[RunType]:
    """Return folded detector events from a loaded data group."""
    # Remove the tof binning and dimension, as it is not needed and it gets in the way
    # of masking.
    if data is None:
        return cast(TofDetector[RunType], None)
    out = data.squeeze()
    out.coords.pop("tof", None)
    return TofDetector[RunType](
        out.fold(dim="spectrum", sizes={"bank": 40, "column": 154, "row": 7})
    )


def extract_accumulated_proton_charge(
    data: CorrectedDspacing[RunType],
) -> AccumulatedProtonCharge[RunType]:
    """Return the accumulated proton charge stored in reduced data."""
    return AccumulatedProtonCharge[RunType](data.coords["gd_prtn_chrg"])


def source_position_mlz(data: TofDetector[RunType]) -> Position[snx.NXsource, RunType]:
    """Return the source position from detector data."""
    if data is None:
        return cast(Position[snx.NXsource, RunType], None)
    return Position[snx.NXsource, RunType](data.coords["source_position"])


def sample_position_mlz(data: TofDetector[RunType]) -> Position[snx.NXsample, RunType]:
    """Return the sample position from detector data."""
    if data is None:
        return cast(Position[snx.NXsample, RunType], None)
    return Position[snx.NXsample, RunType](data.coords["sample_position"])


def extract_masked_intensity(
    data: MaskedExtendedIntensity2D,
) -> MaskedIntensity2D:
    """Extract intensity and attach a combined mask."""
    out = data["intensity"].copy()
    combined_mask = _calculate_combined_mask(data, out.dims)
    if combined_mask is not None:
        out.masks["combined_mask"] = combined_mask
    return MaskedIntensity2D(out)


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


def prepare_for_p2d_export(
    data2d: MaskedExtendedIntensity2D,
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
        masks={
            "combined_mask": combined_mask
        },
    )
    out = out.flatten(dims=out.dims, to="d_pixel")
    valid = ~out.masks["combined_mask"]
    return IntensityP2D(out[valid])


def export_to_p2d(
    data: IntensityP2D,
    filename: P2DOutputFilename
) -> DataFrameP2D:
    """Export P2D-ready intensity data to a text file."""
    df = pd.DataFrame(
        {
            "#2theta": data.coords["two_theta"].values,
            "lambda": data.coords["lambda"].values,
            "d-value": data.coords["dspacing"].values,
            "dp-value": data.coords["dperp"].values,
            "counts": data.values,
        }
    )

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "#Title: \n"
        "#Inst: POWGEN.prm\n"
        "#Binning: ddperp \n"
        "#Bank: 1\n"
    )

    formatter = "{:12.7f}".format

    with open(filepath, "w") as f:
        f.write(header + df.to_string(index=False, justify="left", formatters={column: formatter for column in df.columns}).lstrip())
    print(f"The output file has been saved to {filepath}.")
    return DataFrameP2D(df)


providers = (
    extract_accumulated_proton_charge,
    extract_masked_intensity,
    extract_raw_data_mlz,
    export_to_p2d,
    load_powgen_calibration_mlz,
    load_powgen_dperp_mask_mlz,
    load_powgen_dspacing_mask_mlz,
    load_powgen_mlz,
    load_powgen_tof_mask_mlz,
    load_powgen_two_theta_mask_mlz,
    load_powgen_wavelength_mask_mlz,
    prepare_for_p2d_export,
    sample_position_mlz,
    source_position_mlz,
)
