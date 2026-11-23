# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Data handling for POWTEX powder reduction."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipp as sc
import scippnexus as snx

from ..powtex.types import (
    AccumulatedProtonCharge,
    CalibrationData,
    CalibrationRun,
    CorrectedDspacing,
    Filename,
    Filepath,
    Position,
    RawDataAndMetadata,
    RunType,
    TofDetector,
)
from .types import (
    DataFrameP2D,
    DperpMask,
    DSpacingMask,
    IntensityP2D,
    MaskingData,
    P2DOutputFilename,
    TofMask,
    TwoThetaMask,
    WavelengthMask,
)


def load_powgen(
    filename: Filename[RunType], datapath: Filepath[RunType]
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


def load_powgen_calibration(
    filename: Filename[CalibrationRun], datapath: Filepath[CalibrationRun]
) -> CalibrationData:
    """Load a diamond calibration file."""
    if filename is None:
        return CalibrationData(None)
    filepath = Path(datapath) / filename
    ds = sc.io.load_hdf5(filepath)
    # print(f"Loading calibration data from {filepath}")
    out = sc.Dataset(
        {
            key: da.fold(dim="spectrum", sizes={"wire": 40, "strip": 154, "segment": 7})
            for key, da in ds.items()
        }
    )
    return CalibrationData(out)


def load_powgen_tof_mask() -> MaskingData[TofMask]:
    """Return the time-of-flight mask for POWGEN P2D reduction."""
    mask_function = None
    return MaskingData[TofMask](mask_function)


def load_powgen_wavelength_mask() -> MaskingData[WavelengthMask]:
    """Return the wavelength mask for POWGEN P2D reduction."""
    return MaskingData[WavelengthMask](
        lambda x: (
            (x <= sc.scalar(0.3, unit="angstrom"))
            | (x >= sc.scalar(1.25, unit="angstrom"))
            | (
                (x >= sc.scalar(1.03, unit="angstrom"))
                & (x <= sc.scalar(1.07, unit="angstrom"))
            )
        )
    )


def load_powgen_two_theta_mask() -> MaskingData[TwoThetaMask]:
    """Return the two-theta mask for POWGEN P2D reduction."""

    def deg(value: float) -> sc.Variable:
        return sc.scalar(value, unit="degree")

    return MaskingData[TwoThetaMask](
        lambda x: (
            (
                x < deg(10.0)  # extra cut as in powtex
            )
            | (x > deg(150.0))
            | ((x > deg(13.6)) & (x < deg(16.6)))
            | ((x > deg(23.6)) & (x < deg(26.6)))
            | ((x > deg(34.3)) & (x < deg(37.3)))
            | ((x > deg(45.2)) & (x < deg(48.2)))
            | ((x > deg(56.0)) & (x < deg(61.0)))
            | ((x > deg(69.8)) & (x < deg(72.8)))
            | ((x > deg(83.5)) & (x < deg(86.5)))
            | ((x > deg(97.1)) & (x < deg(101.1)))
            | ((x > deg(113.0)) & (x < deg(116.0)))
            | ((x > deg(131.0)) & (x < deg(134.0)))
            | ((x > deg(146.0)) & (x < deg(150.0)))
        )
    )


def load_powgen_dspacing_mask() -> MaskingData[DSpacingMask]:
    """Return the d-spacing mask for POWGEN P2D reduction."""
    return MaskingData[DSpacingMask](
        lambda x: (
            (x <= sc.scalar(0.4, unit="angstrom"))
            | (x >= sc.scalar(2.0, unit="angstrom"))
        )
    )


def load_powgen_dperp_mask() -> MaskingData[DperpMask]:
    """Return the dperp mask for POWGEN P2D reduction."""
    return MaskingData[DperpMask](
        lambda x: (
            (x <= sc.scalar(0.2, unit="angstrom"))
            | (x >= sc.scalar(4.0, unit="angstrom"))
        )
    )


def extract_raw_data(data: RawDataAndMetadata[RunType]) -> TofDetector[RunType]:
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


def source_position(data: TofDetector[RunType]) -> Position[snx.NXsource, RunType]:
    """Return the source position from detector data."""
    if data is None:
        return cast(Position[snx.NXsource, RunType], None)
    return Position[snx.NXsource, RunType](data.coords["source_position"])


def sample_position(data: TofDetector[RunType]) -> Position[snx.NXsample, RunType]:
    """Return the sample position from detector data."""
    if data is None:
        return cast(Position[snx.NXsample, RunType], None)
    return Position[snx.NXsample, RunType](data.coords["sample_position"])


def export_to_p2d(data: IntensityP2D, filename: P2DOutputFilename) -> DataFrameP2D:
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

    filepath = Path(filename).expanduser()
    if not filepath.is_absolute():
        filepath = Path.home() / filepath
    filepath.parent.mkdir(parents=True, exist_ok=True)

    header = "#Title: \n#Inst: POWGEN.prm\n#Binning: ddperp \n#Bank: 1\n"

    formatter = "{:12.7f}".format

    with filepath.open("w") as f:
        table = df.to_string(
            index=False,
            justify="left",
            formatters=dict.fromkeys(df.columns, formatter),
        )
        f.write(header + table.lstrip())
    print("The output has been saved to: ", filepath)
    return DataFrameP2D(df)


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
    extract_accumulated_proton_charge,
    extract_raw_data,
    export_to_p2d,
    load_powgen_calibration,
    load_powgen_dperp_mask,
    load_powgen_dspacing_mask,
    load_powgen,
    load_powgen_tof_mask,
    load_powgen_two_theta_mask,
    load_powgen_wavelength_mask,
    sample_position,
    source_position,
)
