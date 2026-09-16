# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Data loading helpers for POWTEX texture reduction."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipp as sc

from ..powder.types import (
    CalibrationData,
    CalibrationRun,
    FilenameMLZ,
    FilepathMLZ,
    RunType,
    TofDetector,
)


def load_powtex_csv(
    filename: FilenameMLZ[RunType],
    datapath: FilepathMLZ[RunType]
) -> TofDetector[RunType]:
    """Load a POWTEX CSV event file."""
    if filename is None:
        return cast(TofDetector[RunType], None)
    data = pd.read_csv(Path(datapath) / filename, sep="\t")

    data_tof = data["tof [ns]"].to_numpy()
    data_xyz = data[["voxel_x [mm]", "voxel_y [mm]", "voxel_z [mm]"]].to_numpy()

    event_coords = sc.vectors(
        dims=["event"],
        values=data_xyz,
        unit="mm",
    )
    event_data = sc.ones(dims=["event"], shape=[len(data)], unit="counts")
    event_tof = sc.array(
        dims=["event"], values=data_tof, unit="ns"
    )

    # TODO: Read these from the data file once they are available there.
    sample_pos = sc.vector(value=np.array([0.0, 0.0, 0.0]), unit="mm")
    source_pos = sc.vector(value=np.array([0.0, 0.0, -41410.0]), unit="mm")

    out = sc.DataArray(
        data=event_data,
        coords={
            "position": event_coords,
            "sample_position": sample_pos,
            "source_position": source_pos,
            "tof": event_tof,
        },
    )
    return TofDetector[RunType](out)


def load_powtex_calibration(
    filename: FilenameMLZ[CalibrationRun],
    datapath: FilepathMLZ[CalibrationRun],
) -> CalibrationData:
    """Load POWTEX calibration data."""
    _ = filename
    _ = datapath
    return CalibrationData(None)
