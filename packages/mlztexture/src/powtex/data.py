# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from pathlib import Path
from typing import cast
import os
import pandas as pd
import numpy as np
import scipp as sc

from ess.packages.essdiffraction.src.ess.powder.types import (
    FilepathMLZ,
    FilenameMLZ,
    RunType,
    CalibrationRun,
    CalibrationData,
    TofDetector
)


def load_powtex_csv(
    filename: FilenameMLZ[RunType],
    datapath: FilepathMLZ[RunType]
) -> TofDetector[RunType]:
    """Load a csv file."""
    if filename is None:
        return cast(TofDetector[RunType], None)
    filepath = Path(os.path.join(datapath, filename))
    data = pd.read_csv(filepath, sep = '\t')

    data_tof = data["tof [ns]"].to_numpy()
    data_x = data["voxel_x [mm]"].to_numpy()
    data_y = data["voxel_y [mm]"].to_numpy()
    data_z = data["voxel_z [mm]"].to_numpy()

    data_xyz = np.dstack((data_x, data_y, data_z))

    event_coords = sc.vectors(dims=["event"], values=data_xyz[0], unit="mm")
    event_data = sc.array(dims=["event"], values=np.ones(event_coords.shape[0]), unit="counts")
    event_tof = sc.array(dims=["event"], values=data_tof, unit="ns")

    # ToDo: these should be read from the file, hardcoded for now
    sample_pos = sc.vector(value=np.array([0.0, 0.0, 0.0]), unit="mm")
    source_pos = sc.vector(value=np.array([0.0, 0.0, -41410.0]), unit="mm")

    out = sc.DataArray(
        data=event_data,
        coords={
            "position": event_coords,
            "sample_position": sample_pos,
            "source_position": source_pos,
            "tof": event_tof
        },
    )
    return TofDetector[RunType](out)

def load_powtex_calibration(
    filename: FilenameMLZ[CalibrationRun],
    datapath: FilepathMLZ[CalibrationRun]
) -> CalibrationData:
    return CalibrationData(None)
