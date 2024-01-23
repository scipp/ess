# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
File loading functions for ISIS data using Mantid.
"""
from typing import NewType

import sciline
import scipp as sc
import scippneutron as scn
from mantid.api import MatrixWorkspace, Workspace
from mantid.simpleapi import CopyInstrumentParameters, Load
from scipp.constants import g

from ..types import (
    DirectBeam,
    DirectBeamFilename,
    LoadedFileContents,
    RunType,
    SampleRun,
)
from .io import CalibrationFilename, Filename, FilePath

CalibrationWorkspace = NewType('CalibrationWorkspace', MatrixWorkspace)


class DataWorkspace(sciline.Scope[RunType, MatrixWorkspace], MatrixWorkspace):
    """Workspace containing data"""


def _make_detector_info(ws: MatrixWorkspace) -> sc.DataGroup:
    det_info = scn.mantid.make_detector_info(ws, 'spectrum')
    return sc.DataGroup(det_info.coords)


def get_detector_ids(ws: DataWorkspace[SampleRun]) -> sc.Variable:
    det_info = _make_detector_info(ws)
    dim = 'spectrum'
    da = sc.DataArray(det_info['detector'], coords={dim: det_info[dim]})
    da = sc.sort(da, dim)  # sort by spectrum index
    if not sc.identical(
        da.coords[dim],
        sc.arange('detector', da.sizes['detector'], dtype='int32', unit=None),
    ):
        raise ValueError("Spectrum-detector mapping is not 1:1, this is not supported.")
    return da.data.rename_dims(detector='spectrum')


def load_calibration(filename: FilePath[CalibrationFilename]) -> CalibrationWorkspace:
    ws = Load(Filename=str(filename), StoreInADS=False)
    return CalibrationWorkspace(ws)


def load_direct_beam(filename: FilePath[DirectBeamFilename]) -> DirectBeam:
    dg = scn.load_with_mantid(
        filename=filename,
        mantid_alg="LoadRKH",
        mantid_args={"FirstColumnValue": "Wavelength"},
    )
    da = dg['data']
    del da.coords['spectrum']
    return DirectBeam(da)


def from_data_workspace(
    ws: DataWorkspace[RunType],
    calibration: CalibrationWorkspace,
) -> LoadedFileContents[RunType]:
    CopyInstrumentParameters(
        InputWorkspace=calibration, OutputWorkspace=ws, StoreInADS=False
    )
    up = ws.getInstrument().getReferenceFrame().vecPointingUp()
    dg = scn.from_mantid(ws)
    dg['data'] = dg['data'].squeeze()
    dg['data'].coords['detector_id'] = get_detector_ids(ws)
    dg['data'].coords['gravity'] = sc.vector(value=-up) * g
    return LoadedFileContents[RunType](dg)


def load_run(filename: FilePath[Filename[RunType]]) -> DataWorkspace[RunType]:
    loaded = Load(Filename=str(filename), LoadMonitors=True, StoreInADS=False)
    if isinstance(loaded, Workspace):
        # A single workspace
        data_ws = loaded
    else:
        # Separate data and monitor workspaces
        data_ws = loaded.OutputWorkspace
    return DataWorkspace[RunType](data_ws)


providers = (
    from_data_workspace,
    load_calibration,
    load_direct_beam,
    load_run,
)
