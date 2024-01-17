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
from mantid.simpleapi import CopyInstrumentParameters, Load, LoadMask
from scipp.constants import g

from ..types import (
    DirectBeam,
    DirectBeamFilename,
    LoadedFileContents,
    RunType,
    SampleRun,
)

IDFFilename = NewType('IDFFilename', str)
PixelMaskFilename = NewType('PixelMaskFilename', str)
PixelMask = NewType('PixelMask', sc.Variable)

PixelMaskWorkspace = NewType('PixelMaskWorkspace', MatrixWorkspace)

CalibrationFilename = NewType('CalibrationFilename', str)
CalibrationWorkspace = NewType('CalibrationWorkspace', MatrixWorkspace)

DetectorIDs = NewType('DetectorIDs', sc.Variable)
"""Detector ID corresponding to each spectrum."""


class DataWorkspace(sciline.Scope[RunType, MatrixWorkspace], MatrixWorkspace):
    """Workspace containing data"""


def _make_detector_info(ws: MatrixWorkspace) -> sc.DataGroup:
    det_info = scn.mantid.make_detector_info(ws, 'spectrum')
    return sc.DataGroup(det_info.coords)


def get_detector_ids(ws: DataWorkspace[SampleRun]) -> DetectorIDs:
    det_info = _make_detector_info(ws)
    dim = 'spectrum'
    da = sc.DataArray(det_info['detector'], coords={dim: det_info[dim]})
    da = sc.sort(da, dim)  # sort by spectrum index
    if not sc.identical(
        da.coords[dim],
        sc.arange('detector', da.sizes['detector'], dtype='int32', unit=None),
    ):
        raise ValueError("Spectrum-detector mapping is not 1:1, this is not supported.")
    return DetectorIDs(da.data.rename_dims(detector='spectrum'))


def get_idf_filename(ws: DataWorkspace[SampleRun]) -> IDFFilename:
    lines = repr(ws).split('\n')
    line = [line for line in lines if 'Instrument from:' in line][0]
    path = line.split('Instrument from:')[1].strip()
    return IDFFilename(path)


def from_pixel_mask_workspace(ws: PixelMaskWorkspace, detids: DetectorIDs) -> PixelMask:
    mask = scn.from_mantid(ws)['data']
    # The 'spectrum' coord of `mask` is the spectrum *number*, but the detector info
    # uses the spectrum *index*, i.e., a simple 0-based index.
    mask.coords['spectrum'] = sc.arange(
        'spectrum', mask.sizes['spectrum'], dtype='int32', unit=None
    )
    index_to_mask = sc.lookup(mask, dim='spectrum', mode='previous')
    mask_det_info = _make_detector_info(ws)
    det_mask = sc.DataArray(
        index_to_mask[mask_det_info['spectrum']],
        coords={'detector': mask_det_info['detector']},
    )
    det_mask = sc.sort(det_mask, 'detector')
    detid_to_mask = sc.lookup(det_mask, dim='detector', mode='previous')
    return PixelMask(detid_to_mask[detids])


def load_calibration(filename: CalibrationFilename) -> CalibrationWorkspace:
    ws = Load(Filename=str(filename), StoreInADS=False)
    return CalibrationWorkspace(ws)


def load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    dg = scn.load_with_mantid(
        filename=filename,
        mantid_alg="LoadRKH",
        mantid_args={"FirstColumnValue": "Wavelength"},
    )
    da = dg['data']
    del da.coords['spectrum']
    return DirectBeam(da)


class Filename(sciline.Scope[RunType, str], str):
    """Filename of a run"""


def from_data_workspace(
    ws: DataWorkspace[RunType], calibration: CalibrationWorkspace
) -> LoadedFileContents[RunType]:
    CopyInstrumentParameters(
        InputWorkspace=calibration, OutputWorkspace=ws, StoreInADS=False
    )
    up = ws.getInstrument().getReferenceFrame().vecPointingUp()
    dg = scn.from_mantid(ws)
    dg['data'] = dg['data'].squeeze()
    dg['data'].coords['gravity'] = sc.vector(value=-up) * g
    return LoadedFileContents[RunType](dg)


def load_pixel_mask(
    filename: PixelMaskFilename,
    idf_path: IDFFilename,
    ref_workspace: DataWorkspace[SampleRun],
) -> PixelMaskWorkspace:
    mask = LoadMask(
        Instrument=idf_path,
        InputFile=str(filename),
        RefWorkspace=ref_workspace,
        StoreInADS=False,
    )
    return PixelMaskWorkspace(mask)


def load_run(filename: Filename[RunType]) -> DataWorkspace[RunType]:
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
    from_pixel_mask_workspace,
    get_detector_ids,
    get_idf_filename,
    load_calibration,
    load_direct_beam,
    load_pixel_mask,
    load_run,
)
