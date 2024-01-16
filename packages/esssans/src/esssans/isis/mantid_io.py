# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
File loading function for ISIS data, using Mantid.
"""
from typing import NewType

import sciline
import scipp as sc
import scippneutron as scn

from ..common import gravity_vector
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
InstrumentName = NewType('InstrumentName', str)


DetectorInfo = NewType('DetectorInfo', sc.Dataset)


def get_detector_info(data: LoadedFileContents[SampleRun]) -> DetectorInfo:
    return DetectorInfo(data['detector_info'])


def get_idf_filename(data: LoadedFileContents[SampleRun]) -> IDFFilename:
    return IDFFilename(data['idf_filename'])


def get_instrument_name(data: LoadedFileContents[SampleRun]) -> InstrumentName:
    return InstrumentName(data['instrument_name'])


def load_pixel_mask(
    filename: PixelMaskFilename,
    instrument: IDFFilename,
    detector_info: DetectorInfo,
    instrument_name: InstrumentName,
) -> PixelMask:
    mask = scn.load_with_mantid(
        filename=instrument, mantid_alg="LoadMask", mantid_args={"InputFile": filename}
    )
    mask = mask['data']
    if instrument_name == 'ZOOM':
        # This is a hack to get the correct spectrum numbers. It would be better
        # to use the `RefWorkspace` argument to `LoadMask`, but this would require
        # exposing the underlying Mantid workspaces in the Sciline graph.
        mask.coords['spectrum'] = detector_info['spectrum'].rename_dims(
            detector='spectrum'
        ) + sc.index(9, dtype='int32')
        mask = sc.sort(mask, 'spectrum')
    return PixelMask(mask)


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


def _get_idf_path(ws) -> str:
    """Get the IDF path from a workspace"""
    lines = repr(ws).split('\n')
    line = [line for line in lines if 'Instrument from:' in line][0]
    path = line.split('Instrument from:')[1].strip()
    return path


# It would be nice if we could use the generic run-merging facility,
# but this does not work directly since we need to rely on a different loader.
# Could we use a param table for the filenames, apply the loader to each,
# and then merge?
def load_run(filename: Filename[RunType]) -> LoadedFileContents[RunType]:
    with scn.mantid.run_mantid_alg('Load', str(filename), LoadMonitors=True) as loaded:
        data_ws = loaded.OutputWorkspace
        dg = scn.from_mantid(data_ws)
        det_info = scn.mantid.make_detector_info(data_ws, 'spectrum')
        idf = _get_idf_path(data_ws)
    dg['detector_info'] = sc.DataGroup(det_info.coords)
    dg['idf_filename'] = idf
    dg['data'] = dg['data'].squeeze()
    # TODO Is this correct for ISIS? Can we get it from the workspace?
    dg['data'].coords['gravity'] = gravity_vector()
    return LoadedFileContents[RunType](dg)
