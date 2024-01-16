# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
"""
from typing import NewType

import sciline
import scipp as sc
import scippneutron as scn

from ..common import gravity_vector
from ..types import DirectBeam, DirectBeamFilename, LoadedFileContents, RunType

IDFFilename = NewType('IDFFilename', str)
PixelMaskFilename = NewType('PixelMaskFilename', str)
PixelMask = NewType('PixelMask', sc.Variable)


def load_pixel_mask(filename: PixelMaskFilename, instrument: IDFFilename) -> PixelMask:
    mask = scn.load_with_mantid(
        filename=instrument, mantid_alg="LoadMask", mantid_args={"InputFile": filename}
    )
    return PixelMask(mask['data'].data)


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


# It would be nice if we could use the generic run-merging facility,
# but this does not work directly since we need to rely on a different loader.
# Could we use a param table for the filenames, apply the loader to each,
# and then merge?
def load_run(filename: Filename[RunType]) -> LoadedFileContents[RunType]:
    dg = scn.load_with_mantid(filename=filename, mantid_args={'LoadMonitors': True})
    # TODO Is this correct for ISIS? Can we get it from the workspace?
    dg['data'].coords['gravity'] = gravity_vector()
    return LoadedFileContents[RunType](dg)
