# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
"""
from typing import NewType

import scipp as sc
import scippneutron as scn

from ..types import DirectBeam, DirectBeamFilename

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
