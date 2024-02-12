# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Optional


from ...data import Registry
from ..io import DataFolder, FilenameType, FilePath

_registry = Registry(
    instrument='sans2d',
    files={
        # Direct beam file (efficiency of detectors as a function of wavelength)
        'DIRECT_SANS2D_REAR_34327_4m_8mm_16Feb16.dat.h5': 'md5:43f4188301d709aa49df0631d03a67cb',  # noqa: E501
        # Empty beam run (no sample and no sample holder/can)
        'SANS2D00063091.nxs.h5': 'md5:ec7f78d51a4abc643bbe1965b7a876b9',
        # Sample run (sample and sample holder/can)
        'SANS2D00063114.nxs.h5': 'md5:d0701afe88c09e6a714b31ecfbd79c0c',
        # Background run (no sample, sample holder/can only)
        'SANS2D00063159.nxs.h5': 'md5:8d740b29d8965c8d9ca4f20f1e68ec15',
        # Solid angles of the SANS2D detector pixels computed by Mantid (for tests)
        'SANS2D00063091.SolidAngle_from_mantid.h5': 'md5:d57b82db377cb1aea0beac7202713861',  # noqa: E501
    },
    version='1',
)


def get_path(
    filename: FilenameType, folder: Optional[DataFolder]
) -> FilePath[FilenameType]:
    if folder is not None:
        return f'{folder}/{filename}'
    mapping = {
        'DIRECT_SANS2D_REAR_34327_4m_8mm_16Feb16.dat': 'DIRECT_SANS2D_REAR_34327_4m_8mm_16Feb16.dat.h5',  # noqa: E501
        'SANS2D00063091.nxs': 'SANS2D00063091.nxs.h5',
        'SANS2D00063114.nxs': 'SANS2D00063114.nxs.h5',
        'SANS2D00063159.nxs': 'SANS2D00063159.nxs.h5',
    }
    filename = mapping.get(filename, filename)
    return _registry.get_path(filename)


providers = (get_path,)
