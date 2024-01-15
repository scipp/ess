# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


from ..data import Registry

_registry = Registry(
    instrument='loki',
    files={
        # Direct beam file (efficiency of detectors as a function of wavelength)
        'DIRECT_SANS2D_REAR_34327_4m_8mm_16Feb16.hdf5': 'md5:43f4188301d709aa49df0631d03a67cb',  # noqa: E501
        # Empty beam run (no sample and no sample holder/can) - NeXus format
        'SANS2D00063091.nxs': 'md5:05929753ea06eca5fe4be164cb06b4d6',
        # Empty beam run (no sample and no sample holder/can) - Scipp-hdf5 format
        'SANS2D00063091.hdf5': 'md5:1fdbe36a496e914336f2f9b5cad9f00e',
        # Sample run (sample and sample holder/can)
        'SANS2D00063114.hdf5': 'md5:536303077b9f55286af5ef6ec5124e1c',
        # Background run (no sample, sample holder/can only)
        'SANS2D00063159.hdf5': 'md5:e2f2aea3ed9251e579210718de568203',
        # Solid angles of the SANS2D detector pixels computed by Mantid (for tests)
        'SANS2D00063091.SolidAngle_from_mantid.hdf5': 'md5:d57b82db377cb1aea0beac7202713861',  # noqa: E501
    },
    version='2',
)


get_path = _registry.get_path

__all__ = ['get_path']
