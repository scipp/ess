# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from pathlib import Path

from ess.sans.data import Registry
from ess.sans.types import (
    BackgroundRun,
    Filename,
    PixelMaskFilename,
    SampleRun,
    TransmissionRun,
)

_registry = Registry(
    instrument='loki',
    files={
        # Files from LoKI@Larmor detector test experiment
        #
        # Background run 1 (no sample, sample holder/can only, no transmission monitor)
        '60248-2022-02-28_2215.nxs': 'md5:d9f17b95274a0fc6468df7e39df5bf03',
        # Sample run 1 (sample + sample holder/can, no transmission monitor in beam)
        '60250-2022-02-28_2215.nxs': 'md5:6a519ceaacbae702a6d08241e86799b1',
        # Sample run 2 (sample + sample holder/can, no transmission monitor in beam)
        '60339-2022-02-28_2215.nxs': 'md5:03c86f6389566326bb0cbbd80b8f8c4f',
        # Background transmission run (sample holder/can + transmission monitor)
        '60392-2022-02-28_2215.nxs': 'md5:9ecc1a9a2c05a880144afb299fc11042',
        # Background run 2 (no sample, sample holder/can only, no transmission monitor)
        '60393-2022-02-28_2215.nxs': 'md5:bf550d0ba29931f11b7450144f658652',
        # Sample transmission run (sample + sample holder/can + transmission monitor)
        '60394-2022-02-28_2215.nxs': 'md5:c40f38a62337d86957af925296c4c615',
        # Analytical model for the I(Q) of the Poly-Gauss sample
        'PolyGauss_I0-50_Rg-60.h5': 'md5:f5d60d9c2286cb197b8cd4dc82db3d7e',
        # XML file for the pixel mask
        'mask_new_July2022.xml': 'md5:421b6dc9db74126ffbc5d88164d017b0',
    },
    version='2',
)


def loki_tutorial_sample_run_60250() -> Filename[SampleRun]:
    """Sample run with sample and sample holder/can, no transmission monitor in beam."""
    return Filename[SampleRun](_registry.get_path('60250-2022-02-28_2215.nxs'))


def loki_tutorial_sample_run_60339() -> Filename[SampleRun]:
    """Sample run with sample and sample holder/can, no transmission monitor in beam."""
    return Filename[SampleRun](_registry.get_path('60339-2022-02-28_2215.nxs'))


def loki_tutorial_background_run_60248() -> Filename[BackgroundRun]:
    """Background run with sample holder/can only, no transmission monitor."""
    return Filename[BackgroundRun](_registry.get_path('60248-2022-02-28_2215.nxs'))


def loki_tutorial_background_run_60393() -> Filename[BackgroundRun]:
    """Background run with sample holder/can only, no transmission monitor."""
    return Filename[BackgroundRun](_registry.get_path('60393-2022-02-28_2215.nxs'))


def loki_tutorial_sample_transmission_run() -> Filename[TransmissionRun[SampleRun]]:
    """Sample transmission run (sample + sample holder/can + transmission monitor)."""
    return Filename[TransmissionRun[SampleRun]](
        _registry.get_path('60394-2022-02-28_2215.nxs')
    )


def loki_tutorial_run_60392() -> Filename[TransmissionRun[BackgroundRun]]:
    """Background transmission run (sample holder/can + transmission monitor), also
    used as empty beam run."""
    return Filename[TransmissionRun[BackgroundRun]](
        _registry.get_path('60392-2022-02-28_2215.nxs')
    )


def loki_tutorial_mask_filenames() -> list[PixelMaskFilename]:
    """List of pixel mask filenames for the LoKI@Larmor detector test experiment."""
    return [
        PixelMaskFilename(_registry.get_path('mask_new_July2022.xml')),
    ]


def loki_tutorial_poly_gauss_I0() -> Path:
    """Analytical model for the I(Q) of the Poly-Gauss sample."""
    return Path(_registry.get_path('PolyGauss_I0-50_Rg-60.h5'))
