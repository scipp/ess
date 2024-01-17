# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


from ..data import Registry

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
    },
    version='2',
)


get_path = _registry.get_path

__all__ = ['get_path']
