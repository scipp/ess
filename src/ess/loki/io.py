# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn


def load_sans2d(filename: str,
                spectrum_size: int = 245760 // 4,
                tof_bins: sc.Variable = None) -> sc.DataArray:
    """
    Loading wrapper for ISIS SANS2D files.
    By default, we keep only a quater of the pixels:
      - there are two detector panels, the second one is unused
      - half of the pixels in each panel are used for live display
    """
    out = scn.load(filename=filename, mantid_args={"LoadMonitors": True})
    out = out["spectrum", :spectrum_size]
    if tof_bins is None:
        return out.copy()
    else:
        return out.bin({tof_bins.dim: tof_bins})


def load_rkh_wav(filename: str) -> sc.DataArray:
    """
    Loading wrapper for RKH files
    """
    return scn.load(filename=filename,
                    mantid_alg="LoadRKH",
                    mantid_args={"FirstColumnValue": "Wavelength"})


def load_rkh_q(filename: str) -> sc.DataArray:
    """
    Loading wrapper for RKH files
    """
    return scn.load(filename=filename,
                    mantid_alg="LoadRKH",
                    mantid_args={"FirstColumnValue": "MomentumTransfer"})
