# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional

import scipp as sc
import scippneutron as scn


def load_sans2d(filename: str, spectrum_size: Optional[int] = None) -> sc.DataArray:
    """
    Loading wrapper for ISIS SANS2D files.

    Parameters
    ----------
    filename:
        The path to the file that will be loaded from disk.
    spectrum_size:
        Read only the first ``spectrum_size`` pixels. In many SANS2D experiments, only
        a quater of the pixels are kept:
          - there are two detector panels, the second one is unused
          - half of the pixels in each panel are used for live display
    """
    try:
        out = sc.io.open_hdf5(filename)
    except KeyError:
        out = scn.load(filename=filename, mantid_args={"LoadMonitors": True})
    if spectrum_size is None:
        return out
    else:
        return out["spectrum", :spectrum_size].copy()


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
