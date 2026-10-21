# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from types import MappingProxyType

import scipp as sc
from scippneutron.chopper import DiskChopper


def nexus_name(name: str) -> str:
    """
    Get the NeXus path for a given detector or monitor name.
    The returned path is the location inside the `/entry/instrument` group of the NeXus
    file.

    Parameters
    ----------
    name:
        Name of the detector or monitor.
    """
    name = name.lower()
    if ("timepix" in name) or ("tmpx" in name):
        return "timepix3"
    if "orca" in name:
        return "orca"
    if "monitor" in name:
        # Assume that the monitor number is given at the end of the string
        num = name[-1]
        return f"beam_monitor_{num}"
