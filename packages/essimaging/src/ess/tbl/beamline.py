# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)


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
    if ("he3" in name) or ("helium" in name):
        # Assume that the monitor number is given at the end of the string
        num = name[-1]
        return f"he3_bank_{num}"
    if "multiblade" in name:
        return "multiblade"
    if "ngem" in name:
        return "ngem"
    if "orca" in name:
        return "orca"
    if ("timepix" in name) or ("tmpx" in name) or ("tpx3" in name):
        return "timepix3"
    if "monitor" in name:
        return "beam_monitor_1"
    raise ValueError(f"TBL: Unknown detector or monitor name: {name}")
