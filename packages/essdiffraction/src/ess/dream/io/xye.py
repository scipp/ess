# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""XYE writer for DREAM."""

import scippneutron as scn
from ess.powder.types import IntensityTof, OutFilename

from ._common import prepare_reduced_data


def save_xye(filename: OutFilename, da: IntensityTof) -> None:
    """Save reduced data to an XYE file.

    This function can be used as

    .. code-block:: python

        from ess.powder.types import OutFilename
        from ess.dream.io import save_xye

        workflow = ...
        workflow[OutFilename] = "..."
        workflow.bind_and_call(save_xye)

    Note that this function is not suitable as a provider as it
    has side effects (writes a file).

    Parameters
    ----------
    filename:
        Path of a file to write to.
    da:
        Reduced 1d data with a ``'tof'`` dimension and coordinate.
    """
    scn.io.save_xye(filename, prepare_reduced_data(da), coord="tof")
