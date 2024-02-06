# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippnexus as snx
from scippnexus.application_definitions import nxcansas

from .types import BackgroundSubtractedIofQ, OutFilename, RunNumber, RunTitle


def save_background_subtracted_iofq(
    *,
    iofq: BackgroundSubtractedIofQ,
    out_filename: OutFilename,
    run_number: RunNumber,
    run_title: RunTitle,
) -> None:
    """Save background-subtracted I(Q) histogram as an NXcanSAS file."""
    if iofq.bins is None:
        da = iofq.copy(deep=False)
    else:
        da = iofq.hist()
    if da.coords.is_edges('Q'):
        da.coords['Q'] = sc.midpoints(da.coords['Q'])
    with snx.File(out_filename, 'w') as f:
        f['sasentry'] = nxcansas.SASentry(title=run_title, run=run_number)
        f['sasentry']['sasdata'] = nxcansas.SASdata(da, Q_variances='resolutions')
