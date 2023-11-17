# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippnexus as snx
from scippnexus.application_definitions import nxcansas

from .types import BackgroundSubtractedIofQ, OutFilename, RawData, SampleRun


def save_background_subtracted_iofq(
    *,
    iofq: BackgroundSubtractedIofQ,
    out_filename: OutFilename,
    raw_data: RawData[SampleRun],
) -> None:
    """Save background subtracted IofQ as an NXcanSAS file."""
    da = iofq.copy(deep=False)
    da.coords['Q'] = sc.midpoints(da.coords['Q'])

    with snx.File(out_filename, 'w') as f:
        f['sasentry'] = nxcansas.SASentry(
            title=raw_data['run_title'].value, run=int(raw_data['run_number'])
        )
        f['sasentry']['sasdata'] = nxcansas.SASdata(da, Q_variances='resolutions')
