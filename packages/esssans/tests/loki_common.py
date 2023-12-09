# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import List, Optional

import sciline
import scipp as sc


from esssans.types import (
    BackgroundRun,
    CorrectForGravity,
    EmptyBeamRun,
    Filename,
    FinalDims,
    Incident,
    NeXusMonitorName,
    QBins,
    RunID,
    SampleRun,
    Transmission,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
)


def make_param_tables(
    sample_runs: Optional[List[int]] = None, background_runs: Optional[List[int]] = None
) -> List[sciline.ParamTable]:
    sample_runs = [60339]  # If more runs are added, their events will be merged
    sample_filenames = [f'{i}-2022-02-28_2215.nxs' for i in sample_runs]
    sample_runs_table = sciline.ParamTable(
        RunID[SampleRun], {Filename[SampleRun]: sample_filenames}, index=sample_runs
    )

    background_runs = [60393]  # If more runs are added, their events will be merged
    background_filenames = [f'{i}-2022-02-28_2215.nxs' for i in background_runs]
    background_runs_table = sciline.ParamTable(
        RunID[BackgroundRun],
        {Filename[BackgroundRun]: background_filenames},
        index=background_runs,
    )
    return sample_runs_table, background_runs_table


def make_params(n_wavelength_bands: int = 1) -> dict:
    params = {}

    params[Filename[TransmissionRun[SampleRun]]] = '60394-2022-02-28_2215.nxs'
    params[Filename[TransmissionRun[BackgroundRun]]] = '60392-2022-02-28_2215.nxs'
    params[Filename[EmptyBeamRun]] = '60392-2022-02-28_2215.nxs'

    params[NeXusMonitorName[Incident]] = 'monitor_1'
    params[NeXusMonitorName[Transmission]] = 'monitor_2'

    # Wavelength binning parameters
    wavelength_min = sc.scalar(1.0, unit='angstrom')
    wavelength_max = sc.scalar(13.0, unit='angstrom')
    n_wavelength_bins = 200

    params[WavelengthBins] = sc.linspace(
        'wavelength', wavelength_min, wavelength_max, n_wavelength_bins + 1
    )

    if n_wavelength_bands == 1:
        sampling_width = wavelength_max - wavelength_min
    else:
        sampling_width = 2.0 * (params[WavelengthBins][1] - params[WavelengthBins][0])

    band_start = sc.linspace(
        'band', wavelength_min, wavelength_max - sampling_width, n_wavelength_bands
    )
    band_end = band_start + sampling_width
    params[WavelengthBands] = sc.concat(
        [band_start, band_end], dim='wavelength'
    ).transpose()

    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound

    params[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom'
    )

    # The final result should have dims of Q only
    params[FinalDims] = ['Q']

    return params
