# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import List, Optional

import scipp as sc

from esssans.loki import default_parameters
from esssans.types import (
    BackgroundRun,
    CorrectForGravity,
    EmptyBeamRun,
    FileList,
    FinalDims,
    QBins,
    SampleRun,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
)


def make_params(
    sample_runs: Optional[List[str]] = None,
    background_runs: Optional[List[str]] = None,
    n_wavelength_bands: int = 1,
) -> dict:
    params = default_parameters.copy()

    if sample_runs is None:
        sample_runs = ['60339-2022-02-28_2215.nxs']
    if background_runs is None:
        background_runs = ['60393-2022-02-28_2215.nxs']

    # List of files
    params[FileList[SampleRun]] = sample_runs
    params[FileList[BackgroundRun]] = background_runs
    params[FileList[TransmissionRun[SampleRun]]] = ['60394-2022-02-28_2215.nxs']
    params[FileList[TransmissionRun[BackgroundRun]]] = ['60392-2022-02-28_2215.nxs']
    params[FileList[EmptyBeamRun]] = ['60392-2022-02-28_2215.nxs']

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
        sampling_width = 2.0 * (wavelength_max - wavelength_min) / n_wavelength_bands
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
