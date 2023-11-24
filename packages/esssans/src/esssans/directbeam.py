# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import List, Tuple

import numpy as np
import scipp as sc
from sciline import Pipeline

from .types import DirectBeam, FinalDims, IofQ, SampleRun, WavelengthBands


def make_wavelength_bins_and_bands(
    wavelength_min: sc.Variable,
    wavelength_max: sc.Variable,
    n_wavelength_bins: int,
    n_wavelength_bands: int,
    sampling_width: sc.Variable,
) -> Tuple[sc.Variable, sc.Variable]:
    """
    Make wavelength bins and bands from a wavelength range and a sampling width.

    Parameters
    ----------
    wavelength_min:
        The minimum wavelength.
    wavelength_max:
        The maximum wavelength.
    n_wavelength_bins:
        The number of wavelength bins.
    n_wavelength_bands:
        The number of wavelength bands.
    sampling_width:
        The width of the wavelength bands.
    """
    sampling_half_width = sampling_width * 0.5

    wavelength_sampling_points = sc.linspace(
        dim='wavelength',
        start=wavelength_min + sampling_half_width,
        stop=wavelength_max - sampling_half_width,
        num=n_wavelength_bands,
    )

    band_start = wavelength_sampling_points - sampling_half_width
    band_end = wavelength_sampling_points + sampling_half_width
    wavelength_bins = sc.linspace(
        'wavelength', wavelength_min, wavelength_max, n_wavelength_bins + 1
    )
    wavelength_bands = sc.concat(
        [
            sc.concat([start, end], dim='wavelength')
            for start, end in zip(band_start, band_end)
        ],
        dim='band',
    )
    return wavelength_bins, wavelength_bands


def _direct_beam_iteration(
    iofq_full: sc.DataArray,
    iofq_slices: sc.DataArray,
    direct_beam: sc.DataArray,
    I0: sc.Variable,
) -> sc.DataArray:
    eff = []
    for sl in sc.collapse(iofq_slices, keep='Q').values():
        vals = sl.values
        sel = (vals > 0.0) & np.isfinite(vals)
        f = np.median(vals[sel] / iofq_full.values[sel])
        eff.append(f)

    out = direct_beam * sc.array(dims=['wavelength'], values=eff)
    scaling = sc.values(iofq_full['Q', 0].data) / I0
    out *= scaling
    return out


def direct_beam(
    pipelines: List[Pipeline], I0: sc.Variable, niter: int = 5
) -> List[dict]:
    """
    Compute the direct beam function.

    Procedure:

    The idea behind the direct beam iterations is to determine an efficiency of the
    detectors as a function of wavelength.
    To calculate this, it is possible to compute $I(Q)$ for the full wavelength range,
    and for individual slices (bands) of the wavelength range.
    If the direct beam function used in the $I(Q)$ computation is correct, then $I(Q)$
    curves for the full wavelength range and inside the bands should overlap.

    We require two pipelines, one for the full wavelength range and one for the bands.

    The steps are as follows:

     1. Create a flat direct beam function, as a function of wavelength, with
        wavelength bins corresponding to the wavelength bands
     2. Calculate inside each band by how much one would have to multiply the final
        $I(Q)$ so that the curve would overlap with the full range curve
     3. Multiply the direct beam values inside each wavelength band by this factor
     4. Compare the full-range $I(Q)$ to a theoretical reference and add the
        corresponding additional scaling to the direct beam function
     5. Iterate until the changes to the direct beam function become small

    Parameters
    ----------
    pipelines:
        List of two pipelines, one for the full wavelength range and one for the bands.
        The order in which the pipelines are given does not matter.
    I0:
        The intensity of the I(Q) for the known sample at the lowest Q value.
    niter:
        The number of iterations to perform.
    """
    if len(pipelines) != 2:
        raise ValueError("Expected two pipelines, got {}".format(len(pipelines)))

    # Determine which pipeline is for the full wavelength range
    if pipelines[0].compute(WavelengthBands).ndim == 1:
        pipeline_full, pipeline_bands = pipelines
    else:
        pipeline_bands, pipeline_full = pipelines

    per_layer = 'layer' in pipeline_bands.compute(FinalDims)

    # Make a flat direct beam to start with
    wavelength_bands = pipeline_bands.compute(WavelengthBands)
    sizes = {'wavelength': wavelength_bands.sizes['band']}
    if per_layer:
        sizes['layer'] = 4
    direct_beam = sc.DataArray(
        data=sc.ones(sizes=sizes),
        coords={
            'wavelength': sc.midpoints(wavelength_bands, dim='wavelength')
            .squeeze()
            .rename_dims(band='wavelength')
        },
    )

    pipeline_full._providers[DirectBeam] = lambda: direct_beam
    pipeline_bands._providers[DirectBeam] = lambda: direct_beam

    results = []

    for it in range(niter):
        print("Iteration", it)

        iofq_full = pipeline_full.compute(IofQ[SampleRun])
        iofq_slices = pipeline_bands.compute(IofQ[SampleRun])

        if per_layer:
            for i in range(iofq_full.sizes['layer']):
                direct_beam['layer', i] = _direct_beam_iteration(
                    iofq_full=iofq_full['layer', i],
                    iofq_slices=iofq_slices['layer', i],
                    direct_beam=direct_beam['layer', i],
                    I0=I0,
                )
        else:
            direct_beam = _direct_beam_iteration(
                iofq_full=iofq_full,
                iofq_slices=iofq_slices,
                direct_beam=direct_beam,
                I0=I0,
            )

        results.append(
            {
                'iofq_full': iofq_full,
                'iofq_slices': iofq_slices,
                'direct_beam': direct_beam,
            }
        )
    return results
