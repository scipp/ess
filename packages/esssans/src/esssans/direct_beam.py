# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import List, Tuple

import numpy as np
import scipp as sc
from sciline import Pipeline

from .types import BackgroundSubtractedIofQ, DirectBeam, FinalDims, WavelengthBands


def _direct_beam_iteration(
    iofq_full: sc.DataArray,
    iofq_slices: sc.DataArray,
    direct_beam_function: sc.DataArray,
    I0: sc.Variable,
) -> sc.DataArray:
    eff = []
    for sl in sc.collapse(iofq_slices, keep='Q').values():
        vals = sl.values
        sel = (vals > 0.0) & np.isfinite(vals)
        f = np.median(vals[sel] / iofq_full.values[sel])
        eff.append(f)

    out = direct_beam_function * sc.array(dims=['wavelength'], values=eff)
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
    direct_beam_function = sc.DataArray(
        data=sc.ones(sizes=sizes),
        coords={
            'wavelength': sc.midpoints(wavelength_bands, dim='wavelength')
            .squeeze()
            .rename_dims(band='wavelength')
        },
    )

    # TODO: This feels like an ugly hack. What is the best way to achieve this?
    pipeline_full._providers[DirectBeam] = lambda: direct_beam_function
    pipeline_bands._providers[DirectBeam] = lambda: direct_beam_function

    results = []

    for it in range(niter):
        print("Iteration", it)

        iofq_full = pipeline_full.compute(BackgroundSubtractedIofQ)
        iofq_slices = pipeline_bands.compute(BackgroundSubtractedIofQ)

        if per_layer:
            for i in range(iofq_full.sizes['layer']):
                direct_beam_function['layer', i] = _direct_beam_iteration(
                    iofq_full=iofq_full['layer', i],
                    iofq_slices=iofq_slices['layer', i],
                    direct_beam_function=direct_beam_function['layer', i],
                    I0=I0,
                )
        else:
            direct_beam_function = _direct_beam_iteration(
                iofq_full=iofq_full,
                iofq_slices=iofq_slices,
                direct_beam_function=direct_beam_function,
                I0=I0,
            )

        results.append(
            {
                'iofq_full': iofq_full,
                'iofq_slices': iofq_slices,
                'direct_beam': direct_beam_function,
            }
        )
    return results
