# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import List

import numpy as np
import scipp as sc
from sciline import Pipeline

from .types import BackgroundSubtractedIofQ, DirectBeam, WavelengthBands


def get_I0(filename: str, q: sc.Variable) -> sc.Variable:
    """
    Get the intensity of the I(Q) for the known sample at a specified Q value from a
    reference file.

    Parameters
    ----------
    filename:
        The name of the reference file.
    q:
        The Q value at which to compute the intensity.

    Returns
    -------
    :
        The interpolated intensity of the I(Q) for the requested Q value.

    """
    from .loki.data import get_path

    data = np.loadtxt(get_path(filename))

    qcoord = sc.array(dims=["Q"], values=data[:, 0], unit='1/angstrom')
    theory = sc.DataArray(
        data=sc.array(dims=["Q"], values=data[:, 1], unit=''), coords={"Q": qcoord}
    )

    ind = np.argmax((qcoord > q).values)
    I0 = (theory.data[ind] - theory.data[ind - 1]) / (qcoord[ind] - qcoord[ind - 1]) * (
        q - qcoord[ind - 1]
    ) + theory.data[ind - 1]
    return I0


def _compute_efficiency_correction(
    iofq_full: sc.DataArray,
    iofq_bands: sc.DataArray,
    wavelength_band_dim: str,
    I0: sc.Variable,
) -> sc.DataArray:
    """
    Compute the factor by which to multiply the direct beam function inside each
    wavelength band so that the $I(Q)$ curves for the full wavelength range and inside
    the bands overlap.

    Parameters
    ----------
    iofq_full:
        The $I(Q)$ for the full wavelength range.
    iofq_bands:
        The $I(Q)$ for the wavelength bands.
    wavelength_band_dim:
        The name of the wavelength band dimension.
    I0:
        The intensity of the I(Q) for the known sample at the lowest Q value.
    """
    invalid = (iofq_bands.data <= sc.scalar(0.0)) | ~sc.isfinite(iofq_bands.data)
    data = np.where(invalid.values, np.nan, (iofq_bands.data / iofq_full).values)
    eff = np.nanmedian(data, axis=iofq_bands.dims.index('Q'))

    dims = set(iofq_bands.dims) - {'Q'}
    dims.remove(wavelength_band_dim)
    dims.add('wavelength')

    scaling = sc.values(iofq_full['Q', 0].data) / I0
    return sc.array(dims=dims, values=eff) * scaling


def direct_beam(pipeline: Pipeline, I0: sc.Variable, niter: int = 5) -> List[dict]:
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
     5. Iterate a given number of times (typically less than 10) so as to gradually
        converge on a direct beam function

    TODO: For simplicity and robustness, we currently specify the number of times to
    iterate. We could imagine in the future having a convergence criterion instead to
    determine when to stop iterating.

    Parameters
    ----------
    pipeline:
        The pipeline to compute the differential scattering cross section I(Q).
    I0:
        The intensity of the I(Q) for the known sample at the lowest Q value.
    niter:
        The number of iterations to perform.
    """

    direct_beam_function = None
    wavelength_bands = pipeline.compute(WavelengthBands)
    band_dim = (set(wavelength_bands.dims) - {'wavelength'}).pop()

    full_wavelength_range = sc.concat(
        [wavelength_bands.min(), wavelength_bands.max()], dim='wavelength'
    )

    pipeline_bands = pipeline.copy()
    pipeline_full = pipeline_bands.copy()
    pipeline_full[WavelengthBands] = full_wavelength_range

    results = []

    for it in range(niter):
        print("Iteration", it)

        # The first time we compute I(Q), the direct beam function is not in the
        # parameters, nor given by any providers, so it will be considered flat.
        # TODO: Should we have a check that DirectBeam cannot be computed from the
        # pipeline?
        iofq_full = pipeline_full.compute(BackgroundSubtractedIofQ)
        iofq_bands = pipeline_bands.compute(BackgroundSubtractedIofQ)

        if direct_beam_function is None:
            # Make a flat direct beam
            dims = set(iofq_bands.dims) - {'Q'}
            direct_beam_function = sc.DataArray(
                data=sc.ones(sizes={dim: iofq_bands.sizes[dim] for dim in dims}),
                coords={
                    band_dim: sc.midpoints(wavelength_bands, dim='wavelength').squeeze()
                },
            ).rename({band_dim: 'wavelength'})

        direct_beam_function *= _compute_efficiency_correction(
            iofq_full=iofq_full,
            iofq_bands=iofq_bands,
            wavelength_band_dim=band_dim,
            I0=I0,
        )

        # Insert new direct beam function into pipelines
        pipeline_bands[DirectBeam] = direct_beam_function
        pipeline_full[DirectBeam] = direct_beam_function

        results.append(
            {
                'iofq_full': iofq_full,
                'iofq_bands': iofq_bands,
                'direct_beam': direct_beam_function,
            }
        )
    return results
