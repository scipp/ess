# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

from .supermirror import SupermirrorReflectivityCorrection
from .tools import fwhm_to_std
from .types import (
    BeamSize,
    FootprintCorrectedData,
    IdealReferenceIntensity,
    MaskedData,
    ReferenceIntensity,
    ReferenceRun,
    RunType,
    SampleSize,
    WavelengthBins,
)


def footprint_correction(
    data_array: MaskedData[RunType],
    beam_size: BeamSize[RunType],
    sample_size: SampleSize[RunType],
) -> FootprintCorrectedData[RunType]:
    """
    Corrects the event weights by the fraction of the beam hitting the sample.
    Depends on :math:`\\theta`.

    Parameters
    ----------
    data_array:
        Data array to perform footprint correction on.
    beam_size:
        Full width half maximum of the beam.
    sample_size:
        Size of the sample.
        TODO: check what sample size actually means. Is it the sample diameter? etc.

    Returns
    -------
    :
       Footprint corrected data array.
    """
    size_of_beam_on_sample = beam_size / sc.sin(data_array.bins.coords["theta"])
    footprint_scale = sc.erf(fwhm_to_std(sample_size / size_of_beam_on_sample))
    data_array_fp_correction = data_array / footprint_scale
    return FootprintCorrectedData[RunType](data_array_fp_correction)


def compute_reference_intensity(
    da: FootprintCorrectedData[ReferenceRun], wb: WavelengthBins
) -> ReferenceIntensity:
    """Creates a reference intensity map over (z_index, wavelength).
    Rationale:
        The intensity expressed in those variables should not vary
        with the experiment parameters (such as sample rotation).
        Therefore it can be used to normalize sample measurements.
    """
    b = da.bin(wavelength=wb, dim=set(da.dims) - set(da.coords["z_index"].dims))
    h = b.hist()
    h.masks["too_few_events"] = h.data < sc.scalar(1, unit="counts")
    # Add a Q coordinate to each bin, the Q is not completely unique in every bin,
    # but it is close enough.
    h.coords["Q"] = b.bins.coords["Q"].bins.mean()
    return ReferenceIntensity(h)


def calibrate_reference(
    da: ReferenceIntensity, cal: SupermirrorReflectivityCorrection
) -> IdealReferenceIntensity:
    """Calibrates the reference intensity by the
    inverse of the supermirror reflectivity"""
    return IdealReferenceIntensity(da * cal)


providers = (
    footprint_correction,
    calibrate_reference,
    compute_reference_intensity,
)
