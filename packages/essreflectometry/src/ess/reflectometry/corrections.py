# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Optional

import scipp as sc

from .supermirror import SupermirrorReflectivityCorrection
from .tools import fwhm_to_std
from .types import (
    BeamSize,
    CorrectionMatrix,
    FootprintCorrectedData,
    HistogrammedReference,
    MaskedEventData,
    Reference,
    Run,
    SampleSize,
    WavelengthBins,
)


def footprint_correction(
    data_array: MaskedEventData[Run],
    beam_size: BeamSize[Run],
    sample_size: Optional[SampleSize[Run]],
) -> FootprintCorrectedData[Run]:
    """
    Perform the footprint correction on the data array that has a :code:`beam_size` and
    binned :code:`theta` values.

    Parameters
    ----------
    data_array:
        Data array to perform footprint correction on.
    beam_size:
        Full width half maximum of the beam.
    sample_size:
        Size of the sample

    Returns
    -------
    :
       Footprint corrected data array.
    """
    if sample_size is None:
        return FootprintCorrectedData[Run](data_array)
    size_of_beam_on_sample = beam_size / sc.sin(data_array.bins.coords['theta'])
    footprint_scale = sc.erf(fwhm_to_std(sample_size / size_of_beam_on_sample))
    data_array_fp_correction = data_array / footprint_scale
    return FootprintCorrectedData[Run](data_array_fp_correction)


def compute_reference_intensity(
    da: FootprintCorrectedData[Reference], wb: WavelengthBins
) -> HistogrammedReference:
    """Creates a reference intensity map over (z_index, wavelength).
    Rationale:
        The intensity expressed in those variables should not vary
        with the experiment parameters (such as sample rotation).
        Therefore it can be used to normalize sample measurements.
    """
    b = da.bins.concat(set(da.dims) - set(da.coords['z_index'].dims)).bin(wavelength=wb)
    h = b.hist()
    h.masks['too_few_events'] = h.data < sc.scalar(1, unit='counts')
    h.coords['wavelength'] = sc.midpoints(h.coords['wavelength'])
    # Add a Q coordinate to each bin, the Q is not completely unique in every bin,
    # but it is close enough.
    h.coords['Q'] = b.bins.coords['Q'].bins.mean()
    return HistogrammedReference(h)


def calibrate_reference(
    da: HistogrammedReference, cal: SupermirrorReflectivityCorrection
) -> CorrectionMatrix:
    '''Calibrates the reference intensity by the
    inverse of the supermirror reflectivity'''
    return CorrectionMatrix(da * cal)


providers = (
    footprint_correction,
    calibrate_reference,
    compute_reference_intensity,
)
