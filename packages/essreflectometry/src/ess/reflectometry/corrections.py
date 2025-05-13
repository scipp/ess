# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from .tools import fwhm_to_std
from .types import RawSampleRotation, RunType, SampleRotation, SampleRotationOffset


def footprint_on_sample(
    theta: sc.Variable,
    beam_size: sc.Variable,
    sample_size: sc.Variable,
) -> sc.Variable:
    """
    Computes the fraction of the beam hitting the sample.
    Depends on :math:`\\theta`.

    Parameters
    ----------
    theta:
        Incidence angle relative to sample surface.
    beam_size:
        Full width half maximum of the beam.
    sample_size:
        Size of the sample, width in the beam direction.

    Returns
    -------
    :
       Fraction of beam hitting the sample.
    """
    size_of_beam_on_sample = beam_size / sc.sin(theta)
    return sc.erf(
        fwhm_to_std((sample_size / size_of_beam_on_sample).to(unit='dimensionless'))
    )


def correct_by_footprint(da: sc.DataArray) -> sc.DataArray:
    "Corrects the data by the size of the footprint on the sample."
    return da / footprint_on_sample(
        da.bins.coords['theta'] if 'theta' in da.bins.coords else da.coords['theta'],
        da.coords['beam_size'],
        da.coords['sample_size'],
    )


def correct_by_proton_current(da: sc.DataArray) -> sc.DataArray:
    "Corrects the data by the proton current during the time of data collection"
    return da / da.bins.coords['proton_current']


def correct_sample_rotation(
    mu: RawSampleRotation[RunType], mu_offset: SampleRotationOffset[RunType]
) -> SampleRotation[RunType]:
    return mu + mu_offset.to(unit=mu.unit)


providers = (correct_sample_rotation,)
