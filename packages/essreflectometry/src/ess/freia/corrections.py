# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline
import scipp as sc
from ess.reduce.uncertainty import UncertaintyBroadcastMode

from ..reflectometry import corrections as common_corrections
from ..reflectometry.corrections import RunNormalization
from ..reflectometry.types import (
    BeamSize,
    ReducibleData,
    RunType,
    RunUnnormalizedData,
    Sample,
    SampleRun,
    SampleSize,
)
from .types import SampleIlluminatedFraction, WavelengthMonitor


def normalize_by_monitor_histogram(
    detector: RunUnnormalizedData[RunType],
    *,
    monitor: WavelengthMonitor[RunType],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> ReducibleData[RunType]:
    """Normalize detector data by a histogrammed monitor.

    The detector is normalized according to

    .. math::

        d_i^\\text{Norm} = \\frac{d_i}{m_i} \\Delta \\lambda_i

    Parameters
    ----------
    detector:
        Input event data in wavelength.
    monitor:
        A histogrammed monitor in wavelength.
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor are broadcast to the sample data.

    Returns
    -------
    :
        `detector` normalized by a monitor.

    See also
    --------
    ess.reduce.normalization.normalize_by_monitor_histogram:
        For details and the actual implementation.
    """
    return common_corrections.normalize_by_monitor_histogram(
        detector=detector,
        monitor=monitor,
        uncertainty_broadcast_mode=uncertainty_broadcast_mode,
    )


def normalize_by_monitor_integrated(
    detector: RunUnnormalizedData[RunType],
    *,
    monitor: WavelengthMonitor[RunType],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> ReducibleData[RunType]:
    """Normalize detector data by an integrated wavelength monitor."""
    return common_corrections.normalize_by_monitor_integrated(
        detector=detector,
        monitor=monitor,
        uncertainty_broadcast_mode=uncertainty_broadcast_mode,
    )


def insert_run_normalization(
    workflow: sciline.Pipeline, run_norm: RunNormalization
) -> None:
    """Insert providers for a specific FREIA run normalization into a workflow."""
    common_corrections.insert_run_normalization(
        workflow,
        run_norm,
        monitor_histogram_provider=normalize_by_monitor_histogram,
        monitor_integrated_provider=normalize_by_monitor_integrated,
    )


def sample_illuminated_fraction(
    sample: ReducibleData[SampleRun],
    beam_size: BeamSize[SampleRun],
    sample_size: SampleSize[SampleRun],
) -> SampleIlluminatedFraction:
    """Use Amor's Gaussian footprint model with the specular incidence angle.

    Beam size is the FWHM at the sample; sample size is its length along the
    beam. These must be supplied explicitly, since slit openings alone do not
    determine the profile of the beam reaching the sample.
    """
    for name, size in [('BeamSize', beam_size), ('SampleSize', sample_size)]:
        if (
            not sc.isfinite(size).value
            or not (size > sc.scalar(0.0, unit=size.unit)).value
        ):
            raise ValueError(f'{name} must be finite and positive.')
    return SampleIlluminatedFraction(
        common_corrections.footprint_on_sample(
            sample.bins.coords['theta'], beam_size=beam_size, sample_size=sample_size
        )
    )


def prepare_sample(
    sample: ReducibleData[SampleRun],
    illuminated_fraction: SampleIlluminatedFraction,
) -> Sample:
    """Correct the reflected beam for footprint; the direct beam has no sample."""
    if illuminated_fraction.bins is None:
        illuminated_fraction = sc.bins_like(sample, illuminated_fraction)
    valid = sc.isfinite(illuminated_fraction) & (illuminated_fraction > sc.scalar(0.0))
    valid &= illuminated_fraction <= sc.scalar(1.0)
    sample = sample.bins.assign_masks(
        footprint=~valid,
        non_reflected=sample.bins.coords['theta'] <= sc.scalar(0.0, unit='rad'),
    )
    fraction = sc.where(valid, illuminated_fraction, sc.scalar(1.0))
    return Sample(sample / fraction)


providers = (sample_illuminated_fraction, prepare_sample)
