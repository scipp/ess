# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for computing real neutron time-of-flight for indirect geometry."""

from collections.abc import Iterable

import sciline
import scippnexus as snx

from ess.reduce import unwrap as reduce_unwrap
from ess.reduce.unwrap.types import DetectorLtotal

from ..types import (
    DataAtSample,
    ErrorLimitedLookupTable,
    LookupTable,
    LookupTableRelativeErrorThreshold,
    MonitorCoordTransformGraph,
    MonitorLtotal,
    MonitorType,
    NeXusDetectorName,
    PulseStrideOffset,
    RawDetector,
    RawMonitor,
    RunType,
    WavelengthDetector,
    WavelengthMonitor,
)


def TofWorkflow(
    *,
    run_types: Iterable[sciline.typing.Key],
    monitor_types: Iterable[sciline.typing.Key],
) -> sciline.Pipeline:
    workflow = reduce_unwrap.GenericUnwrapWorkflow(
        run_types=run_types,
        monitor_types=monitor_types,
    )
    for provider in providers:
        workflow.insert(provider)
    return workflow


def detector_wavelength_data(
    sample_data: DataAtSample[RunType],
    lookup: ErrorLimitedLookupTable[snx.NXdetector],
    pulse_stride_offset: PulseStrideOffset,
) -> WavelengthDetector[RunType]:
    """
    Convert the time-of-arrival data to wavelength data using a lookup table.

    The output data will have a wavelength coordinate.

    This is a wrapper around
    :func:`ess.reduce.unwrap.detector_wavelength_data`
    for indirect geometry spectrometers.
    """
    result = reduce_unwrap.to_wavelength.detector_wavelength_data(
        detector_data=RawDetector[RunType](sample_data),
        lookup=lookup,
        ltotal=DetectorLtotal(sample_data.coords['L1']),
        pulse_stride_offset=pulse_stride_offset,
    )
    # This is the incident wavelength at the sample.
    result.bins.coords['incident_wavelength'] = result.bins.coords.pop('wavelength')
    del result.bins.coords['event_time_zero']
    return result


def monitor_wavelength_data(
    monitor_data: RawMonitor[RunType, MonitorType],
    lookup: ErrorLimitedLookupTable[MonitorType],
    ltotal: MonitorLtotal[RunType, MonitorType],
    pulse_stride_offset: PulseStrideOffset,
) -> WavelengthMonitor[RunType, MonitorType]:
    """
    Convert the time-of-arrival data to wavelength data using a lookup table.

    The output data will have a wavelength coordinate.

    This is a wrapper around
    :func:`ess.reduce.unwrap.monitor_wavelength_data`
    for indirect geometry spectrometers.
    """
    result = reduce_unwrap.to_wavelength.monitor_wavelength_data(
        monitor_data=monitor_data.rename(t='frame_time'),
        lookup=lookup,
        ltotal=ltotal,
        pulse_stride_offset=pulse_stride_offset,
    )
    result = result.rename(wavelength='incident_wavelength')
    return result


def compute_monitor_ltotal(
    monitor_data: RawMonitor[RunType, MonitorType],
    coord_transform_graph: MonitorCoordTransformGraph,
) -> MonitorLtotal[RunType, MonitorType]:
    """Compute the path length from the source to the monitor."""
    return MonitorLtotal[RunType, MonitorType](
        monitor_data.transform_coords(
            'Ltotal',
            graph=coord_transform_graph,
            keep_intermediate=False,
            keep_aliases=False,
            rename_dims=False,
        ).coords['Ltotal']
    )


def mask_large_uncertainty_in_lut_detector(
    table: LookupTable,
    error_threshold: LookupTableRelativeErrorThreshold,
) -> ErrorLimitedLookupTable[snx.NXdetector]:
    """
    Mask regions in the wavelength lookup table with large uncertainty using NaNs.

    The threshold is looked up under the key ``'detector'``.
    The same threshold is applied to all triplets.

    Parameters
    ----------
    table:
        Lookup table with wavelength as a function of distance and time-of-arrival.
    error_threshold:
        Threshold for the relative standard deviation (coefficient of variation) of the
        projected wavelength above which values are masked.

    See also
    --------
    essreduce.unwrap.mask_large_uncertainty_in_lut:
        The underlying implementation.
    """
    from ess.reduce.unwrap.to_wavelength import (
        mask_large_uncertainty_in_lut_detector,
    )

    return ErrorLimitedLookupTable[snx.NXdetector](
        mask_large_uncertainty_in_lut_detector(
            table=table,
            error_threshold=error_threshold,
            detector_name=NeXusDetectorName('detector'),
        )
    )


providers = (
    compute_monitor_ltotal,
    detector_wavelength_data,
    mask_large_uncertainty_in_lut_detector,
    monitor_wavelength_data,
)
"""Providers for wavelength calculation on indirect geometry spectrometers.

The providers here override the default providers of
:class:`ess.reduce.unwrap.GenericUnwrapWorkflow`
to customize the workflow for indirect geometry spectrometers.
"""
