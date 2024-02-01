# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict, List, Optional, Union
from uuid import uuid4

import scipp as sc
from scipp.core.concepts import irreducible_mask
from scipp.scipy.interpolate import interp1d

from .common import mask_range
from .logging import get_logger
from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    CleanDirectBeam,
    CleanMonitor,
    CleanQ,
    CleanSummedQ,
    DimsToKeep,
    DirectBeam,
    IofQ,
    IofQPart,
    MonitorType,
    NonBackgroundWavelengthRange,
    QBins,
    RunType,
    SampleRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
    WavelengthMonitor,
)
from .uncertainty import broadcast_with_upper_bound_variances


def preprocess_monitor_data(
    monitor: WavelengthMonitor[RunType, MonitorType],
    wavelength_bins: WavelengthBins,
    non_background_range: Optional[NonBackgroundWavelengthRange],
    uncertainties: UncertaintyBroadcastMode,
) -> CleanMonitor[RunType, MonitorType]:
    """
    Prepare monitor data for computing the transmission fraction.
    The input data are first converted to wavelength (if needed).
    If a ``non_background_range`` is provided, it defines the region where data is
    considered not to be background, and regions outside are background. A mean
    background level will be computed from the background and will be subtracted from
    the non-background counts.
    Finally, if wavelength bins are provided, the data is rebinned to match the
    requested binning.

    Parameters
    ----------
    monitor:
        The monitor to be pre-processed.
    wavelength_bins:
        The binning in wavelength to use for the rebinning.
    non_background_range:
        The range of wavelengths that defines the data which does not constitute
        background. Everything outside this range is treated as background counts.
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        The input monitors converted to wavelength, cleaned of background counts, and
        rebinned to the requested wavelength binning.
    """
    background = None
    if non_background_range is not None:
        mask = sc.DataArray(
            data=sc.array(dims=[non_background_range.dim], values=[True]),
            coords={non_background_range.dim: non_background_range},
        )
        background = mask_range(monitor, mask=mask).mean()

    if monitor.bins is not None:
        monitor = monitor.hist(wavelength=wavelength_bins)
    else:
        monitor = monitor.rebin(wavelength=wavelength_bins)

    if background is not None:
        if uncertainties == UncertaintyBroadcastMode.drop:
            monitor -= sc.values(background)
        elif uncertainties == UncertaintyBroadcastMode.upper_bound:
            monitor -= broadcast_with_upper_bound_variances(
                background, sizes=monitor.sizes
            )
        else:
            monitor -= background
    return CleanMonitor(monitor)


def resample_direct_beam(
    direct_beam: DirectBeam, wavelength_bins: WavelengthBins
) -> CleanDirectBeam:
    """
    If the wavelength binning of the direct beam function does not match the requested
    ``wavelength_bins``, perform a 1d interpolation of the function onto the bins.

    Parameters
    ----------
    direct_beam:
        The DataArray containing the direct beam function (it should have a dimension
        of wavelength).
    wavelength_bins:
        The binning in wavelength that the direct beam function should be resampled to.

    Returns
    -------
    :
        The direct beam function resampled to the requested resolution.
    """
    if sc.identical(direct_beam.coords['wavelength'], wavelength_bins):
        return direct_beam
    if direct_beam.variances is not None:
        logger = get_logger('sans')
        logger.warning(
            'An interpolation is being performed on the direct_beam function. '
            'The variances in the direct_beam function will be dropped.'
        )
    func = interp1d(
        sc.values(direct_beam),
        'wavelength',
        fill_value="extrapolate",
        bounds_error=False,
    )
    return CleanDirectBeam(func(wavelength_bins, midpoints=True))


def _process_wavelength_bands(
    wavelength_bands: Optional[WavelengthBands],
    wavelength_bins: WavelengthBins,
) -> Optional[WavelengthBands]:
    """
    Perform some checks and potential reshaping on the wavelength bands.

    The wavelength bands must be either one- or two-dimensional.
    If the wavelength bands are defined as a one-dimensional array, convert them to a
    two-dimensional array with start and end wavelengths.

    The final bands must have a size of 2 in the wavelength dimension, defining a start
    and an end wavelength.
    """
    if wavelength_bands is None:
        wavelength_bands = sc.concat(
            [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength'
        )
    if wavelength_bands.ndim == 1:
        wavelength_bands = sc.concat(
            [wavelength_bands[:-1], wavelength_bands[1:]], dim='x'
        ).rename(x='wavelength', wavelength='band')
    if wavelength_bands.ndim != 2:
        raise ValueError(
            'Wavelength_bands must be one- or two-dimensional, '
            f'got {wavelength_bands.ndim}.'
        )
    if wavelength_bands.sizes['wavelength'] != 2:
        raise ValueError(
            'Wavelength_bands must have a size of 2 in the wavelength dimension, '
            'defining a start and an end wavelength, '
            f'got {wavelength_bands.sizes["wavelength"]}.'
        )
    return wavelength_bands


def merge_spectra(
    data: CleanQ[RunType, IofQPart],
    q_bins: QBins,
    wavelength_bins: WavelengthBins,
    wavelength_bands: Optional[WavelengthBands],
    dims_to_keep: Optional[DimsToKeep],
) -> CleanSummedQ[RunType, IofQPart]:
    """
    Merges all spectra:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    q_bins:
        The binning in Q to be used.
    wavelength_bins:
        The binning in wavelength to be used.
    wavelength_bands:
        Defines bands in wavelength that can be used to separate different wavelength
        ranges that contribute to different regions in Q space. Note that this needs to
        be defined, so if all wavelengths should be used, this should simply be a start
        and end edges that encompass the entire wavelength range.
    dims_to_keep:
        Dimensions that should not be reduced and thus still be present in the final
        I(Q) result (this is typically the layer dimension).

    Returns
    -------
    :
        The input data converted to Q and then summed over all detector pixels.
    """
    dims_to_reduce = set(data.dims) - {'Q'}
    if dims_to_keep is not None:
        dims_to_reduce -= set(dims_to_keep)

    wavelength_bands = _process_wavelength_bands(
        wavelength_bands=wavelength_bands, wavelength_bins=wavelength_bins
    )

    if data.bins is not None:
        out = _events_merge_spectra(
            data_q=data,
            q_bins=q_bins,
            wavelength_bands=wavelength_bands,
            dims_to_reduce=dims_to_reduce,
        )
    else:
        out = _dense_merge_spectra(
            data_q=data,
            q_bins=q_bins,
            wavelength_bands=wavelength_bands,
            dims_to_reduce=dims_to_reduce,
        )
    return CleanSummedQ[RunType, IofQPart](out.squeeze())


def _to_q_bins(q_bins: Union[int, sc.Variable]) -> Dict[str, Union[int, sc.Variable]]:
    """
    If the input bins are an integer, convert them to a dictionary that can be used
    to bin a DataArray.
    """
    if isinstance(q_bins, int):
        return {'Q': q_bins}
    return {q_bins.dim: q_bins}


def _events_merge_spectra(
    data_q: sc.DataArray,
    q_bins: Union[int, sc.Variable],
    dims_to_reduce: List[str],
    wavelength_bands: sc.Variable,
) -> sc.DataArray:
    """
    Merge spectra of event data
    """
    q_all_pixels = data_q.bins.concat(dims_to_reduce)
    edges = _to_q_bins(q_bins)
    q_binned = q_all_pixels.bin(**edges)
    dim = 'wavelength'
    wav_binned = q_binned.bin({dim: sc.sort(wavelength_bands.flatten(to=dim), dim)})
    # At this point we kind of already have what we need, would be cheapest to just
    # return, if follow up providers can work with the result.
    # Otherwise we need to duplicate events:
    sections = []
    for bounds in sc.collapse(wavelength_bands, keep=dim).values():
        # The extra concat can probably be avoided if we insert some dummy edges for
        # first and last band, but we would need to know how many edges to insert, as
        # the bands can be very wide and overlap by more than one bin.
        sections.append(wav_binned[dim, bounds[0] : bounds[1]].bins.concat(dim))
    band_dim = (set(wavelength_bands.dims) - {'wavelength'}).pop()
    out = sc.concat(sections, band_dim)
    out.coords[dim] = wavelength_bands
    return out


def _dense_merge_spectra(
    data_q: sc.DataArray,
    q_bins: Union[int, sc.Variable],
    dims_to_reduce: List[str],
    wavelength_bands: sc.Variable,
) -> sc.DataArray:
    """
    Merge spectra of dense data
    """
    edges = _to_q_bins(q_bins)
    bands = []
    band_dim = (set(wavelength_bands.dims) - {'wavelength'}).pop()

    # We want to flatten data to make histogramming cheaper (avoiding allocation of
    # large output before summing). We strip unnecessary content since it makes
    # flattening more expensive.
    stripped = data_q.copy(deep=False)
    for name, coord in data_q.coords.items():
        if name not in ['Q', 'wavelength'] and any(
            [dim in dims_to_reduce for dim in coord.dims]
        ):
            del stripped.coords[name]
    to_flatten = [dim for dim in data_q.dims if dim in dims_to_reduce]

    dummy_dim = str(uuid4())
    flat = stripped.flatten(dims=to_flatten, to=dummy_dim)

    # Apply masks once, to avoid repeated work when iterating over bands
    mask = irreducible_mask(flat, dummy_dim)
    # When not all dims are reduced there may be extra dims in the mask and it is not
    # possible to select data based on it. In this case the masks will be applied
    # in the loop below, which is slightly slower.
    if mask.ndim == 1:
        flat = flat.drop_masks(
            [name for name, mask in flat.masks.items() if dummy_dim in mask.dims]
        )
        flat = flat[~mask]

    dims_to_reduce = tuple(dim for dim in dims_to_reduce if dim not in to_flatten)
    for wav_range in sc.collapse(wavelength_bands, keep='wavelength').values():
        band = flat['wavelength', wav_range[0] : wav_range[1]]
        # By flattening before histogramming we avoid allocating a large output array,
        # which would then require summing over all pixels.
        bands.append(band.flatten(dims=(dummy_dim, 'Q'), to='Q').hist(**edges))
    return sc.concat(bands, band_dim)


def subtract_background(
    sample: IofQ[SampleRun], background: IofQ[BackgroundRun]
) -> BackgroundSubtractedIofQ:
    if sample.bins is not None:
        sample = sample.bins.sum()
    if background.bins is not None:
        background = background.bins.sum()
    return BackgroundSubtractedIofQ(sample - background)


providers = (
    preprocess_monitor_data,
    resample_direct_beam,
    merge_spectra,
    subtract_background,
)
