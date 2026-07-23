# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""CIF writer for DREAM."""

import scipp as sc
from ess.powder.calibration import OutputCalibrationData
from ess.powder.types import (
    Beamline,
    CIFAuthors,
    EmptyCanSubtractedIntensityTof,
    IntensityTof,
    Measurement,
    ReducedEmptyCanSubtractedTofCIF,
    ReducedTofCIF,
    ReducerSoftware,
    SampleRun,
    Source,
)
from scipp.core import irreducible_mask
from scippneutron.io import cif


def prepare_reduced_tof_cif(
    da: IntensityTof,
    *,
    authors: CIFAuthors,
    beamline: Beamline[SampleRun],
    source: Source[SampleRun],
    measurement: Measurement[SampleRun],
    reducers: ReducerSoftware,
    calibration: OutputCalibrationData,
) -> ReducedTofCIF:
    """Construct a CIF builder with reduced data in d-spacing.

    The object contains the d-spacing coordinate, intensities,
    and some metadata.

    Parameters
    ----------
    da:
        Reduced 1d data with a ``'tof'`` dimension and coordinate.
    authors:
        List of authors to write to the file.
    beamline:
        Information about the beamline that the data was produced at.
    source:
        Information about the neutron source.
    reducers:
        List of software pieces used to reduce the data.
    calibration:
        Coefficients for conversion between d-spacing and final ToF.
        See :meth:`scippneutron.io.cif.CIF.with_powder_calibration`.

    Returns
    -------
    :
        An object that contains the reduced data and metadata.
        Us its ``save`` method to write the CIF file.
    """
    return _prepare_reduced_tof_cif_impl(
        da,
        authors=authors,
        beamline=beamline,
        source=source,
        measurement=measurement,
        reducers=reducers,
        calibration=calibration,
    )


def prepare_reduced_empty_can_subtracted_tof_cif(
    da: EmptyCanSubtractedIntensityTof,
    *,
    authors: CIFAuthors,
    beamline: Beamline[SampleRun],
    source: Source[SampleRun],
    measurement: Measurement[SampleRun],
    reducers: ReducerSoftware,
    calibration: OutputCalibrationData,
) -> ReducedEmptyCanSubtractedTofCIF:
    """Construct a CIF builder with reduced data in d-spacing.

    The object contains the d-spacing coordinate, intensities,
    and some metadata.

    Parameters
    ----------
    da:
        Reduced 1d data with a ``'tof'`` dimension and coordinate.
    authors:
        List of authors to write to the file.
    beamline:
        Information about the beamline that the data was produced at.
    source:
        Information about the neutron source.
    reducers:
        List of software pieces used to reduce the data.
    calibration:
        Coefficients for conversion between d-spacing and final ToF.
        See :meth:`scippneutron.io.cif.CIF.with_powder_calibration`.

    Returns
    -------
    :
        An object that contains the reduced data and metadata.
        Us its ``save`` method to write the CIF file.
    """
    return _prepare_reduced_tof_cif_impl(
        da,
        authors=authors,
        beamline=beamline,
        source=source,
        measurement=measurement,
        reducers=reducers,
        calibration=calibration,
    )


def _prepare_reduced_tof_cif_impl(
    da: IntensityTof,
    *,
    authors: CIFAuthors,
    beamline: Beamline[SampleRun],
    source: Source[SampleRun],
    measurement: Measurement[SampleRun],
    reducers: ReducerSoftware,
    calibration: OutputCalibrationData,
) -> ReducedTofCIF:
    to_save = _prepare_data(da)
    return ReducedTofCIF(
        cif.CIF('reduced_tof')
        .with_measurement(measurement)
        .with_reducers(*(reducer.compact_repr for reducer in reducers))
        .with_authors(*authors)
        .with_beamline(beamline, source)
        .with_powder_calibration(calibration.to_cif_format())
        .with_reduced_powder_data(to_save)
    )


def _prepare_data(da: sc.DataArray) -> sc.DataArray:
    if da.ndim != 1:
        raise sc.DimensionError(f"Can only save 1D data, got {da.sizes}")

    hist = da.hist() if da.is_binned else da.copy(deep=False)
    hist.coords[hist.dim] = sc.midpoints(hist.coords[hist.dim])

    if hist.masks:
        # No file format we use here supports masks, so the next
        # best thing is to zero out masked data:
        hist.data = hist.data.copy()
        hist.values *= irreducible_mask(hist.masks, hist.dim).values

    return hist
