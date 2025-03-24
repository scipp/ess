import scipp as sc
import scippnexus as snx

from ess.spectroscopy.types import (
    BeamlineWithSpectrometerCoords,
    CalibratedDetector,
    DetectorPositionOffset,
    NeXusComponent,
    NeXusTransformation,
    RunType,
)

from .types import ArcNumber


def arc_number(
    beamline: BeamlineWithSpectrometerCoords[RunType],
) -> ArcNumber[RunType]:
    """Calculate BIFROST arc index number from pixel final energy

    The BIFROST analyzers are each set to diffract an
    energy in the set (2.7, 3.2, 3.8, 4.4. 5.0) meV.
    This energy is only valid for the central point of the center
    tube of the associated detector triplet. All other pixels
    will have a final energy slightly higher or lower.

    This function assigns the closest arc number indexing the
    ordered set above.

    Parameters
    ----------
    beamline:
        A data array with a 'final_energy' coordinate which is the
        per-pixel (or event) final neutron energy.

    Returns
    -------
    :
        The arc index of the analyzer from which the neutron scattered
    """
    minimum = sc.scalar(2.7, unit='meV')
    step = sc.scalar(0.575, unit='meV')
    final_energy = beamline.coords['final_energy']
    return ArcNumber[RunType](sc.round((final_energy - minimum) / step).to(dtype='int'))


def get_calibrated_detector_bifrost(
    detector: NeXusComponent[snx.NXdetector, RunType],
    *,
    transform: NeXusTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
) -> CalibratedDetector[RunType]:
    """Extract the data array corresponding to a detector's signal field.

    The returned data array includes coords and masks pertaining directly to the
    signal values array, but not additional information about the detector.
    The data array is reshaped to the logical detector shape.

    This function is specific to BIFROST and differs from the generic
    :func:`ess.reduce.nexus.workflow.get_calibrated_detector` in that it does not
    fold the detectors into logical dimensions because the files already contain
    the detectors in the correct shape.

    Parameters
    ----------
    detector:
        Loaded NeXus detector.
    transform:
        Transformation that determines the detector position.
    offset:
        Offset to add to the detector position.

    Returns
    -------
    :
        Detector data.
    """

    from ess.reduce.nexus.types import DetectorBankSizes
    from ess.reduce.nexus.workflow import get_calibrated_detector

    da = get_calibrated_detector(
        detector=detector,
        transform=transform,
        offset=offset,
        # The detectors are folded in the file, no need to do that here.
        bank_sizes=DetectorBankSizes({}),
    )
    da = da.rename(dim_0='tube', dim_1='length')
    return CalibratedDetector[RunType](da)


def merge_triplets(
    *triplets: sc.DataArray,
) -> sc.DataArray:
    """Merge BIFROST detector triplets into a single data array.

    Parameters
    ----------
    triplets:
        Data arrays to merge.

    Returns
    -------
    :
        Input data arrays stacked along the "triplet" dimension.
    """
    return sc.concat(triplets, dim="triplet")


providers = (arc_number, get_calibrated_detector_bifrost)
