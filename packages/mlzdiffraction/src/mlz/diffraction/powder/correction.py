# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Correction algorithms for POWTEX powder reduction."""

import logging
from typing import cast

import scipp as sc

from ess.reduce.uncertainty import broadcast_uncertainties

from ..powtex.types import (
    AccumulatedProtonCharge,
    EmptyCanRun,
    EmptyVanadiumCanRun,
    NegativeIntensityResetValue,
    NormalisedDspacing,
    RunType,
    SampleRun,
    ShiftNegativeIntensity,
    UncertaintyBroadcastMode,
    VanadiumRun,
)
from .types import (
    Dspacing2D,
    EmptyCanSubtracted2D,
    AdjustedIntensity2D,
    NormalisedIntensity2D,
)

logger = logging.getLogger(__name__)


def normalise_by_vanadium(
    data: AdjustedIntensity2D[SampleRun],
    vanadium: AdjustedIntensity2D[VanadiumRun],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> NormalisedIntensity2D:
    """Normalise sample intensity by a vanadium measurement."""
    if data is None:
        return cast(NormalisedIntensity2D, None)
    if vanadium is None:
        return NormalisedIntensity2D(data)
    norm = broadcast_uncertainties(
        vanadium, prototype=data, mode=uncertainty_broadcast_mode
    )
    # Convert the unit such that the end-result has unit 'one' because the division
    # might otherwise produce a unit with a large scale if the proton charges in data
    # and vanadium were measured with different units.
    # norm = norm.to(unit=data.unit, copy=False) <- Seems not needed in our worlflow
    normalised = data / norm
    mask = norm.data == sc.scalar(0.0, unit=norm.unit)
    if mask.any():
        normalised.masks["zero_vanadium"] = mask
    return NormalisedIntensity2D(normalised)


def normalize_by_proton_charge(
    data: Dspacing2D[RunType],
    proton_charge: AccumulatedProtonCharge[RunType],
) -> NormalisedDspacing[RunType]:
    """Normalize data by accumulated proton charge."""
    if data is None:
        return cast(NormalisedDspacing[RunType], None)
    return NormalisedDspacing[RunType](data / proton_charge)


def subtract_sample_empty_can(
    data: Dspacing2D[SampleRun],
    background: Dspacing2D[EmptyCanRun],
) -> EmptyCanSubtracted2D[SampleRun]:
    """Subtract the sample empty-can run from the sample data."""
    if data is None:
        return cast(EmptyCanSubtracted2D[SampleRun], None)
    if background is None:
        return EmptyCanSubtracted2D[SampleRun](data)
    if data.bins is None:
        return EmptyCanSubtracted2D[SampleRun](data - background)
    return EmptyCanSubtracted2D[SampleRun](data.bins.concatenate(-background))


def subtract_vanadium_empty_can(
    data: Dspacing2D[VanadiumRun],
    background: Dspacing2D[EmptyVanadiumCanRun],
) -> EmptyCanSubtracted2D[VanadiumRun]:
    """Subtract the vanadium empty-can run from the vanadium data."""
    if data is None:
        return cast(EmptyCanSubtracted2D[VanadiumRun], None)
    if background is None:
        return EmptyCanSubtracted2D[VanadiumRun](data)
    if data.bins is None:
        return EmptyCanSubtracted2D[VanadiumRun](data - background)
    return EmptyCanSubtracted2D[VanadiumRun](data.bins.concatenate(-background))


def handle_negative_intensity_sample(
    data: EmptyCanSubtracted2D[SampleRun],
    shift_flag: ShiftNegativeIntensity[SampleRun],
    reset_value: NegativeIntensityResetValue[SampleRun],
) -> AdjustedIntensity2D[SampleRun]:
    """Handle negative intensity values in sample data by either shifting them
    by the most negative value or replacing them with a reset value."""
    if data is None:
        return cast(AdjustedIntensity2D[SampleRun], None)
    if shift_flag:
        min_intensity = sc.min(data)
        min_value = min_intensity.value
        if min_value < 0.0:
            offset = sc.scalar(-min_value, unit=min_intensity.unit)
            logger.info("Sample negative intensity values shifted by %s.", min_value)
            return AdjustedIntensity2D[SampleRun](data + offset)
    else:
        if reset_value is not None:
            logger.info("Sample negative intensity values replaced by %s.", reset_value)
            data.values[data.values < 0.0] = reset_value
            # Q: What to do with uncertainties? For now, we just keep them as they are.
    return AdjustedIntensity2D[SampleRun](data)


def handle_negative_intensity_vanadium(
    data: EmptyCanSubtracted2D[VanadiumRun],
    shift_flag: ShiftNegativeIntensity[VanadiumRun],
    reset_value: NegativeIntensityResetValue[VanadiumRun],
) -> AdjustedIntensity2D[VanadiumRun]:
    """Handle negative intensity values in vanadium data by either shifting them
    by the most negative value or replacing them with a reset value."""
    if data is None:
        return cast(AdjustedIntensity2D[VanadiumRun], None)
    if shift_flag:
        min_intensity = sc.min(data)
        min_value = min_intensity.value
        if min_value < 0.0:
            offset = sc.scalar(-min_value, unit=min_intensity.unit)
            logger.info("Vanadium negative intensity values shifted by %s.", min_value)
            return AdjustedIntensity2D[VanadiumRun](data + offset)
    else:
        if reset_value is not None:
            logger.info(
                "Vanadium negative intensity values replaced by %s.", reset_value
            )
            data.values[data.values < 0.0] = reset_value
            # Q: What to do with uncertainties? For now, we just keep them as they are.
    return AdjustedIntensity2D[VanadiumRun](data)


providers = (
    handle_negative_intensity_sample,
    handle_negative_intensity_vanadium,
    normalize_by_proton_charge,
    normalise_by_vanadium,
    subtract_sample_empty_can,
    subtract_vanadium_empty_can,
)
