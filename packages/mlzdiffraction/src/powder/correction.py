# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Correction algorithms for POWTEX powder reduction."""

from typing import cast

import scipp as sc

from ess.reduce.uncertainty import broadcast_uncertainties

from .types import (
    AccumulatedProtonCharge,
    Dspacing2D,
    EmptyCanRun,
    EmptyVanadiumCanRun,
    EmptyCanSubtracted2D,
    EmptyCanSubtractedNoNegative2D,
    NegativeIntensityResetValue,
    NormalizedDspacing,
    NormalizedIntensity2D,
    RunType,
    SampleRun,
    ShiftNegativeIntensity,
    UncertaintyBroadcastMode,
    VanadiumRun,
)


def normalize_by_vanadium_mlz(
    data: EmptyCanSubtractedNoNegative2D[SampleRun],
    vanadium: EmptyCanSubtractedNoNegative2D[VanadiumRun],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> NormalizedIntensity2D:
    """Normalize sample intensity by a vanadium measurement."""
    if data is None:
        return cast(NormalizedIntensity2D, None)
    if vanadium is None:
        return NormalizedIntensity2D(data)
    norm = broadcast_uncertainties(
        vanadium, prototype=data, mode=uncertainty_broadcast_mode
    )
    # Convert the unit such that the end-result has unit 'one' because the division
    # might otherwise produce a unit with a large scale if the proton charges in data
    # and vanadium were measured with different units.
    # norm = norm.to(unit=data.unit, copy=False) <- Seems not needed in our worlflow
    normalized = data / norm
    mask = norm.data == sc.scalar(0.0, unit=norm.unit)
    if mask.any():
        normalized.masks["zero_vanadium"] = mask
    return NormalizedIntensity2D(normalized)


def normalize_by_proton_charge_mlz(
    data: Dspacing2D[RunType],
    proton_charge: AccumulatedProtonCharge[RunType],
) -> NormalizedDspacing[RunType]:
    """Normalize data by accumulated proton charge."""
    if data is None:
        return cast(NormalizedDspacing[RunType], None)
    return NormalizedDspacing[RunType](data / proton_charge)


def subtract_sample_empty_can_mlz(
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


def subtract_vanadium_empty_can_mlz(
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
) -> EmptyCanSubtractedNoNegative2D[SampleRun]:
    if data is None:
        return cast(EmptyCanSubtractedNoNegative2D[SampleRun], None)
    if shift_flag:
        min_intensity = sc.min(data)
        min_value = min_intensity.value
        if min_value < 0.0:
            offset = sc.scalar(-min_value, unit=min_intensity.unit)
            print(f"Sample: negative intensity values are SHIFTED by {min_value}.")
            return EmptyCanSubtractedNoNegative2D[SampleRun](data + offset)
    else:
        if reset_value is not None:
            print(f"Sample: negative intensity values are REPLACED by {reset_value}.")
            data.values[data.values < 0.0] = reset_value
            # Q: What to do with uncertainties? For now, we just keep them as they are.
    return EmptyCanSubtractedNoNegative2D[SampleRun](data)


def handle_negative_intensity_vanadium(
    data: EmptyCanSubtracted2D[VanadiumRun],
    shift_flag: ShiftNegativeIntensity[VanadiumRun],
    reset_value: NegativeIntensityResetValue[VanadiumRun],
) -> EmptyCanSubtractedNoNegative2D[VanadiumRun]:
    if data is None:
        return cast(EmptyCanSubtractedNoNegative2D[VanadiumRun], None)
    if shift_flag:
        min_intensity = sc.min(data)
        min_value = min_intensity.value
        if min_value < 0.0:
            offset = sc.scalar(-min_value, unit=min_intensity.unit)
            print(f"Vanadium: negative intensity values are SHIFTED by {min_value}.")
            return EmptyCanSubtractedNoNegative2D[VanadiumRun](data + offset)
    else:
        if reset_value is not None:
            print(f"Vanadium: negative intensity values are REPLACED by {reset_value}.")
            data.values[data.values < 0.0] = reset_value
            # Q: What to do with uncertainties? For now, we just keep them as they are.
    return EmptyCanSubtractedNoNegative2D[VanadiumRun](data)


providers = (
    handle_negative_intensity_sample,
    handle_negative_intensity_vanadium,
    normalize_by_proton_charge_mlz,
    normalize_by_vanadium_mlz,
    subtract_sample_empty_can_mlz,
    subtract_vanadium_empty_can_mlz,
)
