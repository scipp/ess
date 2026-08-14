from ess.packages.mlztexture.src.texture.types import (
    GammaBinWidth,
    TwoThetaBinWidth,
    DspacingBinWidth,
    BinningSettings,
    DspacingCylSurfaceDetector,
    GammaBinningData,
    TwoThetaBinningData,
    DspacingBinningData,
    Texture3D,
    DetectorView,
    FlatDetectorView,
    TwoThetaRings,
    GammaSlices,
    EmptyCanSubtracted3D,
    NormalisedTexture3D
)

from ess.packages.essdiffraction.src.ess.powder.types import (
    SampleRun,
    EmptyCanRun,
    VanadiumRun,
    EmptyVanadiumCanRun
)

from typing import cast


def subtract_empty_can_from_sample(
    data: Texture3D[SampleRun],
    background: Texture3D[EmptyCanRun],
) -> EmptyCanSubtracted3D[SampleRun]:
    if data is None:
        return cast(EmptyCanSubtracted3D[SampleRun], None)
    if background is None:
        return EmptyCanSubtracted3D[SampleRun](data)
    # What if negative values after subtr?
    return EmptyCanSubtracted3D[SampleRun](data - background)


def subtract_empty_can_from_vanadium(
    data: Texture3D[VanadiumRun],
    background: Texture3D[EmptyVanadiumCanRun],
) -> EmptyCanSubtracted3D[VanadiumRun]:
    if data is None:
        return cast(EmptyCanSubtracted3D[VanadiumRun], None)
    if background is None:
        return EmptyCanSubtracted3D[VanadiumRun](data)
    # What if negative values after subtr?
    return EmptyCanSubtracted3D[VanadiumRun](data - background)


def normalize_by_vanadium(
    data: EmptyCanSubtracted3D[SampleRun],
    vanadium: EmptyCanSubtracted3D[VanadiumRun],
    # uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> NormalisedTexture3D:
    if data is None:
        return cast(NormalisedTexture3D, None)
    if vanadium is None:
        return NormalisedTexture3D(data)
    # norm = broadcast_uncertainties(
    #     vanadium, prototype=data, mode=uncertainty_broadcast_mode
    # )
    # Convert the unit such that the end-result has unit 'one' because the division
    # might otherwise produce a unit with a large scale if the proton charges in data
    # and vanadium were measured with different units.
    # norm = norm.to(unit=data.unit, copy=False) <- Seems not needed in our worlflow
    norm = vanadium
    normalized = data / norm
    # mask = norm.data == sc.scalar(0.0, unit=norm.unit)
    # if mask.any():
    #     normalized.masks['zero_vanadium'] = mask
    return NormalisedTexture3D(normalized)
