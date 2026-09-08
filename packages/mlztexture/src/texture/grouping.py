import scipp as sc


from ess.packages.mlztexture.src.texture.types import (
    TwoThetaRings,
    GammaSlices,
    NormalisedTexture3D,
    GammaSliceWidth,
    DspacingHist,
    TwoThetaHist,
    IntegratedTwoThetaHist,
    ProcessedData
)


def view_2d_rings(
    data: NormalisedTexture3D,
) -> TwoThetaRings:
    """
    ToDo
    """
    out = data
    return TwoThetaRings(out)


def view_1d_gamma_slices(
    data: NormalisedTexture3D,
) -> GammaSlices:
    """
    ToDo
    """
    out = data
    return GammaSlices(out)


def generate_dspacing_hist(
    #data: NormalisedTexture3D,
    gamma_bin_width: GammaSliceWidth,
    dummy_gamma_view: GammaSlices,
    dummy_theta_view: TwoThetaRings,
) -> DspacingHist:
    DspacingHist(dummy_gamma_view)


def generate_two_theta_hist(
    #data: NormalisedTexture3D,
    gamma_bin_width: GammaSliceWidth,
    dummy_gamma_view: GammaSlices,
    dummy_theta_view: TwoThetaRings,
) -> TwoThetaHist:
    return TwoThetaHist(dummy_gamma_view)


def generate_integrated_two_theta_hist(
    #data: NormalisedTexture3D,
    gamma_bin_width: GammaSliceWidth,
    dummy_gamma_view: GammaSlices,
    dummy_theta_view: TwoThetaRings,
) -> IntegratedTwoThetaHist:
    return IntegratedTwoThetaHist(dummy_gamma_view)


def export_data(
    data_1: DspacingHist,
    data_2: TwoThetaHist,
    data_3: IntegratedTwoThetaHist
) -> ProcessedData:
    return ProcessedData(data_1)