# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline
import scipp as sc


from ess.packages.essdiffraction.src.ess.powder.types import (
    FilepathMLZ,
    FilenameMLZ,
    SampleRun,
    EmptyCanRun,
    VanadiumRun,
    EmptyVanadiumCanRun,
    CalibrationRun,
)

from ess.packages.mlztexture.src.powtex.data import (
    load_powtex_csv,
)

from ess.packages.mlztexture.src.texture.conversion import (
    project_onto_cylindrical_surface,
    cylindrical_surface_projection_graph,
)

from ess.packages.mlztexture.src.texture.types import (
    BinningSettings,
)

from ess.packages.mlztexture.src.texture.histogramming import (
    generate_gamma_binning_data,
    generate_two_theta_binning_data,
    hist_3d,
)

def default_parameters() -> dict:
    return {
            FilepathMLZ[SampleRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/texture/csv",
            FilepathMLZ[VanadiumRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/texture/csv",
            FilepathMLZ[EmptyCanRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/texture/csv",
            FilepathMLZ[CalibrationRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/texture/csv",
            FilepathMLZ[EmptyVanadiumCanRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/texture/csv",
            BinningSettings: BinningSettings(
                gamma_cut_start=sc.scalar(225.0, unit="deg"),
                gamma_cut_end=sc.scalar(315.0, unit="deg"),
                drop_partial_bins=True,
                include_gamma_gap_bins=False,
                two_theta_min=sc.scalar(45.0, unit="deg"),
                two_theta_max=sc.scalar(135.0, unit="deg")
            ),
    }

def TextureReduce() -> sciline.Pipeline:
    """
    Workflow with default parameters for POWTEX.
    """
    # The package does not depend on pooch which is needed for the tutorial
    # data. Delay import until workflow is actually used.

    return sciline.Pipeline(
        providers=(
            [
                load_powtex_csv,
                project_onto_cylindrical_surface,
                generate_gamma_binning_data,
                generate_two_theta_binning_data,
                hist_3d,
                # load_powgen_calibration_mlz,
                # # sample_position,
                # #sample_position_mlz,
                # # source_position,
                # #source_position_mlz,
                # # coordinate_transformation_graph_mlz,
                # p2d_coordinate_transformation_graph_mlz,
                # #gravity_vector_neg_y,
                # #powder_coordinate_transformation_graph,
                # #powder_coordinate_transformation_graph_mlz, # includes evaluation of dperp
                # #extract_raw_data_mlz,
                # #add_scattering_coordinates_from_positions,
                # add_scattering_coordinates_from_positions_mlz,
                # #add_masks_and_corrections,
                # # add_masks_mlz, <- might be better to add masks in this step in future, because masking binned data can bring additional distortion
                # flatten_to_spectra,
                # focus_to_dspacing_dperp,
                # #focus_data_dspacing_and_two_theta,
                # # extract_accumulated_proton_charge,
                # # normalize_by_proton_charge_mlz, <- this and above seem to be absent in PowderReduceP2D
                # # integrate_two_theta,
                # subtract_sample_empty_can_mlz,
                # subtract_vanadium_empty_can_mlz,
                # handle_negative_intensity_sample,
                # handle_negative_intensity_vanadium,
                # normalize_by_vanadium_mlz,
                # append_two_theta_lambda,
                # add_masks_p2d_data,
                # extract_masked_intensity,
                # prepare_for_p2d_export,
                # export_to_p2d
            ]
        ),
        params=default_parameters()
    )


