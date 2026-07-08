# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import sciline
from ess.packages.essdiffraction.src.ess.powder import providers as powder_providers

from ess.packages.essdiffraction.src.ess.powder.types import (
    KeepEvents,
    KeepEventsMLZ,
    NeXusDetectorName,
    SampleRun,
    TwoThetaBins,
    VanadiumRun,
    EmptyVanadiumCanRun,
    CalibrationRun,
    FilepathMLZ,
    FilenameMLZ,
    ShiftNegativeIntensity,
    NegativeIntensityResetValue,
    P2DOutputFilename,
)

from ess.packages.essreduce.src.ess.reduce.nexus.workflow import gravity_vector_neg_y

from . import beamline

from ess.packages.essdiffraction.src.ess.snspowder.powgen.data import (
    load_powgen_mlz,
    load_powgen_calibration_mlz,
    sample_position,
    sample_position_mlz,
    source_position,
    source_position_mlz,
    extract_raw_data_mlz,
    extract_accumulated_proton_charge,
    extract_masked_intensity,
    prepare_for_p2d_export,
    export_to_p2d
)

from ess.packages.essdiffraction.src.ess.powder.conversion import (
    powder_coordinate_transformation_graph,
    powder_coordinate_transformation_graph_mlz,
    add_scattering_coordinates_from_positions,
    p2d_coordinate_transformation_graph_mlz,
    add_scattering_coordinates_from_positions_mlz,
)

from ess.packages.essdiffraction.src.ess.powder.workflow import (
    add_masks_and_corrections,
    add_masks_mlz,
    append_two_theta_lambda,
    add_masks_p2d_data
)

from ess.packages.essdiffraction.src.ess.powder.grouping import (
    focus_data_dspacing_and_two_theta,
    bin_dspacing_dperp_mlz,
    focus_to_dspacing_dperp,
    integrate_two_theta,
    integrate_dperp
)

from ess.packages.essdiffraction.src.ess.powder.correction import (
    normalize_by_proton_charge,
    normalize_by_proton_charge_mlz,
    normalize_by_vanadium_dspacing,
    handle_negative_intensity_sample,
    handle_negative_intensity_vanadium,
    subtract_sample_empty_can_mlz,
    subtract_vanadium_empty_can_mlz,
    normalize_by_vanadium_mlz
)

from ess.packages.essdiffraction.src.ess.powder.filtering import (
    flatten_to_spectra
)

def default_parameters() -> dict:
    """Return the default parameters for the POWGEN workflow."""
    return {
        KeepEvents[SampleRun]: KeepEvents[SampleRun](True),
        KeepEvents[VanadiumRun]: KeepEvents[VanadiumRun](False),
        KeepEvents[EmptyCanRun]: KeepEvents[EmptyCanRun](True),
        NeXusDetectorName: "powgen_detector",
        TwoThetaBins: None,
    }


def PowgenWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Powgen SNS instrument.
    """
    # The package does not depend on pooch which is needed for the tutorial
    # data. Delay import until workflow is actually used.
    from . import data

    return sciline.Pipeline(
        providers=(
            *powder_providers,
            *beamline.providers,
            *data.providers,
            gravity_vector_neg_y,
        ),
        params=default_parameters(),
    )


def PowderReduceP2D() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Powgen SNS instrument.
    """
    # The package does not depend on pooch which is needed for the tutorial
    # data. Delay import until workflow is actually used.
    from . import data

    return sciline.Pipeline(
        providers=(
            [
                load_powgen_mlz,
                load_powgen_calibration_mlz,
                # sample_position,
                #sample_position_mlz,
                # source_position,
                #source_position_mlz,
                # coordinate_transformation_graph_mlz,
                p2d_coordinate_transformation_graph_mlz,
                #gravity_vector_neg_y,
                #powder_coordinate_transformation_graph,
                #powder_coordinate_transformation_graph_mlz, # includes evaluation of dperp
                #extract_raw_data_mlz,
                #add_scattering_coordinates_from_positions,
                add_scattering_coordinates_from_positions_mlz,
                #add_masks_and_corrections,
                # add_masks_mlz, <- might be better to add masks in this step in future, because masking binned data can bring additional distortion
                flatten_to_spectra,
                focus_to_dspacing_dperp,
                #focus_data_dspacing_and_two_theta,
                # extract_accumulated_proton_charge,
                # normalize_by_proton_charge_mlz, <- this and above seem to be absent in PowderReduceP2D
                # integrate_two_theta,
                subtract_sample_empty_can_mlz,
                subtract_vanadium_empty_can_mlz,
                handle_negative_intensity_sample,
                handle_negative_intensity_vanadium,
                normalize_by_vanadium_mlz,
                append_two_theta_lambda,
                add_masks_p2d_data,
                extract_masked_intensity,
                prepare_for_p2d_export,
                export_to_p2d
            ]
        ),
        params={
            FilepathMLZ[SampleRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/yannick/converted",
            FilepathMLZ[VanadiumRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/yannick/converted",
            FilepathMLZ[EmptyCanRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/yannick/converted",
            FilepathMLZ[CalibrationRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/yannick/converted",
            FilepathMLZ[EmptyVanadiumCanRun]: "/home/koshchii/Instruments/POWTEX/Scipp/data/yannick/converted",
            KeepEventsMLZ: False,
            ShiftNegativeIntensity[SampleRun]: True,
            ShiftNegativeIntensity[VanadiumRun]: False,
            NegativeIntensityResetValue[SampleRun]: 0.0,
            NegativeIntensityResetValue[VanadiumRun]: 0.0,
            P2DOutputFilename: "/home/koshchii/Instruments/POWTEX/Scipp/data/output/reduced.p2d"
            # CalibrationData: None,
        }
    )

__all__ = ['default_parameters', 'PowderReduceP2D']
