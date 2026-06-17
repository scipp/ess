# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import sciline
from ess.packages.essdiffraction.src.ess.powder import providers as powder_providers

from ess.packages.essdiffraction.src.ess.powder.types import (
    EmptyCanRun,
    KeepEvents,
    NeXusDetectorName,
    SampleRun,
    TwoThetaBins,
    VanadiumRun,
    CalibrationRun,
    FilepathMLZ,
    FilenameMLZ,
)

from ess.packages.essreduce.src.ess.reduce.nexus.workflow import gravity_vector_neg_y

from . import beamline

from ess.packages.essdiffraction.src.ess.snspowder.powgen.data import (
    load_powgen_mlz,
    load_powgen_calibration_mlz,
    sample_position,
    source_position,
    extract_raw_data_mlz,
    extract_accumulated_proton_charge,
)

from ess.packages.essdiffraction.src.ess.powder.conversion import (
    powder_coordinate_transformation_graph,
    powder_coordinate_transformation_graph_mlz,
    add_scattering_coordinates_from_positions,
    add_scattering_coordinates_from_positions_mlz,
)

from ess.packages.essdiffraction.src.ess.powder.workflow import (
    add_masks_and_corrections,
    add_masks_and_corrections_mlz
)

from ess.packages.essdiffraction.src.ess.powder.grouping import (
    focus_data_dspacing_and_two_theta,
    focus_data_dspacing_and_dperp_mlz,
    integrate_two_theta,
)

from ess.packages.essdiffraction.src.ess.powder.correction import (
    normalize_by_proton_charge,
    normalize_by_vanadium_dspacing
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


def PowgenMantidWorkflow() -> sciline.Pipeline:
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
                sample_position,
                source_position,
                gravity_vector_neg_y,
                #powder_coordinate_transformation_graph,
                powder_coordinate_transformation_graph_mlz, # includes evaluation of dperp
                extract_raw_data_mlz,
                #add_scattering_coordinates_from_positions,
                add_scattering_coordinates_from_positions_mlz,
                #add_masks_and_corrections,
                add_masks_and_corrections_mlz,
                #focus_data_dspacing_and_two_theta,
                focus_data_dspacing_and_dperp_mlz,
                extract_accumulated_proton_charge,
                normalize_by_proton_charge,
                integrate_two_theta,
                normalize_by_vanadium_dspacing,
            ]
        ),
        params={
            FilepathMLZ[SampleRun]: '/home/koshchii/Instruments/POWTEX/DreamScipp/data/yannick/converted',
            FilepathMLZ[VanadiumRun]: '/home/koshchii/Instruments/POWTEX/DreamScipp/data/yannick/converted',
            FilepathMLZ[EmptyCanRun]: '/home/koshchii/Instruments/POWTEX/DreamScipp/data/yannick/converted',
            FilepathMLZ[CalibrationRun]: '/home/koshchii/Instruments/POWTEX/DreamScipp/data/yannick/converted',
            # need to figute out how the below keep events works
            KeepEvents[SampleRun]: KeepEvents[SampleRun](True),
            KeepEvents[VanadiumRun]: KeepEvents[VanadiumRun](False),
            KeepEvents[EmptyCanRun]: KeepEvents[EmptyCanRun](True),
            # CalibrationData: None,
        }
    )

__all__ = ['default_parameters', 'PowgenMantidWorkflow']
