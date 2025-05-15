# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import itertools

import sciline
import scipp as sc
import scippnexus as snx
from scippneutron.metadata import Software

from ess.powder import providers as powder_providers
from ess.powder import with_pixel_mask_filenames
from ess.powder.conversion import convert_monitor_to_wavelength
from ess.powder.correction import (
    RunNormalization,
    insert_run_normalization,
)
from ess.powder.types import (
    AccumulatedProtonCharge,
    CaveMonitorPosition,  # Should this be a DREAM-only parameter?
    MonitorType,
    PixelMaskFilename,
    Position,
    ReducerSoftwares,
    ResampledMonitorTofData,
    RunType,
    SampleRun,
    TimeOfFlightLookupTableFilename,
    TofMask,
    TwoThetaMask,
    VanadiumRun,
    WavelengthMask,
    WavelengthMonitor,
)
from ess.reduce import time_of_flight
from ess.reduce.parameter import parameter_mappers
from ess.reduce.time_of_flight import resample_monitor_time_of_flight_data
from ess.reduce.workflow import register_workflow

from .beamline import InstrumentConfiguration
from .io.cif import CIFAuthors, prepare_reduced_tof_cif
from .io.geant4 import LoadGeant4Workflow
from .parameters import typical_outputs


def _get_lookup_table_filename_from_configuration(
    configuration: InstrumentConfiguration,
) -> TimeOfFlightLookupTableFilename:
    from .data import tof_lookup_table_high_flux

    match configuration:
        case InstrumentConfiguration.high_flux:
            out = tof_lookup_table_high_flux()
        case InstrumentConfiguration.high_resolution:
            raise NotImplementedError("High resolution configuration not yet supported")

    return TimeOfFlightLookupTableFilename(out)


_dream_providers = (
    prepare_reduced_tof_cif,
    _get_lookup_table_filename_from_configuration,
)

parameter_mappers[PixelMaskFilename] = with_pixel_mask_filenames


def default_parameters() -> dict:
    # Quantities not available in the simulated data
    sample_position = sc.vector([0.0, 0.0, 0.0], unit="mm")
    source_position = sc.vector([-3.478, 0.0, -76550], unit="mm")
    charge = sc.scalar(1.0, unit="µAh")
    return {
        Position[snx.NXsample, SampleRun]: sample_position,
        Position[snx.NXsample, VanadiumRun]: sample_position,
        Position[snx.NXsource, SampleRun]: source_position,
        Position[snx.NXsource, VanadiumRun]: source_position,
        AccumulatedProtonCharge[SampleRun]: charge,
        AccumulatedProtonCharge[VanadiumRun]: charge,
        TofMask: None,
        WavelengthMask: None,
        TwoThetaMask: None,
        CaveMonitorPosition: sc.vector([0.0, 0.0, -4220.0], unit='mm'),
        CIFAuthors: CIFAuthors([]),
        ReducerSoftwares: _collect_reducer_software(),
    }


def _collect_reducer_software() -> ReducerSoftwares:
    return ReducerSoftwares(
        [
            Software.from_package_metadata('essdiffraction'),
            Software.from_package_metadata('scippneutron'),
            Software.from_package_metadata('scipp'),
        ]
    )


def convert_dream_monitor_to_wavelength(
    monitor: ResampledMonitorTofData[RunType, MonitorType],
) -> WavelengthMonitor[RunType, MonitorType]:
    """
    We know that DREAM monitors are recording in histogram mode, so we need to use the
    resampled monitor data to avoid having nans in the time-of-flight coordinates.

    This provider should be inserted in the Dream workflow below.
    """
    return convert_monitor_to_wavelength(monitor)


def DreamGeant4Workflow(*, run_norm: RunNormalization) -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation.
    """
    wf = LoadGeant4Workflow()
    for provider in itertools.chain(powder_providers, _dream_providers):
        wf.insert(provider)
    wf.insert(convert_dream_monitor_to_wavelength)
    wf.insert(resample_monitor_time_of_flight_data)
    insert_run_normalization(wf, run_norm)
    for key, value in itertools.chain(
        default_parameters().items(), time_of_flight.default_parameters().items()
    ):
        wf[key] = value
    wf.typical_outputs = typical_outputs
    return wf


@register_workflow
def DreamGeant4MonitorHistogramWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation, using a
    histogrammed monitor for the normalization.
    """
    return DreamGeant4Workflow(run_norm=RunNormalization.monitor_histogram)


@register_workflow
def DreamGeant4MonitorIntegratedWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation, using
    integrated counts of the monitor for the normalization.
    """
    return DreamGeant4Workflow(run_norm=RunNormalization.monitor_integrated)


@register_workflow
def DreamGeant4ProtonChargeWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation, using
    proton charge for the normalization.
    """
    return DreamGeant4Workflow(run_norm=RunNormalization.proton_charge)


__all__ = [
    'DreamGeant4MonitorHistogramWorkflow',
    'DreamGeant4MonitorIntegratedWorkflow',
    'DreamGeant4ProtonChargeWorkflow',
    'DreamGeant4Workflow',
    'default_parameters',
]
