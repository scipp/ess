# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import itertools

import sciline
import scipp as sc
import scippnexus as snx
from scippneutron.metadata import Software

from ess.powder import providers as powder_providers
from ess.powder import with_pixel_mask_filenames
from ess.powder.correction import RunNormalization, insert_run_normalization
from ess.powder.types import (
    AccumulatedProtonCharge,
    BunkerMonitor,
    CaveMonitor,
    CaveMonitorPosition,  # Should this be a DREAM-only parameter?
    EmptyCanRun,
    KeepEvents,
    PixelMaskFilename,
    Position,
    ReducerSoftware,
    SampleRun,
    TimeOfFlightLookupTableFilename,
    TofMask,
    TwoThetaMask,
    VanadiumRun,
    WavelengthMask,
)
from ess.reduce.nexus.types import DetectorBankSizes, NeXusName
from ess.reduce.parameter import parameter_mappers
from ess.reduce.time_of_flight import GenericTofWorkflow
from ess.reduce.workflow import register_workflow

from .beamline import InstrumentConfiguration
from .io.cif import (
    CIFAuthors,
    prepare_reduced_empty_can_subtracted_tof_cif,
    prepare_reduced_tof_cif,
)
from .io.geant4 import providers as geant4_providers
from .parameters import typical_outputs

DETECTOR_BANK_SIZES = {
    "endcap_backward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 11,
        "segment": 28,
        "counter": 2,
    },
    "endcap_forward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 5,
        "segment": 28,
        "counter": 2,
    },
    "mantle_detector": {
        "wire": 32,
        "module": 5,
        "segment": 6,
        "strip": 256,
        "counter": 2,
    },
    "high_resolution_detector": {"strip": 32, "other": -1},
    "sans_detector": {"strip": 32, "other": -1},
}


def _get_lookup_table_filename_from_configuration(
    configuration: InstrumentConfiguration,
) -> TimeOfFlightLookupTableFilename:
    from .data import tof_lookup_table_high_flux

    match configuration:
        case InstrumentConfiguration.high_flux_BC215:
            out = tof_lookup_table_high_flux(bc=215)
        case InstrumentConfiguration.high_flux_BC240:
            out = tof_lookup_table_high_flux(bc=240)
        case InstrumentConfiguration.high_resolution:
            raise NotImplementedError("High resolution configuration not yet supported")

    return TimeOfFlightLookupTableFilename(out)


def _collect_reducer_software() -> ReducerSoftware:
    return ReducerSoftware(
        [
            Software.from_package_metadata('essdiffraction'),
            Software.from_package_metadata('scippneutron'),
            Software.from_package_metadata('scipp'),
        ]
    )


def DreamWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Dream generic workflow with default parameters.
    The workflow is based on the GenericTofWorkflow.
    It can load data from a NeXus file recorded on the DREAM instrument, and can
    compute time-of-flight for the neutron events.

    It can be used as is, or as a base for more specific workflows, such as
    ``DreamPowderWorkflow``.

    Parameters
    ----------
    kwargs:
        Additional keyword arguments are forwarded to the base
        :func:`GenericTofWorkflow`.
    """
    wf = GenericTofWorkflow(
        run_types=[SampleRun, VanadiumRun, EmptyCanRun],
        monitor_types=[BunkerMonitor, CaveMonitor],
        **kwargs,
    )
    wf[DetectorBankSizes] = DETECTOR_BANK_SIZES
    wf[NeXusName[BunkerMonitor]] = "monitor_bunker"
    wf[NeXusName[CaveMonitor]] = "monitor_cave"
    wf.insert(_get_lookup_table_filename_from_configuration)
    wf[ReducerSoftware] = _collect_reducer_software()
    return wf


_cif_providers = (
    prepare_reduced_tof_cif,
    prepare_reduced_empty_can_subtracted_tof_cif,
)

parameter_mappers[PixelMaskFilename] = with_pixel_mask_filenames


def default_parameters() -> dict:
    return {
        KeepEvents[SampleRun]: KeepEvents[SampleRun](True),
        KeepEvents[VanadiumRun]: KeepEvents[VanadiumRun](False),
        KeepEvents[EmptyCanRun]: KeepEvents[EmptyCanRun](True),
        TofMask: None,
        WavelengthMask: None,
        TwoThetaMask: None,
        CIFAuthors: CIFAuthors([]),
    }


def DreamPowderWorkflow(*, run_norm: RunNormalization, **kwargs) -> sciline.Pipeline:
    """
    Dream powder workflow with default parameters.

    Parameters
    ----------
    run_norm:
        Select how to normalize each run (sample, vanadium, etc.).
    kwargs:
        Additional keyword arguments are forwarded to the base :func:`DreamWorkflow`.

    Returns
    -------
    :
        A workflow object for DREAM.
    """
    wf = DreamWorkflow(**kwargs)
    for provider in itertools.chain(powder_providers, _cif_providers):
        wf.insert(provider)
    insert_run_normalization(wf, run_norm)
    for key, value in default_parameters().items():
        wf[key] = value
    wf.typical_outputs = typical_outputs
    return wf


def DreamGeant4Workflow(*, run_norm: RunNormalization, **kwargs) -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation.

    Parameters
    ----------
    run_norm:
        Select how to normalize each run (sample, vanadium, etc.).
    kwargs:
        Additional keyword arguments are forwarded to the base :func:`DreamWorkflow`.

    Returns
    -------
    :
        A workflow object for DREAM.
    """
    wf = DreamWorkflow(**kwargs)
    for provider in itertools.chain(geant4_providers, powder_providers, _cif_providers):
        wf.insert(provider)
    insert_run_normalization(wf, run_norm)
    for key, value in default_parameters().items():
        wf[key] = value

    # Quantities not available in the simulated data
    sample_position = sc.vector([0.0, 0.0, 0.0], unit="mm")
    source_position = sc.vector([-3.478, 0.0, -76550], unit="mm")
    charge = sc.scalar(1.0, unit="µAh")

    additional_parameters = {
        Position[snx.NXsample, SampleRun]: sample_position,
        Position[snx.NXsample, VanadiumRun]: sample_position,
        Position[snx.NXsample, EmptyCanRun]: sample_position,
        Position[snx.NXsource, SampleRun]: source_position,
        Position[snx.NXsource, VanadiumRun]: source_position,
        Position[snx.NXsource, EmptyCanRun]: source_position,
        AccumulatedProtonCharge[SampleRun]: charge,
        AccumulatedProtonCharge[VanadiumRun]: charge,
        AccumulatedProtonCharge[EmptyCanRun]: charge,
        CaveMonitorPosition: sc.vector([0.0, 0.0, -4220.0], unit='mm'),
    }
    for key, value in additional_parameters.items():
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
    'DreamPowderWorkflow',
    'DreamWorkflow',
    'default_parameters',
]
