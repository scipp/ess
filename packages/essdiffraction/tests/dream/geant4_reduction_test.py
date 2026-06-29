# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io
from pathlib import Path

import ess.dream.data  # noqa: F401
import pytest
import sciline
import scipp as sc
import scipp.testing
from ess import dream, powder
from ess.dream.parameters import parameters as dream_parameters
from ess.dream.workflows import (
    DreamGeant4MonitorHistogramWorkflow,
    DreamGeant4MonitorIntegratedWorkflow,
    DreamGeant4ProtonChargeWorkflow,
)
from ess.powder.types import (
    CalibrationFilename,
    CIFAuthors,
    CorrectedDetector,
    DspacingBins,
    EmptyCanRun,
    EmptyCanSubtractedIofDspacing,
    Filename,
    FocussedDataDspacing,
    IntensityDspacing,
    IntensityDspacingTwoTheta,
    IntensityTof,
    KeepEvents,
    LookupTableFilename,
    MaskedDetectorIDs,
    MonitorFilename,
    NeXusDetectorName,
    ReducedTofCIF,
    SampleRun,
    TofMask,
    TwoThetaBins,
    TwoThetaMask,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthMask,
)
from scippneutron import metadata
from scippnexus import NXsource

from ess.reduce import unwrap
from ess.reduce import workflow as reduce_workflow
from ess.reduce.nexus.types import AnyRun, Position
from ess.reduce.parameter import keep_default

PARAMETRIZATION = ("detector_name", ["mantle", "endcap_backward", "endcap_forward"])


params = {
    Filename[SampleRun]: dream.data.simulated_diamond_sample(small=True),
    Filename[VanadiumRun]: dream.data.simulated_vanadium_sample(small=True),
    Filename[EmptyCanRun]: dream.data.simulated_empty_can(small=True),
    MonitorFilename[SampleRun]: dream.data.simulated_monitor_diamond_sample(),
    MonitorFilename[VanadiumRun]: dream.data.simulated_monitor_vanadium_sample(),
    MonitorFilename[EmptyCanRun]: dream.data.simulated_monitor_empty_can(),
    dream.InstrumentConfiguration: dream.beamline.InstrumentConfiguration.high_flux_BC215,  # noqa: E501
    Position[NXsource, AnyRun]: sc.vector(value=[0, 0, -76.55], unit="m"),
    CalibrationFilename: None,
    UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    DspacingBins: sc.linspace('dspacing', 0.0, 2.3434, 201, unit='angstrom'),
    TofMask: None,
    TwoThetaMask: None,
    WavelengthMask: None,
    CIFAuthors: CIFAuthors(
        [
            metadata.Person(
                name="Jane Doe",
                email="jane.doe@ess.eu",
                orcid_id="0000-0000-0000-0001",
                corresponding=True,
            ),
        ]
    ),
}


def _make_workflow(
    detector_name,
    run_norm=powder.RunNormalization.proton_charge,
    wavelength_from="file",
):
    wf = dream.DreamGeant4Workflow(run_norm=run_norm, wavelength_from=wavelength_from)
    wf[NeXusDetectorName] = detector_name

    for key, value in params.items():
        wf[key] = value
    return wf


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_pipeline_can_compute_dspacing_result(detector_name):
    workflow = powder.with_pixel_mask_filenames(
        _make_workflow(detector_name=detector_name), []
    )
    result = workflow.compute(EmptyCanSubtractedIofDspacing)
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_pipeline_can_compute_dspacing_result_without_empty_can(detector_name):
    workflow = _make_workflow(detector_name=detector_name)
    workflow[Filename[EmptyCanRun]] = None
    workflow[MonitorFilename[EmptyCanRun]] = None
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IntensityDspacing[SampleRun])
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


@pytest.mark.parametrize(*PARAMETRIZATION)
@pytest.mark.parametrize("lut_mode", ["file", "simulation", "analytical"])
def test_pipeline_can_compute_dspacing_result_different_lookup_tables(
    detector_name,
    lut_mode,
):
    workflow = _make_workflow(detector_name=detector_name, wavelength_from=lut_mode)

    if lut_mode == "file":
        workflow[LookupTableFilename] = dream.data.lookup_table_high_flux()
    else:
        workflow[unwrap.DiskChoppers] = dream.beamline.choppers(
            dream.beamline.InstrumentConfiguration.high_flux_BC215
        )
        workflow[unwrap.DistanceResolution] = sc.scalar(0.1, unit="m")
        workflow[unwrap.TimeResolution] = sc.scalar(250.0, unit='us')
        if lut_mode == "simulation":
            workflow[unwrap.NumberOfSimulatedNeutrons] = 200_000
            workflow[unwrap.SimulationSeed] = 555

    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(EmptyCanSubtractedIofDspacing)
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


@pytest.mark.parametrize(*PARAMETRIZATION)
@pytest.mark.parametrize("keep_events", [True, False])
@pytest.mark.parametrize("norm", ["monitor_histogram", "monitor_integrated"])
def test_pipeline_can_compute_dspacing_result_with_different_norm(
    detector_name, keep_events: bool, norm: str
):
    workflow = _make_workflow(detector_name=detector_name, run_norm=norm)
    workflow[KeepEvents[SampleRun]] = KeepEvents[SampleRun](keep_events)
    workflow[KeepEvents[VanadiumRun]] = KeepEvents[VanadiumRun](keep_events)
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IntensityDspacing[SampleRun])
    assert result.sizes == {'dspacing': len(params[DspacingBins]) - 1}
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_pipeline_normalizes_and_subtracts_empty_can_as_expected(detector_name) -> None:
    sample = sc.data.binned_x(13, 3)
    vanadium = sc.data.binned_x(16, 3)
    empty_can = sc.data.binned_x(9, 3)

    workflow = _make_workflow(detector_name=detector_name)
    workflow[FocussedDataDspacing[SampleRun]] = sample
    workflow[FocussedDataDspacing[VanadiumRun]] = vanadium
    workflow[FocussedDataDspacing[EmptyCanRun]] = empty_can
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(EmptyCanSubtractedIofDspacing)

    subtracted = sample.bins.concatenate(-empty_can)
    expected = powder.correction.normalize_by_vanadium_dspacing(
        FocussedDataDspacing[SampleRun](subtracted),
        FocussedDataDspacing[VanadiumRun](vanadium),
        UncertaintyBroadcastMode.drop,
    )
    sc.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_workflow_is_deterministic(detector_name):
    workflow = powder.with_pixel_mask_filenames(
        _make_workflow(detector_name=detector_name), []
    )
    # This is Sciline's default scheduler, but we want to be explicit here
    scheduler = sciline.scheduler.DaskScheduler()
    graph = workflow.get(IntensityTof, scheduler=scheduler)
    reference = graph.compute().data
    result = graph.compute().data
    assert sc.identical(sc.values(result), sc.values(reference))


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_pipeline_can_compute_intermediate_results(detector_name):
    workflow = powder.with_pixel_mask_filenames(
        _make_workflow(detector_name=detector_name), []
    )
    results = workflow.compute((CorrectedDetector[SampleRun], NeXusDetectorName))
    result = results[CorrectedDetector[SampleRun]]

    detector_name = results[NeXusDetectorName]
    expected_dims = {'segment', 'wire', 'counter', 'strip', 'module'}
    if detector_name in ('endcap_backward', 'endcap_forward'):
        expected_dims.add('sumo')

    assert expected_dims.issubset(set(result.dims))


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_pipeline_group_by_two_theta(detector_name):
    two_theta_bins = sc.linspace(
        dim='two_theta', unit='rad', start=0.8, stop=2.4, num=17
    )
    workflow = _make_workflow(detector_name=detector_name)
    workflow[TwoThetaBins] = two_theta_bins
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IntensityDspacingTwoTheta[SampleRun])
    assert result.sizes['two_theta'] == 16
    assert result.sizes['dspacing'] == len(params[DspacingBins]) - 1
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])
    assert sc.allclose(result.coords['two_theta'], two_theta_bins)


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_pipeline_wavelength_masking(detector_name):
    workflow = _make_workflow(detector_name=detector_name)
    wmin = sc.scalar(0.18, unit="angstrom")
    wmax = sc.scalar(0.21, unit="angstrom")
    workflow[WavelengthMask] = lambda x: (x > wmin) & (x < wmax)
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    masked_sample = workflow.compute(CorrectedDetector[SampleRun])
    assert 'wavelength' in masked_sample.bins.masks
    sum_in_masked_region = (
        masked_sample.bin(wavelength=sc.concat([wmin, wmax], dim='wavelength'))
        .sum()
        .data
    )
    assert sc.allclose(
        sum_in_masked_region,
        sc.scalar(0.0, unit=sum_in_masked_region.unit),
    )


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_pipeline_two_theta_masking(detector_name):
    workflow = _make_workflow(detector_name=detector_name)
    tmin = sc.scalar(1.0, unit="rad")
    tmax = sc.scalar(1.2, unit="rad")
    workflow[TwoThetaMask] = lambda x: (x > tmin) & (x < tmax)
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    masked_sample = workflow.compute(CorrectedDetector[SampleRun])
    assert 'two_theta' in masked_sample.masks
    sum_in_masked_region = (
        masked_sample.bin(two_theta=sc.concat([tmin, tmax], dim='two_theta')).sum().data
    )
    assert sc.allclose(
        sum_in_masked_region,
        sc.scalar(0.0, unit=sum_in_masked_region.unit),
    )


@pytest.mark.parametrize(*PARAMETRIZATION)
def test_pipeline_can_save_data(detector_name):
    workflow = powder.with_pixel_mask_filenames(
        _make_workflow(detector_name=detector_name), []
    )
    result = workflow.compute(ReducedTofCIF)

    buffer = io.StringIO()
    result.save(buffer)
    buffer.seek(0)
    content = buffer.read()

    assert content.startswith(r'#\#CIF_1.1')
    _assert_contains_source_info(content)
    _assert_contains_author_info(content)
    _assert_contains_beamline_info(content)
    _assert_contains_tof_data(content)


def test_pipeline_save_data_to_disk(output_folder: Path):
    """
    This test saves a reduced CIF file to disk using the DREAM workflow.
    The reduced results are uploaded as an artifact in GH actions, and is subsequently
    used by analysis software as part of the integration tests. Therefore, we need
    to have enough signal: we thus use the large files instead of small, and use the
    mantle detector bank.
    """
    wf = _make_workflow(
        detector_name="mantle", run_norm=powder.RunNormalization.proton_charge
    )

    wf[Filename[SampleRun]] = dream.data.simulated_diamond_sample(small=False)
    wf[Filename[VanadiumRun]] = dream.data.simulated_vanadium_sample(small=False)
    wf[Filename[EmptyCanRun]] = dream.data.simulated_empty_can(small=False)
    wf[DspacingBins] = sc.linspace('dspacing', 0.3, 2.3434, 2001, unit='angstrom')
    wf = powder.with_pixel_mask_filenames(wf, [])

    result = wf.compute(ReducedTofCIF)
    result.comment = """This file was generated with the DREAM data reduction user guide
    in the documentation of ESSdiffraction.
    See https://scipp.github.io/ess/diffraction/
    """
    result.save(output_folder / "dream_reduced.cif")


def _assert_contains_source_info(cif_content: str) -> None:
    assert 'diffrn_source.beamline DREAM' in cif_content


def _assert_contains_author_info(cif_content: str) -> None:
    assert "audit_contact_author.name 'Jane Doe'" in cif_content
    assert 'audit_contact_author.email jane.doe@ess.eu' in cif_content
    assert (
        'audit_contact_author.id_orcid https://orcid.org/0000-0000-0000-0001'
        in cif_content
    )


def _assert_contains_beamline_info(cif_content: str) -> None:
    assert 'diffrn_source.beamline DREAM' in cif_content
    assert 'diffrn_source.facility ESS' in cif_content


def _assert_contains_tof_data(cif_content: str) -> None:
    assert 'pd_meas.time_of_flight' in cif_content
    assert 'pd_proc.intensity_norm' in cif_content
    assert 'pd_proc.intensity_norm_su' in cif_content


def test_dream_workflow_registers_subclasses():
    # Because it was imported
    for wf in (
        DreamGeant4MonitorHistogramWorkflow,
        DreamGeant4MonitorIntegratedWorkflow,
        DreamGeant4ProtonChargeWorkflow,
    ):
        assert wf in reduce_workflow.workflow_registry
    count = len(reduce_workflow.workflow_registry)

    @reduce_workflow.register_workflow()
    class MyWorkflow: ...

    assert MyWorkflow in reduce_workflow.workflow_registry
    assert len(reduce_workflow.workflow_registry) == count + 1


def test_dream_workflow_registers_parameter_registry():
    spec = reduce_workflow.workflow_registry.get(DreamGeant4ProtonChargeWorkflow)
    assert spec.parameters is dream_parameters


def test_dream_workflow_applies_parameter_values():
    wf = DreamGeant4ProtonChargeWorkflow()
    spec = reduce_workflow.workflow_registry.get(DreamGeant4ProtonChargeWorkflow)
    specs = reduce_workflow.get_parameters(
        wf, (IntensityDspacingTwoTheta[SampleRun],), spec.parameters
    )
    values = {
        key: spec.default
        for key, spec in specs.items()
        if spec.default is not keep_default
    }
    values[Filename[SampleRun]] = "sample.csv"

    wf = reduce_workflow.assign_parameter_values(wf, values, specs)

    assert wf.compute(Filename[SampleRun]) == "sample.csv"
    assert wf.compute(CalibrationFilename) is None
    assert wf.compute(MaskedDetectorIDs) == {}
    assert wf.compute(TwoThetaBins).unit == sc.Unit('rad')
