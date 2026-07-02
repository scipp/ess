# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters and workflow for Odin.
"""

import sciline
import scipp as sc
from scippneutron.conversion.tof import tof_from_wavelength

from ess.reduce.nexus import load_from_path
from ess.reduce.nexus.types import NeXusFileSpec, NeXusLocationSpec, NeXusName
from ess.reduce.unwrap import GenericUnwrapWorkflow, WavelengthLutMode

from ..imaging import orca
from ..imaging.types import (
    AllRuns,
    BeamMonitor1,
    BeamMonitor2,
    BeamMonitor3,
    BeamMonitor4,
    CorrectedDetector,
    DarkBackgroundRun,
    FluxNormalizedDetector,
    ImageKey,
    LookupTableRelativeErrorThreshold,
    NeXusMonitorName,
    OpenBeamRun,
    PulseStrideOffset,
    RunType,
    SampleRun,
    TofDetector,
)
from .masking import providers as masking_providers


def default_parameters() -> dict:
    """Return the default workflow parameters for Odin."""
    return {
        NeXusMonitorName[BeamMonitor1]: "beam_monitor_1",
        NeXusMonitorName[BeamMonitor2]: "beam_monitor_2",
        NeXusMonitorName[BeamMonitor3]: "beam_monitor_3",
        NeXusMonitorName[BeamMonitor4]: "beam_monitor_4",
        PulseStrideOffset: None,
        LookupTableRelativeErrorThreshold: {
            "event_mode_detectors/timepix3": float("inf"),
            "histogram_mode_detectors/orca": float("inf"),
            "beam_monitor_1": float("inf"),
            "beam_monitor_2": float("inf"),
            "beam_monitor_3": float("inf"),
            "beam_monitor_4": float("inf"),
        },
    }


def compute_detector_tof(da: CorrectedDetector[RunType]) -> TofDetector[RunType]:
    """
    Compute the time-of-flight of neutrons from their wavelength.
    """
    return da.transform_coords(
        "tof", graph={"tof": tof_from_wavelength}, keep_intermediate=False
    )


def OdinWorkflow(
    wavelength_from: WavelengthLutMode = "analytical", **kwargs
) -> sciline.Pipeline:
    """
    Workflow with default parameters for Odin.

    Parameters
    ----------
    wavelength_from:
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html
    kwargs:
        Additional keyword arguments are forwarded to the base
        :func:`GenericUnwrapWorkflow`."""
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun, OpenBeamRun, DarkBackgroundRun],
        monitor_types=[BeamMonitor1, BeamMonitor2, BeamMonitor3, BeamMonitor4],
        wavelength_from=wavelength_from,
        **kwargs,
    )
    workflow.insert(compute_detector_tof)
    for key, param in default_parameters().items():
        workflow[key] = param
    return workflow


def OdinBraggEdgeWorkflow(
    wavelength_from: WavelengthLutMode = "analytical", **kwargs
) -> sciline.Pipeline:
    """
    Workflow with default parameters and masking providers
    for Odin Bragg-edge reduction.

    Parameters
    ----------
    wavelength_from:
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html
    kwargs:
        Additional keyword arguments are forwarded to the base
        :func:`GenericUnwrapWorkflow`."""
    workflow = OdinWorkflow(wavelength_from=wavelength_from, **kwargs)
    for provider in (*masking_providers,):
        workflow.insert(provider)
    return workflow


def load_image_key(file: NeXusFileSpec[AllRuns], path: NeXusName[ImageKey]) -> ImageKey:
    # Note that putting '/value' at the end of the 'path' in the default_parameters
    # yields different results as it can return a Variable instead of a DataArray,
    # depending on the contents of the NeXus file.
    return ImageKey(
        load_from_path(NeXusLocationSpec(filename=file.value, component_name=path))[
            "value"
        ]
    )


KEY_MAPPING = {
    SampleRun: sc.scalar(0, unit=None),
    OpenBeamRun: sc.scalar(1, unit=None),
    DarkBackgroundRun: sc.scalar(2, unit=None),
}


def _extract_part_of_run(
    data: sc.DataArray, image_key: ImageKey, run_type: RunType
) -> sc.DataArray:
    """ """
    key_lookup = sc.lookup(image_key, "time", mode="previous")
    sel = key_lookup[data.coords["time"]] == KEY_MAPPING[run_type]
    return data[sel]


def extract_dark_run(
    data: FluxNormalizedDetector[AllRuns], image_key: ImageKey
) -> FluxNormalizedDetector[DarkBackgroundRun]:
    """ """
    return FluxNormalizedDetector[DarkBackgroundRun](
        _extract_part_of_run(data=data, image_key=image_key, run_type=DarkBackgroundRun)
    )


def extract_openbeam_run(
    data: FluxNormalizedDetector[AllRuns], image_key: ImageKey
) -> FluxNormalizedDetector[OpenBeamRun]:
    """ """
    return FluxNormalizedDetector[OpenBeamRun](
        _extract_part_of_run(data=data, image_key=image_key, run_type=OpenBeamRun)
    )


def extract_sample_run(
    data: FluxNormalizedDetector[AllRuns], image_key: ImageKey
) -> FluxNormalizedDetector[SampleRun]:
    """ """
    return FluxNormalizedDetector[SampleRun](
        _extract_part_of_run(data=data, image_key=image_key, run_type=SampleRun)
    )


def OdinOrcaWorkflow(**kwargs) -> sciline.Pipeline:
    """ """
    wf = orca.OrcaNormalizedImagesWorkflow(**kwargs)
    for provider in (
        load_image_key,
        extract_dark_run,
        extract_openbeam_run,
        extract_sample_run,
    ):
        wf.insert(provider)
    wf[NeXusName[ImageKey]] = (
        '/entry/instrument/histogram_mode_detectors/orca/image_key'
    )
    return wf


__all__ = [
    "OdinBraggEdgeWorkflow",
    "OdinOrcaWorkflow",
    "OdinWorkflow",
]
