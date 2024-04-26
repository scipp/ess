# This file is used by beamlime to create a workflow for the Loki instrument.
# The function `live_workflow` is registered as the entry point for the workflow.
import sciline
import scipp as sc
from scippneutron.io.nexus.load_nexus import JSONGroup

from ess.loki.general import (
    get_monitor_data,
    get_source_position,
    monitor_to_tof,
    patch_monitor_data,
)
from ess.loki.io import load_nexus_monitor, load_nexus_source
from ess.sans.conversions import monitor_to_wavelength, sans_monitor
from ess.sans.types import (  # FilePath,; Filename,; NeXusMonitorName,
    Incident,
    LoadedNeXusMonitor,
    MonitorType,
    RawSource,
    RunType,
    SampleRun,
    Transmission,
    WavelengthBins,
    WavelengthMonitor,
)


class BinnedMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    ...


def _bin_monitor_wavelength(
    wavelength_bin: WavelengthBins, monitor: WavelengthMonitor[RunType, MonitorType]
) -> BinnedMonitor[RunType, MonitorType]:
    return monitor.bin(wavelength=wavelength_bin)


def build_live_workflow_pipeline(group: JSONGroup) -> sciline.Pipeline:
    """Build a workflow pipeline for live data reduction."""
    # Wavelength binning parameters
    wavelength_min = sc.scalar(1.0, unit="angstrom")
    wavelength_max = sc.scalar(13.0, unit="angstrom")
    n_wavelength_bins = 50

    providers = (
        load_nexus_monitor,
        load_nexus_source,
        get_source_position,
        get_monitor_data,
        patch_monitor_data,
        monitor_to_tof,
        sans_monitor,
        monitor_to_wavelength,
        _bin_monitor_wavelength,
    )

    params = {
        RawSource[SampleRun]: load_nexus_source(group),
        LoadedNeXusMonitor[SampleRun, Incident]: load_nexus_monitor(group, "monitor_1"),
        LoadedNeXusMonitor[SampleRun, Transmission]: load_nexus_monitor(
            group, "monitor_2"
        ),
        # Above lines should be replaced with the following commented code
        # when the `sciline` library is updated to allow setting
        # parameters with instances of different types than the keys.
        # FilePath[Filename[SampleRun]]: group,
        # NeXusMonitorName[Incident]: "monitor_1",
        # NeXusMonitorName[Transmission]: "monitor_2",
        WavelengthBins: sc.linspace(
            "wavelength", wavelength_min, wavelength_max, n_wavelength_bins + 1
        ),
    }
    workflow = sciline.Pipeline(providers, params=params)
    return workflow


def live_workflow(group: JSONGroup) -> dict[str, sc.DataArray]:
    """Example live workflow function for Loki.

    This function is used to process data with beamlime workflows.
    """
    pl = build_live_workflow_pipeline(group)
    results = pl.compute(
        (BinnedMonitor[SampleRun, Incident], BinnedMonitor[SampleRun, Transmission])
    )

    return {str(tp): result for tp, result in results.items()}
