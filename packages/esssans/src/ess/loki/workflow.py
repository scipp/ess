# This file is used by beamlime to create a workflow for the Loki instrument.
# The callable `live_workflow` is registered as the entry point for the workflow.
import sciline
import scipp as sc
import scippnexus as snx
from ess.loki.general import (
    get_monitor_data,
    get_source_position,
    monitor_to_tof,
    patch_monitor_data,
)
from ess.loki.io import load_nexus_monitor, load_nexus_source
from ess.reduce.nexus.json_nexus import JSONGroup
from ess.sans.conversions import monitor_to_wavelength, sans_monitor
from ess.sans.types import (
    Filename,
    Incident,
    MonitorType,
    NeXusMonitorName,
    RunType,
    SampleRun,
    Transmission,
    WavelengthBins,
    WavelengthMonitor,
)


class MonitorHistogram(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...


def _hist_monitor_wavelength(
    wavelength_bin: WavelengthBins, monitor: WavelengthMonitor[RunType, MonitorType]
) -> MonitorHistogram[RunType, MonitorType]:
    return monitor.hist(wavelength=wavelength_bin)


def loki_iofq_workflow(group: JSONGroup) -> dict[str, sc.DataArray]:
    """Example live workflow function for Loki.

    This function is used to process data with beamlime workflows.
    """
    # We do not consume the incoming data right now.
    _ = group
    return {
        'IofQ': sc.DataArray(
            data=sc.zeros(dims=['Q'], shape=[100]),
            coords={
                'Q': sc.linspace(
                    dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom'
                )
            },
        )
    }


class LoKiMonitorWorkflow:
    """LoKi Monitor wavelength histogram workflow for live data reduction."""

    def __init__(self) -> None:
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> sciline.Pipeline:
        """Build a workflow pipeline for live data reduction.

        Returns
        -------
        :
            A pipeline for live data reduction.
            The initial pipeline will be missing the input data.
            It should be set before calling the pipeline.

        """
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
            _hist_monitor_wavelength,
        )

        params = {
            NeXusMonitorName[Incident]: "monitor_1",
            NeXusMonitorName[Transmission]: "monitor_2",
            WavelengthBins: sc.linspace(
                "wavelength", wavelength_min, wavelength_max, n_wavelength_bins + 1
            ),
        }
        workflow = sciline.Pipeline(providers, params=params)
        return workflow

    def __call__(self, group: JSONGroup) -> dict[str, sc.DataArray]:
        """

        Returns
        -------
        :
            Plottable Outputs:

            - MonitorHistogram[SampleRun, Incident]
            - MonitorHistogram[SampleRun, Transmission]

        """
        # ``JsonGroup`` is turned into the ``NexusGroup`` here, not in the ``beamlime``
        # so that the workflow can control the definition of the group.
        self.pipeline[Filename[SampleRun]] = snx.Group(
            group, definitions=snx.base_definitions()
        )
        results = self.pipeline.compute(
            (
                MonitorHistogram[SampleRun, Incident],
                MonitorHistogram[SampleRun, Transmission],
            )
        )

        return {str(tp): result for tp, result in results.items()}


loki_monitor_workflow = LoKiMonitorWorkflow()
