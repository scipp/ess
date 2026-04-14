import sciline

from ess.reduce.nexus.types import DetectorBankSizes
from ess.reduce.unwrap.workflow import GenericUnwrapWorkflow
from ess.reflectometry.types import ReferenceRun, SampleRun

DETECTOR_BANK_SIZES = {
    "multiblade_detector": {
        "strip": 64,
        "blade": 48,
        "wire": 32,
    },
}


def LoadNeXusWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun, ReferenceRun],
        monitor_types=[],
        **kwargs,
    )
    workflow[DetectorBankSizes] = DETECTOR_BANK_SIZES
    return workflow
