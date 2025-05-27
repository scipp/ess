import sciline

from ess.reduce.nexus.types import DetectorBankSizes
from ess.reduce.nexus.workflow import GenericNeXusWorkflow
from ess.reflectometry.types import ReferenceRun, SampleRun

DETECTOR_BANK_SIZES = {
    "multiblade_detector": {
        "blade": 48,
        "wire": 32,
        "strip": 64,
    },
}


def LoadNeXusWorkflow() -> sciline.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    workflow = GenericNeXusWorkflow(
        run_types=[SampleRun, ReferenceRun], monitor_types=[]
    )
    workflow[DetectorBankSizes] = DETECTOR_BANK_SIZES
    return workflow
