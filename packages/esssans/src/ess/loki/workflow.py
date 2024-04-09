# This file is used by beamlime to create a workflow for the Loki instrument.
# The function `live_workflow` is registered as the entry point for the workflow.
import scipp as sc
from beamlime import WorkflowResult
from scippneutron.io.nexus.load_nexus import JSONGroup


def live_workflow(group: JSONGroup) -> WorkflowResult:
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
