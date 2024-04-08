try:
    import numpy as np
    import scipp as sc
    from beamlime import WorkflowResult
    from scippneutron.io.nexus.load_nexus import JSONGroup
except ImportError as e:
    raise ImportError(
        "Please install scipp, scippneutron, and beamlime to run workflows with loki."
    ) from e


def live_workflow(group: JSONGroup) -> WorkflowResult:
    """Example live workflow function for Loki.

    This function is used to process data with beamlime workflows.
    """
    # We do not consume the incoming data right now.
    _ = group
    return {
        'IofQ': sc.DataArray(
            data=sc.array(dims=['Q'], values=np.zeros(10)),
            coords={'Q': sc.array(dims=['Q'], values=np.zeros(10), unit='1/angstrom')},
        )
    }
