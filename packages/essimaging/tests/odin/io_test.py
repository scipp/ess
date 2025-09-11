import io

from tifffile import imread

from ess.imaging.io import tiff_from_nexus
from ess.odin.data import iron_simulation_sample_small

from .data_reduction_test import workflow  # noqa: F401


def test_can_write_tiff_file(workflow):  # noqa: F811
    f = io.BytesIO()
    tiff_from_nexus(
        workflow,
        iron_simulation_sample_small(),
        f,
    )
    f.seek(0)
    imread(f)
