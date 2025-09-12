import io

from tifffile import imread

from ess.imaging.io import tiff_from_nexus
from ess.odin.data import iron_simulation_sample_small


def test_can_write_tiff_file():
    f = io.BytesIO()
    tiff_from_nexus(
        iron_simulation_sample_small(),
        f,
        time_bins=50,
        pulse_stride=2,
    )
    f.seek(0)
    imread(f)
