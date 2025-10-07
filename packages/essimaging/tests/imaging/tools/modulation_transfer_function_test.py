import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose
from scipy.signal import fftconvolve

from ess.imaging import data
from ess.imaging.tools import (
    estimate_cut_off_frequency,
    modulation_transfer_function,
    mtf_less_than,
)


def create_star(shape, cx, cy, spokes):
    '''Generates a Siemens star target similar to the one used in the experiment.

    Parameters
    ---------------------
    shape:
        The requested image shape.
    cx:
        The second coordinate of center of the star in the image.
    cy:
        The first coordinate of center of the star in the image.
    spokes:
        The number of spokes in the star.
    '''
    n = spokes
    y, x = np.indices(shape)
    sf = 1 / (np.prod(shape) / 1024**2) ** 0.5
    r = np.hypot(x - int(cx / sf), y - int(cy / sf))
    t = np.atan2(y - int(cy / sf), x - int(cx / sf))
    return sc.DataArray(
        sc.array(
            dims=('y', 'x'),
            values=(
                np.where(
                    (t % (2 * np.pi / n) > (2 * np.pi / n) / 2) | (r < int(100 / sf)),
                    1,
                    0,
                )
                * np.where(r < int(50 / sf), 0, 1)
                * np.where((r < int(110 / sf)) & (r > int(90 / sf)), 0, 1)
                * np.where((r < int(250 / sf)) & (r > int(230 / sf)), 0, 1)
                * np.where((r < int(470 / sf)) & (r > int(450 / sf)), 0, 1)
            ).astype(np.int32),
        )
    )


@pytest.mark.parametrize(
    ('xlims', 'ylims'),
    [
        ((150, 450), (150, 450)),
        ((350, 750), (150, 450)),
        ((150, 450), (350, 750)),
        ((150, 450), (150, 750)),
        ((150, 750), (150, 450)),
    ],
)
def test_modulation_transfer_function(xlims, ylims):
    measured = sc.io.hdf5.load_hdf5(data.jparc_siemens_star_measured_path())
    openbeam = sc.io.hdf5.load_hdf5(data.jparc_siemens_star_openbeam_path())
    target = create_star(measured.shape, 642, 630, 135)

    slicex = ('x', slice(*xlims))
    slicey = ('y', slice(*ylims))

    mtf = modulation_transfer_function(
        measured[slicey][slicex],
        openbeam[slicey][slicex],
        target[slicey][slicex],
    )
    assert_allclose(
        estimate_cut_off_frequency(mtf), sc.scalar(0.05), atol=sc.scalar(0.01)
    )
    assert_allclose(mtf_less_than(mtf, 0.1), sc.scalar(0.046), atol=sc.scalar(0.01))
    assert isinstance(mtf, sc.DataArray)
    assert 'frequency' in mtf.coords


@pytest.mark.parametrize(
    ('xlims', 'ylims'),
    [
        ((150, 450), (150, 450)),
        ((350, 750), (150, 450)),
        ((150, 450), (350, 750)),
        ((150, 450), (150, 750)),
        ((150, 750), (150, 450)),
    ],
)
def test_modulation_transfer_function_ideal_case(xlims, ylims):
    N = 1024
    f_c = 1 / 5
    target = create_star((N, N), N / 2, N / 2, 135)

    x, y = np.meshgrid(np.arange(1024), np.arange(1024))
    p = (((x - N / 2) ** 2 + (y - N / 2) ** 2) ** 0.5 < N / 2 * f_c).astype('float64')
    P = np.abs(np.fft.fftshift(np.fft.fft2(p))) ** 2
    image = fftconvolve(target.values, P, mode='same')

    image = sc.DataArray(sc.array(dims=('y', 'x'), values=image))
    ob = sc.DataArray(sc.array(dims=('y', 'x'), values=np.ones((N, N))))

    slicex = ('x', slice(*xlims))
    slicey = ('y', slice(*ylims))

    mtf = modulation_transfer_function(
        image[slicey][slicex],
        ob[slicey][slicex],
        target[slicey][slicex],
    )
    assert_allclose(
        estimate_cut_off_frequency(mtf), sc.scalar(f_c), rtol=sc.scalar(5e-2)
    )
