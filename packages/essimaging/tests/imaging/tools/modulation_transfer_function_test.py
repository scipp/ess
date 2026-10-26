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
    normalized_radial_spectrum,
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


@pytest.mark.parametrize(
    'shape', [(128, 128), (127, 127), (128, 256), (127, 257), (32, 256)]
)
def test_normalized_radial_spectrum_keeps_dc_separate(shape):
    image = sc.DataArray(sc.ones(dims=['y', 'x'], shape=shape))

    spectrum = normalized_radial_spectrum(image)

    assert spectrum.coords['frequency'][0] == sc.scalar(0.0)
    assert spectrum['frequency', 0].value == pytest.approx(1.0)
    np.testing.assert_allclose(spectrum.values[1:], 0.0, atol=1e-15)
    assert np.all(np.diff(spectrum.coords['frequency'].values) > 0)
    assert spectrum.coords['frequency'].max() <= sc.scalar(0.5)


@pytest.mark.parametrize(
    'shape', [(128, 128), (127, 127), (128, 256), (127, 257), (32, 256)]
)
def test_normalized_radial_spectrum_sinusoid_frequency(shape):
    y, x = np.indices(shape)
    ky, kx = 3, 9
    values = 2 + np.cos(2 * np.pi * (ky * y / shape[0] + kx * x / shape[1]))
    image = sc.DataArray(sc.array(dims=['y', 'x'], values=values))

    spectrum = normalized_radial_spectrum(image)['frequency', 1:]

    peak = spectrum.coords['frequency'].values[np.argmax(spectrum.values)]
    expected = np.hypot(ky / shape[0], kx / shape[1])
    assert peak == pytest.approx(expected, abs=1 / max(shape))


@pytest.mark.parametrize(
    'shape', [(128, 128), (127, 127), (128, 256), (127, 257), (32, 256)]
)
@pytest.mark.parametrize('sigma', [1.5, 3.0])
def test_modulation_transfer_function_gaussian_blur(shape, sigma):
    # A periodic convolution avoids ROI boundary effects, isolating the MTF estimator.
    rng = np.random.default_rng(42)
    reference = rng.uniform(0.25, 1.0, size=shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    transfer = np.exp(-2 * np.pi**2 * sigma**2 * (fx**2 + fy**2))
    values = np.fft.ifft2(np.fft.fft2(reference) * transfer).real
    target = sc.DataArray(sc.array(dims=['y', 'x'], values=reference))
    image = sc.DataArray(sc.array(dims=['y', 'x'], values=values))
    open_beam = sc.DataArray(sc.ones(dims=['y', 'x'], shape=shape))

    mtf = modulation_transfer_function(image, open_beam, target)

    frequency = mtf.coords['frequency'].values
    assert frequency[0] == 0.0
    assert mtf.values[0] == pytest.approx(1.0, abs=1e-14)
    # A bin averages several frequencies, weighted by the reference spectrum.
    # Its response must lie within the analytic values one bin width either side.
    width = 1 / max(shape)
    lower = np.exp(-2 * np.pi**2 * sigma**2 * (frequency + width) ** 2)
    upper = np.exp(-2 * np.pi**2 * sigma**2 * np.maximum(frequency - width, 0) ** 2)
    assert np.all(mtf.values >= lower - 1e-12)
    assert np.all(mtf.values <= upper + 1e-12)
    expected_mtf10 = np.sqrt(np.log(10)) / (np.sqrt(2) * np.pi * sigma)
    assert mtf_less_than(mtf, sc.scalar(0.1)).value == pytest.approx(
        expected_mtf10, abs=1.5 / max(shape)
    )
