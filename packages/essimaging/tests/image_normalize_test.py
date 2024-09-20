# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing.assertions import assert_allclose, assert_identical

from ess.imaging.io import (
    DarkCurrentImageStacks,
    OpenBeamImageStacks,
    SampleImageStacks,
)
from ess.imaging.normalize import (
    BackgroundImage,
    DarkCurrentImage,
    NormalizedSampleImages,
    OpenBeamImage,
    ScaleFactor,
    average_dark_current_images,
    average_open_beam_images,
    cleanse_sample_images,
    normalize_sample_images,
)
from ess.imaging.workflow import YmirWorkflow


@pytest.fixture
def open_beam_images() -> OpenBeamImageStacks:
    return OpenBeamImageStacks(
        sc.DataArray(
            data=sc.array(
                dims=["time", "dim_1", "dim_2"],
                values=[[[3.0, 3.0], [3.0, 0.0]], [[5.0, 5.0], [5.0, 0.0]]],
                unit="counts",
            ),
            coords={
                "time": sc.array(dims=["time"], values=[1, 2], unit="s"),
            },
        )
    )


@pytest.fixture
def dark_current_images() -> DarkCurrentImageStacks:
    return DarkCurrentImageStacks(
        sc.DataArray(
            data=sc.array(
                dims=["time", "dim_1", "dim_2"],
                values=[[[0.5, 0.5], [0.5, 0.5]], [[1.5, 1.5], [1.5, 1.5]]],
                unit="counts",
            ),
            coords={
                "time": sc.array(dims=["time"], values=[1, 2], unit="s"),
            },
        )
    )


@pytest.fixture
def sample_images() -> SampleImageStacks:
    return SampleImageStacks(
        sc.DataArray(
            data=sc.array(
                dims=["time", "dim_1", "dim_2"],
                values=[[[2.0, 2.0], [2.0, 0.0]], [[4.0, 4.0], [4.0, 0.0]]],
                unit="counts",
            ),
            coords={
                "time": sc.array(dims=["time"], values=[1, 2], unit="s"),
            },
        )
    )


@pytest.fixture
def open_beam_image(open_beam_images: OpenBeamImageStacks) -> sc.DataArray:
    with pytest.warns(
        expected_warning=UserWarning,
        match="Calculating average open beam image assuming constant exposure time.",
    ):
        return average_open_beam_images(open_beam_images)


@pytest.fixture
def dark_current_image(dark_current_images: DarkCurrentImageStacks) -> DarkCurrentImage:
    with pytest.warns(
        expected_warning=UserWarning,
        match="Calculating average dark current image assuming constant exposure time.",
    ):
        return average_dark_current_images(dark_current_images)


def test_average_open_beam_images(open_beam_images: OpenBeamImageStacks) -> None:
    expected_average_open_beam_image = sc.DataArray(
        data=sc.array(
            dims=["dim_1", "dim_2"],
            values=[[4.0, 4.0], [4.0, 0.0]],
            unit="counts",
        ),
        coords={},
    )

    with pytest.warns(
        expected_warning=UserWarning,
        match="Calculating average open beam image assuming constant exposure time.",
    ):
        assert_identical(
            average_open_beam_images(open_beam_images), expected_average_open_beam_image
        )


def test_average_dark_current_images(
    dark_current_images: DarkCurrentImageStacks,
) -> None:
    expected_average_dark_current_image = sc.DataArray(
        data=sc.array(
            dims=["dim_1", "dim_2"],
            values=[[1.0, 1.0], [1.0, 1.0]],
            unit="counts",
        ),
        coords={},
    )

    with pytest.warns(
        expected_warning=UserWarning,
        match="Calculating average dark current image assuming constant exposure time.",
    ):
        assert_identical(
            average_dark_current_images(dark_current_images),
            expected_average_dark_current_image,
        )


def test_calculate_white_beam_background(
    open_beam_image: OpenBeamImage, dark_current_image: DarkCurrentImage
) -> None:
    from ess.imaging.normalize import calculate_white_beam_background

    expected_background_image = sc.DataArray(
        data=sc.array(
            dims=["dim_1", "dim_2"],
            values=[[3.0, 3.0], [3.0, 1.0]],
            # last pixel value will be replaced with 1.0
            unit="counts",
        ),
        coords={},
    )

    assert_identical(
        calculate_white_beam_background(open_beam_image, dark_current_image),
        expected_background_image,
    )


def test_cleanse_sample_images(
    sample_images: SampleImageStacks, dark_current_image: DarkCurrentImage
) -> None:
    expected_cleansed_sample_image = sc.DataArray(
        data=sc.array(
            dims=["time", "dim_1", "dim_2"],
            values=[[[1.0, 1.0], [1.0, 0.0]], [[3.0, 3.0], [3.0, 0.0]]],
            unit="counts",
        ),
        coords={
            "time": sc.array(dims=["time"], values=[1, 2], unit="s"),
        },
    )
    assert_identical(
        cleanse_sample_images(sample_images, dark_current_image),
        expected_cleansed_sample_image,
    )


def test_normalize_negative_scale_factor_raises(
    sample_images: SampleImageStacks,
    dark_current_image: DarkCurrentImage,
) -> None:
    cleansed_sample_image = cleanse_sample_images(sample_images, dark_current_image)
    with pytest.raises(ValueError, match="Scale factor must be positive,"):
        normalize_sample_images(
            cleansed_sample_image,
            ScaleFactor(sc.scalar(-1.0, unit="dimensionless")),
            BackgroundImage(dark_current_image),
        )


def test_normalize_workflow(
    sample_images: SampleImageStacks,
    open_beam_images: OpenBeamImageStacks,
    dark_current_images: DarkCurrentImageStacks,
) -> None:
    expected_normalized_sample_images = sc.DataArray(
        data=sc.array(
            dims=["time", "dim_1", "dim_2"],
            values=[
                [[1 / (3 * 5 / 3), 1 / (3 * 5 / 3)], [1 / (3 * 5 / 3), 0.0]],
                [[3 / (3 * 5 / 3), 3 / (3 * 5 / 3)], [3 / (3 * 5 / 3), 0.0]],
            ],
            unit="counts",
        ),
        coords={
            "time": sc.array(dims=["time"], values=[1, 2], unit="s"),
        },
    )

    wf = YmirWorkflow()
    wf[SampleImageStacks] = sample_images
    wf[OpenBeamImageStacks] = open_beam_images
    wf[DarkCurrentImageStacks] = dark_current_images
    mean_ob_warning_msg = (
        "Calculating average open beam image assuming constant exposure time."
    )
    mean_dc_warning_msg = (
        "Calculating average dark current image assuming constant exposure time."
    )
    mean_sample_warning_msg = (
        "Calculating average sample pixel counts assuming constant exposure time."
    )
    bg_image_warning_msg = (
        "Calculating average background pixel counts assuming constant exposure time."
    )
    normalize_warning_msg = "Normalizing sample images assuming constant exposure time."
    with (
        # Following warnings are expected to be raise
        # until we use the correct exposure time in the data files
        pytest.warns(expected_warning=UserWarning, match=mean_ob_warning_msg),
        pytest.warns(expected_warning=UserWarning, match=mean_dc_warning_msg),
        pytest.warns(expected_warning=UserWarning, match=mean_sample_warning_msg),
        pytest.warns(expected_warning=UserWarning, match=bg_image_warning_msg),
        pytest.warns(expected_warning=UserWarning, match=normalize_warning_msg),
    ):
        normalized = wf.compute(NormalizedSampleImages)
        assert isinstance(normalized, sc.DataArray)
        assert normalized.sizes['time'] == 2
        assert_allclose(normalized, expected_normalized_sample_images)
