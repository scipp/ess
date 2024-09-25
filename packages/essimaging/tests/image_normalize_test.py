# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing.assertions import assert_allclose, assert_identical

from ess.imaging.io import (
    DarkCurrentImageStacks,
    OpenBeamImageStacks,
    RawSampleImageStacks,
    SampleImageStacksWithLogs,
)
from ess.imaging.normalize import (
    BackgroundImage,
    BackgroundPixelThreshold,
    CleansedOpenBeamImage,
    CleansedSampleImages,
    DarkCurrentImage,
    NormalizedSampleImages,
    OpenBeamImage,
    SamplePixelThreshold,
    ScaleFactor,
    apply_threshold_to_background_image,
    apply_threshold_to_sample_images,
    average_dark_current_images,
    average_open_beam_images,
    cleanse_sample_images,
    normalize_sample_images,
)
from ess.imaging.workflow import YmirImageNormalizationWorkflow


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
def sample_images() -> RawSampleImageStacks:
    return RawSampleImageStacks(
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
        match="Computing average open beam image assuming constant exposure time.",
    ):
        return average_open_beam_images(open_beam_images)


@pytest.fixture
def dark_current_image(dark_current_images: DarkCurrentImageStacks) -> DarkCurrentImage:
    with pytest.warns(
        expected_warning=UserWarning,
        match="Computing average dark current image assuming constant exposure time.",
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
        match="Computing average open beam image assuming constant exposure time.",
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
        match="Computing average dark current image assuming constant exposure time.",
    ):
        assert_identical(
            average_dark_current_images(dark_current_images),
            expected_average_dark_current_image,
        )


def test_cleanse_open_beam_image(
    open_beam_image: OpenBeamImage, dark_current_image: DarkCurrentImage
) -> None:
    from ess.imaging.normalize import cleanse_open_beam_image

    expected_background_image = sc.DataArray(
        data=sc.array(
            dims=["dim_1", "dim_2"],
            values=[[3.0, 3.0], [3.0, -1.0]],
            unit="counts",
        ),
        coords={},
    )

    assert_identical(
        cleanse_open_beam_image(open_beam_image, dark_current_image),
        expected_background_image,
    )


def test_cleanse_sample_images(
    sample_images: SampleImageStacksWithLogs, dark_current_image: DarkCurrentImage
) -> None:
    expected_cleansed_sample_image = sc.DataArray(
        data=sc.array(
            dims=["time", "dim_1", "dim_2"],
            values=[[[1.0, 1.0], [1.0, -1.0]], [[3.0, 3.0], [3.0, -1.0]]],
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


def test_apply_threshold_to_sample_images() -> None:
    sample_images_with_negative_values = sc.DataArray(
        data=sc.array(
            dims=["time", "dim_1", "dim_2"],
            values=[[[2.0, 2.0], [2.0, -1.0]], [[4.0, 4.0], [4.0, -1.0]]],
            unit="counts",
        ),
        coords={
            "time": sc.array(dims=["time"], values=[1, 2], unit="s"),
        },
    )
    threshold = sc.scalar(1.0, unit="counts")
    thresholded_sample_images = apply_threshold_to_sample_images(
        CleansedSampleImages(sample_images_with_negative_values),
        SamplePixelThreshold(threshold),
    )
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_identical(sample_images_with_negative_values.data.min(), threshold)
    assert_identical(thresholded_sample_images.data.min(), threshold)


def test_apply_threshold_to_background_image() -> None:
    background_image_with_negative_values = sc.DataArray(
        data=sc.array(
            dims=["dim_1", "dim_2"],
            values=[[3.0, 3.0], [3.0, -1.0]],
            unit="counts",
        ),
        coords={},
    )
    threshold = sc.scalar(1.0, unit="counts")
    thresholded_background_image = apply_threshold_to_background_image(
        CleansedOpenBeamImage(background_image_with_negative_values),
        BackgroundPixelThreshold(threshold),
    )
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_identical(background_image_with_negative_values.data.min(), threshold)
    assert_identical(thresholded_background_image.data.min(), threshold)


def test_normalize_negative_scale_factor_raises(
    sample_images: SampleImageStacksWithLogs,
    dark_current_image: DarkCurrentImage,
) -> None:
    cleansed_sample_image = apply_threshold_to_sample_images(
        cleanse_sample_images(sample_images, dark_current_image),
        SamplePixelThreshold(sc.scalar(0.0, unit="counts")),
    )

    with pytest.raises(ValueError, match="Scale factor must be positive,"):
        normalize_sample_images(
            samples=cleansed_sample_image,
            background=BackgroundImage(dark_current_image),
            factor=ScaleFactor(sc.scalar(-1.0, unit="dimensionless")),
        )


def test_normalize_workflow(
    sample_images: RawSampleImageStacks,
    open_beam_images: OpenBeamImageStacks,
    dark_current_images: DarkCurrentImageStacks,
) -> None:
    expected_normalized_sample_images = sc.DataArray(
        data=sc.array(
            dims=["time", "dim_1", "dim_2"],
            values=[
                [[1 / 3 * (5 / 3), 1 / 3 * (5 / 3)], [1 / 3 * (5 / 3), 0.0]],
                [[3 / 3 * (5 / 3), 3 / 3 * (5 / 3)], [3 / 3 * (5 / 3), 0.0]],
            ],
            unit="counts",
        ),
        coords={
            "time": sc.array(dims=["time"], values=[1, 2], unit="s"),
        },
    )

    wf = YmirImageNormalizationWorkflow()
    wf[SampleImageStacksWithLogs] = sample_images
    wf[OpenBeamImageStacks] = open_beam_images
    wf[DarkCurrentImageStacks] = dark_current_images
    mean_ob_warning_msg = (
        "Computing average open beam image assuming constant exposure time."
    )
    mean_dc_warning_msg = (
        "Computing average dark current image assuming constant exposure time."
    )
    mean_sample_warning_msg = (
        "Computing average sample pixel counts assuming constant exposure time."
    )
    bg_image_warning_msg = (
        "Computing average background pixel counts assuming constant exposure time."
    )
    normalize_warning_msg = (
        "Computing normalized sample image stack assuming constant exposure time."
    )
    with (
        # Following warnings are expected to be raised
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
        assert normalized.unit == "dimensionless"
        assert_allclose(normalized, expected_normalized_sample_images)
