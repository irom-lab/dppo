import numpy as np
import pytest

from env.gym_utils.wrapper.robomimic_image import RobomimicImageWrapper


class _DummyEnv:
    action_dimension = 7


def _wrapper(image_keys=("agentview_image",), channels=3, height=2, width=4):
    return RobomimicImageWrapper(
        env=_DummyEnv(),
        shape_meta={
            "obs": {
                "rgb": {"shape": (channels * len(image_keys), height, width)},
                "state": {"shape": (3,)},
            }
        },
        image_keys=list(image_keys),
    )


def _raw_obs(images):
    return {
        **{name: image for name, image in images.items()},
        "robot0_eef_pos": np.array([0.1], dtype=np.float32),
        "robot0_eef_quat": np.array([0.2], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.3], dtype=np.float32),
    }


def test_hwc_float_observation_is_transposed_and_scaled():
    image = np.arange(24, dtype=np.float32).reshape(2, 4, 3) / 255.0
    wrapper = _wrapper(height=2, width=4)

    result = wrapper.get_observation(_raw_obs({"agentview_image": image}))

    np.testing.assert_allclose(result["rgb"], image.transpose(2, 0, 1) * 255.0)
    assert result["rgb"].shape == (3, 2, 4)
    assert result["rgb"].dtype == np.float32


def test_chw_float_observation_keeps_channel_order():
    image = np.arange(24, dtype=np.float32).reshape(3, 2, 4) / 255.0
    wrapper = _wrapper(height=2, width=4)

    result = wrapper.get_observation(_raw_obs({"agentview_image": image}))

    np.testing.assert_allclose(result["rgb"], image * 255.0)
    assert result["rgb"].shape == image.shape


def test_multiple_hwc_cameras_are_concatenated_on_channels():
    first = np.zeros((2, 4, 3), dtype=np.float32)
    second = np.ones((2, 4, 3), dtype=np.float32)
    wrapper = _wrapper(
        image_keys=("agentview_image", "robot0_eye_in_hand_image"),
        height=2,
        width=4,
    )

    result = wrapper.get_observation(
        _raw_obs(
            {
                "agentview_image": first,
                "robot0_eye_in_hand_image": second,
            }
        )
    )

    assert result["rgb"].shape == (6, 2, 4)
    np.testing.assert_allclose(result["rgb"][:3], 0.0)
    np.testing.assert_allclose(result["rgb"][3:], 255.0)


def test_uint8_observation_does_not_wrap_when_converted_to_float():
    image = np.array(
        [
            [[0, 1, 2], [253, 254, 255]],
            [[10, 20, 30], [40, 50, 60]],
        ],
        dtype=np.uint8,
    )
    wrapper = _wrapper(height=2, width=2)

    result = wrapper.get_observation(_raw_obs({"agentview_image": image}))

    np.testing.assert_array_equal(result["rgb"], image.transpose(2, 0, 1))
    assert result["rgb"].dtype == np.float32


def test_uint8_zero_one_values_are_not_treated_as_normalized_floats():
    image = np.array(
        [
            [[0, 1, 0], [1, 0, 1]],
            [[0, 0, 1], [1, 1, 0]],
        ],
        dtype=np.uint8,
    )
    wrapper = _wrapper(height=2, width=2)

    result = wrapper.get_observation(_raw_obs({"agentview_image": image}))

    np.testing.assert_array_equal(result["rgb"], image.transpose(2, 0, 1))


@pytest.mark.parametrize(
    "image",
    [np.zeros((2, 3), dtype=np.float32), np.zeros((2, 4, 5), dtype=np.float32)],
)
def test_invalid_rgb_shape_raises_instead_of_reaching_conv2d(image):
    wrapper = _wrapper(height=2, width=4)

    with pytest.raises(ValueError, match="RGB"):
        wrapper.get_observation(_raw_obs({"agentview_image": image}))
