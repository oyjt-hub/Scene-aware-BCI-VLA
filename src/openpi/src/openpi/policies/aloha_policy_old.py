import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha将夹爪位置转换到了一个线性空间中。以下代码
    # 将这个变换逆转回来，以便与在角度空间中预训练的pi0模型保持一致。
    #
    # 这些值来自Aloha代码中的常量：
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    # (注：这里的“线性空间”可能指的就是滑块的物理距离，单位是米)
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # 这是Interbotix代码库中，从角度到线性变换的逆运算。
    # (注：Interbotix是您夹爪的品牌，这是他们官方提供的运动学逆解公式)
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        # 警告：这里使用的是 arcsin (反正弦)，而不是我们之前讨论的 arccos (反余弦)。
        # 这意味着这里的几何模型或角度定义可能与之前的版本有所不同。
        # 这是一个非常关键的区别！
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # 这些常量取自Interbotix的代码。
    # (注：arm_length是连杆长度，horn_radius是舵机摇臂半径)
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0模型的夹爪数据是在编码器计数(2405, 3110)之间进行归一化(0, 1)的。
    # 编码器总共有4096个计数单位，而Aloha系统使用2048作为零点。
    # 将这些编码器计数转换为弧度，意味着归一化后的输入范围是在(0.5476, 1.6296)之间。
    # (注：这解释了0.5476和1.6296这两个“魔法数字”的来源。它们是pi0模型训练时
    #  所见到的夹爪开合角度的最小值和最大值，单位是弧度)
    return _normalize(value, min_val=0.5476, max_val=1.6296)

def _gripper_from_angular(value):
    # 将pi0模型使用的夹爪位置，转换为Aloha系统使用的夹爪位置。
    # 注意，单位仍然是角度（弧度），但是数值范围是不同的。
    # (注：这是一个模型空间到另一个模型/控制空间的转换)

    # 我们不对输出进行缩放，因为trossen
    # 的预测值已经是弧度单位了。
    # 关于这个常数的推导，请参见_gripper_to_angular函数中的注释。
    # (注：这里的核心操作是平移。pi0模型的输出范围是[0, 1]，加上这个偏移量
    #  就将其转换回了pi0模型原始的物理角度范围 [0.5476, 1.6296])
    value = value + 0.5476

    # 这些值来自Aloha代码中的常量：
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    # (注：这是Aloha系统能够理解和执行的夹爪关节角度的极限范围)
    return _normalize(value, min_val=-0.6213, max_val=1.4910)

def _gripper_from_angular_inv(value):
    # 直接对 gripper_from_angular 函数进行逆运算。
    # (注：这个函数通常用于数据记录或验证，确保数据转换可以无损地来回进行)
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476

def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions