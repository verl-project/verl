# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Giga Team. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# from https://github.com/open-gigaai/giga-models


from typing import Any

import torch
import torch.nn.functional as F
from torchvision import transforms


class Normalize:
    """Normalize robot state vectors using mean/std or quantiles.

    Args:
        stats: A dict containing either {'mean', 'std'} or {'q01', 'q99'}.
        use_quantiles: If True, use quantile based normalization.
    """

    def __init__(self, stats: dict[str, Any], *, use_quantiles: bool = False) -> None:
        self.EPSILON = 1e-6
        self.stats = stats
        self.use_quantiles = use_quantiles

        required_attrs = ["mean", "std"]
        if self.use_quantiles:
            required_attrs = ["q01", "q99"]

        for attr in required_attrs:
            if attr not in stats:
                raise AttributeError(f"stats object is missing the following attribute: {attr}")

        if self.use_quantiles:
            self.q01 = torch.tensor(stats["q01"], dtype=torch.float32)
            self.q99 = torch.tensor(stats["q99"], dtype=torch.float32)
        else:
            self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self.std = torch.tensor(stats["std"], dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:
        if self.use_quantiles:
            self.q01 = self.q01.to(device)
            self.q99 = self.q99.to(device)
        else:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_dim = x.shape[-1]
        if self.use_quantiles:
            return (x - self.q01[..., :x_dim]) / (
                self.q99[..., :x_dim] - self.q01[..., :x_dim] + self.EPSILON
            ) * 2.0 - 1.0
        else:
            return (x - self.mean[..., :x_dim]) / (self.std[..., :x_dim] + self.EPSILON)


class Unnormalize:
    def __init__(self, stats, *, use_quantiles: bool = False):
        self.EPSILON = 1e-6
        self.stats = stats
        self.use_quantiles = use_quantiles

        if self.use_quantiles:
            self.q01 = torch.tensor(stats["q01"], dtype=torch.float32)
            self.q99 = torch.tensor(stats["q99"], dtype=torch.float32)
        else:
            self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self.std = torch.tensor(stats["std"], dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:
        if self.use_quantiles:
            self.q01 = self.q01.to(device)
            self.q99 = self.q99.to(device)
        else:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_dim = x.shape[-1]
        if self.use_quantiles:
            return (x + 1.0) / 2.0 * (self.q99[..., :x_dim] - self.q01[..., :x_dim] + self.EPSILON) + self.q01[
                ..., :x_dim
            ]
        else:
            return x * (self.std[..., :x_dim] + self.EPSILON) + self.mean[..., :x_dim]


class DeltaActions:
    """Repacks absolute actions into delta action space."""

    def __init__(self):
        # If the robot has mobile base, masks of base action are False and it doesn't need to be specified explicitly.
        self.mask = torch.tensor([True, True, True, True, True, True, False, True, True, True, True, True, True, False])

    def to(self, device: torch.device | str) -> None:
        self.mask = self.mask.to(device)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "action" not in data or "observation.state" not in data:
            return data
        state, action = data["observation.state"], data["action"]
        dims = self.mask.shape[-1]
        action[..., :dims] -= torch.where(self.mask, state[..., :dims], torch.zeros_like(state[..., :dims])).unsqueeze(
            -2
        )
        data["action"] = action
        return data


class AbsoluteActions:
    """Repacks delta actions into absolute action space."""

    def __init__(self):
        # If the robot has mobile base, masks of base action are False and it doesn't need to be specified explicitly.
        self.mask = torch.tensor([True, True, True, True, True, True, False, True, True, True, True, True, True, False])

    def to(self, device: torch.device | str) -> None:
        self.mask = self.mask.to(device)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "action" not in data or "observation.state" not in data:
            return data
        state, action = data["observation.state"], data["action"]
        dims = self.mask.shape[-1]
        action[..., :dims] += torch.where(self.mask, state[..., :dims], torch.zeros_like(state[..., :dims])).unsqueeze(
            -2
        )
        data["action"] = action
        return data


class AlohaInputs:
    """Inputs for the Aloha policy."""

    def __init__(self, adapt_to_pi: bool = True) -> None:
        self.joint_flip_mask = torch.tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        self.adapt_to_pi = adapt_to_pi

    def to(self, device: torch.device | str) -> None:
        self.joint_flip_mask = self.joint_flip_mask.to(device)

    def _gripper_from_angular_inv(self, value: torch.Tensor) -> torch.Tensor:
        # Directly inverts the gripper_from_angular function.
        value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
        return value - 0.5476

    def _gripper_to_angular(self, value: torch.Tensor) -> torch.Tensor:
        # Aloha transforms the gripper positions into a linear space. The following code
        # reverses this transformation to be consistent with pi0 which is pretrained in
        # angular space.
        #
        # These values are coming from the Aloha code:
        # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
        value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

        # This is the inverse of the angular to linear transformation inside the Interbotix code.
        def linear_to_radian(linear_position, arm_length, horn_radius):
            value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
            return torch.arcsin(torch.clip(value, -1.0, 1.0))

        # The constants are taken from the Interbotix code.
        value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

        # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
        # There are 4096 total encoder counts and aloha uses a zero of 2048.
        # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
        return _normalize(value, min_val=0.5476, max_val=1.6296)

    def _encode_actions_inv(self, actions: torch.Tensor) -> torch.Tensor:
        if self.adapt_to_pi:
            actions[:, :14] = self.joint_flip_mask * actions[:, :14]
            actions[:, [6, 13]] = self._gripper_from_angular_inv(actions[:, [6, 13]])
        return actions

    def _decode_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.adapt_to_pi:
            # Flip the joints.
            state[:14] = self.joint_flip_mask * state[:14]
            # Reverse the gripper transformation that is being applied by the Aloha runtime.
            state[[6, 13]] = self._gripper_to_angular(state[[6, 13]])
        return state

    def _decode_aloha(self, state: torch.Tensor) -> torch.Tensor:
        # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
        # dim sizes: [6, 1, 6, 1]
        state = self._decode_state(state)
        return state

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Decode Aloha-specific input formats into the pi0 training/runtime
        format."""
        state = self._decode_aloha(data["observation.state"])
        data["observation.state"] = state
        # Actions are only available during training.
        if "action" in data:
            actions = data["action"]
            actions = self._encode_actions_inv(actions)
            data["action"] = actions
        return data

    # VeRL: Batch Inference

    def _encode_actions_inv_batch(self, actions: torch.Tensor) -> torch.Tensor:
        if self.adapt_to_pi:
            actions[..., :14] = self.joint_flip_mask * actions[..., :14]
            actions[..., [6, 13]] = self._gripper_from_angular_inv(actions[..., [6, 13]])
        return actions

    def _decode_state_batch(self, state: torch.Tensor) -> torch.Tensor:
        if self.adapt_to_pi:
            state[..., :14] = self.joint_flip_mask * state[..., :14]
            state[..., [6, 13]] = self._gripper_to_angular(state[..., [6, 13]])
        return state

    def call_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        state = self._decode_state_batch(data["observation.state"])
        data["observation.state"] = state
        if "action" in data:
            actions = data["action"]
            actions = self._encode_actions_inv_batch(actions)
            data["action"] = actions
        return data


class AlohaOutputs:
    """Outputs for the Aloha policy."""

    def __init__(self, original_action_dim: int, adapt_to_pi: bool = True):
        """
        Args:
            original_action_dim: int. The original action dimension of the policy. dual-arm robot has 14 dims and mobile
                                      dual-arm robot has 16 dims.
            adapt_to_pi: bool. If true, this will convert the joint and gripper values from the standard Aloha space to
            the space used by the pi internal runtime which was used to train the base model.
        """
        self.joint_flip_mask = torch.tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        self.original_action_dim = original_action_dim
        self.adapt_to_pi = adapt_to_pi

    def to(self, device: torch.device | str) -> None:
        self.joint_flip_mask = self.joint_flip_mask.to(device)

    def _gripper_from_angular(self, value: torch.Tensor) -> torch.Tensor:
        # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
        # Note that the units are still angular but the range is different.

        # We do not scale the output since the trossen model predictions are already in radians.
        # See the comment in _gripper_to_angular for a derivation of the constant
        value = value + 0.5476

        # These values are coming from the Aloha code:
        # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
        return _normalize(value, min_val=-0.6213, max_val=1.4910)

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.adapt_to_pi:
            # Flip the joints.
            actions[:, :14] = self.joint_flip_mask * actions[:, :14]
            actions[:, [6, 13]] = self._gripper_from_angular(actions[:, [6, 13]])
        return actions

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = data["action"][:, : self.original_action_dim]
        return {"action": self._encode_actions(actions)}

    # VeRL: Batch Inference

    def _encode_actions_batch(self, actions: torch.Tensor) -> torch.Tensor:
        if self.adapt_to_pi:
            actions[..., :14] = self.joint_flip_mask * actions[..., :14]
            actions[..., [6, 13]] = self._gripper_from_angular(actions[..., [6, 13]])
        return actions

    def call_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = data["action"][..., : self.original_action_dim]
        return {"action": self._encode_actions_batch(actions)}


class PadStatesAndActions:
    """Zero-pads states and actions to the model action dimension."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def _pad_to_dim(self, x: torch.Tensor, target_dim: int, axis: int = -1) -> torch.Tensor:
        """Pad an array to the target dimension with zeros along the specified
        axis."""
        current_dim = x.shape[axis]
        if current_dim < target_dim:
            shape = list(x.shape)
            shape[-1] = target_dim
            new_vector = torch.zeros(*shape, dtype=x.dtype, device=x.device)
            new_vector[..., :current_dim] = x
            x = new_vector
        return x

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data["observation.state"] = self._pad_to_dim(data["observation.state"], self.action_dim, axis=-1)
        if "action" in data:
            data["action"] = self._pad_to_dim(data["action"], self.action_dim, axis=-1)
        return data


def _normalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return x * (max_val - min_val) + min_val


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1.0) -> torch.Tensor:
    """Resize an image to fit inside the given (width, height) while preserving
    aspect ratio, then pad with the specified value so that the final image
    exactly matches the target size.

    Args:
        img: Input image, shape (C, H, W), with values typically in [0, 1].
        width: Target width (W).
        height: Target height (H).
        pad_value: Value to use for padding, defaults to -1.

    Returns:
        A torch.Tensor of shape (C, height, width).
    """
    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f"(C,H,W) expected, but got {img.shape}")

    cur_height, cur_width = img.shape[1:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img.unsqueeze(0), size=(resized_height, resized_width), mode="bilinear", align_corners=False
    ).squeeze(0)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_img = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
    return padded_img.squeeze(0)


class ImageTransform:
    def __init__(
        self,
        resize_imgs_with_padding: tuple[int, int],
        present_img_keys: list[str] | None = None,
        enable_image_aug: bool = False,
    ) -> None:
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.present_img_keys = present_img_keys
        if self.present_img_keys is None:
            self.present_img_keys = [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]
        self.enable_image_aug = enable_image_aug
        self.width, self.height = resize_imgs_with_padding
        if self.enable_image_aug:
            self.color_jitter_transform = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
            )
            self.pose_transform = transforms.Compose(
                [
                    transforms.RandomCrop(int(self.width * 0.95), int(self.height * 0.95)),
                    transforms.Resize((self.width, self.height)),
                    transforms.RandomRotation((-5, 5)),
                ]
            )

    def __call__(self, data: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Preprocesses input images: optionally scales and pads to a fixed size,
        then maps the pixel range from [0,1] to [-1,1].

        Returns two lists:
            images: The processed image arrays (C, H, W).
            img_masks: A list of boolean masks of the same length as images, currently fixed to True.
        """
        images = []
        img_masks = []

        for key in self.present_img_keys:
            if key not in data:
                raise ValueError(
                    f"{key} not found in data. Please check the present_img_keys in the config or the dataset."
                )

            img = data[key]
            # [C, H, W] -> preprocess
            if self.resize_imgs_with_padding is not None:
                original_height, original_width = img.shape[1:]
                target_height, target_width = self.resize_imgs_with_padding
                if original_height != target_height or original_width != target_width:
                    img = resize_with_pad(img, *self.resize_imgs_with_padding, pad_value=0)

            if self.enable_image_aug:
                if "wrist" not in key:
                    img = self.pose_transform(img)
                img = self.color_jitter_transform(img)

            # Normalize pixel values to [-1, 1]
            img = img * 2.0 - 1.0

            images.append(img)
            img_masks.append(torch.tensor(True, dtype=torch.bool, device=img.device))

        return images, img_masks

    # VeRL: Batch Inference

    def call_batch(self, data: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        images = []
        img_masks = []

        for key in self.present_img_keys:
            if key not in data:
                raise ValueError(
                    f"{key} not found in data. Please check the present_img_keys in the config or the dataset."
                )

            img = data[key]
            if img.ndim != 4:
                raise ValueError(f"(B,C,H,W) expected, but got {img.shape}")

            if self.resize_imgs_with_padding is not None:
                original_height, original_width = img.shape[2:]
                target_height, target_width = self.resize_imgs_with_padding
                if original_height != target_height or original_width != target_width:
                    ratio = max(original_width / target_width, original_height / target_height)
                    resized_height = int(original_height / ratio)
                    resized_width = int(original_width / ratio)
                    img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
                    pad_height = max(0, int(target_height - resized_height))
                    pad_width = max(0, int(target_width - resized_width))
                    pad_top = pad_height // 2
                    pad_bottom = pad_height - pad_top
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            if self.enable_image_aug:
                imgs = []
                for sample in img:
                    if "wrist" not in key:
                        sample = self.pose_transform(sample)
                    sample = self.color_jitter_transform(sample)
                    imgs.append(sample)
                img = torch.stack(imgs, dim=0)

            img = img / 255.0 * 2.0 - 1.0  # pi05 libero
            images.append(img)
            img_masks.append(torch.ones((img.shape[0],), dtype=torch.bool, device=img.device))

        return images, img_masks


class PromptTokenizerTransform:
    def __init__(self, max_length: int, discrete_state_input: bool = False) -> None:
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.tokenizer_max_length = max_length
        self.discrete_state_input = discrete_state_input

    def __call__(self, data: dict[str, Any], tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the text input.

        Args:
            data: Dict containing 'task' string and optionally 'observation.state' tensor to infer device.

        Returns:
            A tuple of (lang_tokens, lang_masks), both as torch tensors on the inferred device.
        """
        task = data["task"].strip().replace("_", " ").replace("\n", " ")

        # Infer device from observation.state if available
        device = data["observation.state"].device if "observation.state" in data else torch.device("cpu")

        if self.discrete_state_input:
            assert "observation.state" in data, "discrete_state_input is True, but observation.state is not found."
            discretized_state = (
                torch.bucketize(data["observation.state"], torch.linspace(-1, 1, 256 + 1, device=device)[:-1]) - 1
            )
            state_values = " ".join([str(int(x)) for x in discretized_state.tolist()])
            task = f"Task: {task}, State: {state_values};\nAction: "
        else:
            # PaliGemma prompt has to end with a new line in Pi0
            task = f"{task}\n"

        tokenized_prompt = tokenizer(
            task,
            padding="max_length",
            padding_side="right",
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"][0].to(dtype=torch.int32, device=device)
        lang_masks = tokenized_prompt["attention_mask"][0].to(dtype=torch.bool, device=device)

        return lang_tokens, lang_masks

    # VeRL: Batch Inference

    def call_batch(self, data: dict[str, Any], tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
        task = data["task"]
        if hasattr(task, "tolist") and not isinstance(task, str):
            tasks = task.tolist()
        else:
            tasks = list(task)
        tasks = [str(t).strip().replace("_", " ").replace("\n", " ") for t in tasks]

        device = data["observation.state"].device if "observation.state" in data else torch.device("cpu")

        if self.discrete_state_input:
            assert "observation.state" in data, "discrete_state_input is True, but observation.state is not found."
            state = data["observation.state"]
            discretized_state = torch.bucketize(state, torch.linspace(-1, 1, 256 + 1, device=device)[:-1]) - 1
            state_values = [" ".join([str(int(x)) for x in row.tolist()]) for row in discretized_state]
            tasks = [
                f"Task: {task_item}, State: {state_value};\nAction: "
                for task_item, state_value in zip(tasks, state_values, strict=False)
            ]
        else:
            tasks = [f"{task_item}\n" for task_item in tasks]

        tokenized_prompt = tokenizer(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(dtype=torch.int32, device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(dtype=torch.bool, device=device)

        return lang_tokens, lang_masks
