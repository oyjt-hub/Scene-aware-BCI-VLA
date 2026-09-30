from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._rtc_supported = bool(not is_pytorch and getattr(model, "rtc_supported", False))

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Keep RTC metadata out of the observation transforms and model Observation.
        obs = dict(obs)
        rtc_request = obs.pop("rtc", None)
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if rtc_request is not None:
            if not self._rtc_supported:
                raise ValueError("This policy does not support RTC. Use a JAX Pi0/Pi0.5 checkpoint.")
            if rtc_request.get("version") != 1 or rtc_request.get("mode") != "realtime":
                raise ValueError("Unsupported RTC request version or mode.")
            schedule = rtc_request.get("prefix_attention_schedule", "exp")
            if schedule != "exp":
                raise ValueError(f"Unsupported RTC prefix schedule: {schedule!r}; this server supports 'exp'.")
            action_horizon = int(self._model.action_horizon)
            model_action_dim = int(self._model.action_dim)
            prev_actions = np.asarray(rtc_request["prev_action_chunk"], dtype=np.float32)
            if (
                prev_actions.ndim != 2
                or prev_actions.shape[0] != action_horizon
                or not 0 < prev_actions.shape[1] <= model_action_dim
            ):
                raise ValueError(
                    "RTC prev_action_chunk must contain one external action chunk with "
                    f"{action_horizon} steps and at most {model_action_dim} dimensions; "
                    f"got {prev_actions.shape}."
                )
            if not np.isfinite(prev_actions).all():
                raise ValueError("RTC prev_action_chunk contains non-finite values.")
            # The client sends actions in the same external coordinates it receives.
            # Reapply the policy input transforms to bring that prefix back to the
            # model's normalized action space. AlohaOutputs, for example, returns
            # 14 dimensions while the model pads its internal action space to 32.
            rtc_obs = dict(obs)
            rtc_obs["actions"] = prev_actions.copy()
            rtc_inputs = self._input_transform(rtc_obs)
            normalized_prev_actions = np.asarray(rtc_inputs["actions"], dtype=np.float32)
            expected_model_shape = (action_horizon, model_action_dim)
            if normalized_prev_actions.shape != expected_model_shape:
                raise ValueError(
                    f"Transformed RTC prefix must have shape {expected_model_shape}, "
                    f"got {normalized_prev_actions.shape}."
                )
            if not np.isfinite(normalized_prev_actions).all():
                raise ValueError("Transformed RTC prefix contains non-finite values.")
            max_guidance_weight = float(rtc_request.get("max_guidance_weight", 10.0))
            if not np.isfinite(max_guidance_weight) or max_guidance_weight <= 0:
                raise ValueError("RTC max_guidance_weight must be finite and positive.")
            sample_kwargs.update(
                rtc_prev_action_chunk=jnp.asarray(normalized_prev_actions[None, ...]),
                rtc_inference_delay=jnp.asarray(
                    np.clip(int(rtc_request["inference_delay"]), 0, action_horizon), dtype=jnp.int32
                ),
                rtc_prefix_attention_horizon=jnp.asarray(
                    np.clip(int(rtc_request["prefix_attention_horizon"]), 0, action_horizon), dtype=jnp.int32
                ),
                rtc_max_guidance_weight=jnp.asarray(max_guidance_weight, dtype=jnp.float32),
            )
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata)
        if hasattr(self._model, "action_horizon"):
            metadata.setdefault("action_horizon", int(self._model.action_horizon))
        if hasattr(self._model, "action_dim"):
            metadata.setdefault("action_dim", int(self._model.action_dim))
        if self._rtc_supported:
            metadata["openpi_rtc"] = {
                "version": 1,
                "modes": ["realtime"],
                "prefix_attention_schedule": "exp",
            }
        return metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
