"""Build a MaxText train state and run one GRPO step.

MaxText's pre-train entry point does not return until pretraining finishes.
This stops after ``setup_train_loop`` and drives MaxText's own ``train_step``,
with the cross-entropy loss replaced by the clipped policy loss. The optimizer
update stays MaxText's.
"""

from __future__ import annotations

from typing import Any

from verl.workers.engine.maxtext.loss import clip_from_actor, clipped_policy_loss
from verl.workers.engine.maxtext.params import bind_maxtext, maxtext_argv, prepare_jax_env


class MaxTextSession:
    """One JAX process. It owns every visible device."""

    def __init__(self, engine_config, model_config):
        self._model_config = model_config
        prepare_jax_env(engine_config)
        bind_maxtext(engine_config)
        self._init_train_state(maxtext_argv(engine_config))
        self._clip = (0.2, 0.2, 3.0)
        self._install_loss(self._clip)
        self._compile()
        hf_path = engine_config.hf_weights_path
        if hf_path is None and engine_config.load_hf_weights:
            hf_path = getattr(model_config, "local_path", None) or getattr(model_config, "path", None)
        if hf_path:
            self.load_hf_weights(hf_path)

    def set_actor_config(self, actor_config) -> None:
        if actor_config is None:
            return
        clip = clip_from_actor(actor_config)
        if clip != self._clip:
            self._clip = clip
            self._install_loss(clip)
            if getattr(self, "_train_step", None) is not None:
                self._compile()

    def _mesh(self):
        import jax
        from flax.core.spmd import logical_axis_rules

        return jax.set_mesh(self._device_mesh), logical_axis_rules(self._config.logical_axis_rules)

    def train(self, batch: dict[str, Any]) -> dict[str, float]:
        import contextlib

        import jax

        mesh, rules = self._mesh()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mesh)
            stack.enter_context(rules)
            shard = self._data_sharding
            batch = {key: jax.device_put(value, shard) for key, value in batch.items()}
            self._state, metrics = self._train_step(self._state, batch)
        scalar = metrics["scalar"]
        # learning/grad_norm is measured after clipping, so it saturates at the
        # clipping threshold. The raw norm is the quantity veRL reports.
        grad_norm = scalar.get("learning/raw_grad_norm", scalar.get("learning/grad_norm"))
        if grad_norm is None:
            raise RuntimeError(
                "MaxText did not return learning/grad_norm. Leave optimizer_memory_host_offload disabled."
            )
        return {
            "loss": float(jax.device_get(scalar["learning/loss"])),
            "grad_norm": float(jax.device_get(grad_norm)),
        }

    def log_probs(self, batch: dict[str, Any]):
        return self.log_probs_and_entropy(batch)[0]

    def log_probs_and_entropy(self, batch: dict[str, Any]):
        import contextlib

        import jax
        import numpy as np

        mesh, rules = self._mesh()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mesh)
            stack.enter_context(rules)
            shard = self._data_sharding
            batch = {key: jax.device_put(value, shard) for key, value in batch.items()}
            values, entropy = self._logprob_step(self._state, batch)
        return np.asarray(jax.device_get(values)), np.asarray(jax.device_get(entropy))

    def stacked_weights(self) -> dict:
        """Parameter pytree as sharded device arrays. Nothing is copied to the host."""
        from flax import nnx

        graph, pure = self._model_parts()
        merged = nnx.merge(graph, pure)
        params = nnx.state(merged, nnx.Param)

        def leaf(value):
            return getattr(value, "value", value)

        import jax

        return jax.tree.map(leaf, params.to_pure_dict())

    def load_hf_weights(self, path: str) -> None:
        """Replace the initial parameters with a Hugging Face checkpoint.

        Optimizer state stays as initialised. The compiled steps are reused:
        the replacement arrays have the same shapes, dtypes and shardings.
        """
        import jax
        from flax import nnx

        from verl.workers.engine.maxtext.hf_import import HFReader, load_stacked_from_hf

        full = nnx.merge(self._graph, self._state)
        params = nnx.state(full.model, nnx.Param)
        like = jax.tree.map(lambda v: getattr(v, "value", v), params.to_pure_dict())
        loaded = load_stacked_from_hf(HFReader(path), like, lambda old, host: jax.device_put(host, old.sharding))
        nnx.replace_by_pure_dict(params, loaded)
        nnx.update(full.model, params)
        self._graph, self._state = nnx.split(full)

    @staticmethod
    def fetch_bf16(array):
        """One slice to a host numpy array. float32 slices are cast on the device first."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        if hasattr(array, "dtype") and array.dtype == jnp.float32:
            array = array.astype(jnp.bfloat16)
        return np.asarray(jax.device_get(array))

    @property
    def batch_shape(self) -> tuple[int, int]:
        return self._batch_shape

    def _init_train_state(self, argv: list[str]) -> None:
        from maxtext.trainers.pre_train import train as train_module
        from maxtext.utils import sharding, train_utils

        self._train_module = train_module
        # Grain reads absl flags when the synthetic pipeline starts. This process
        # never goes through absl.app.run, so mark them parsed.
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
        initialized = train_module.initialize(argv)
        config = initialized[0]
        recorder = initialized[1]
        setup = train_utils.setup_train_loop(config, recorder)
        (
            self._init_rng,
            _checkpoint_manager,
            self._state_shardings,
            _model,
            self._device_mesh,
            _learning_rate,
            _data_iterator,
            _data_loader,
            _rampup,
            _eval_iterator,
            state,
        ) = setup
        self._config = config
        from flax import nnx

        graph, pure = nnx.split(state)
        self._graph = graph
        self._state = pure
        self._data_sharding = sharding.get_input_data_sharding(config, self._device_mesh)
        # The step is compiled on the global batch. MaxText sets that to
        # num_devices * per_device_batch_size, then shards it. A data axis of
        # size 1 replicates the whole batch onto every expert-parallel device.
        length = int(config.max_target_length)
        global_batch = int(config.global_batch_size_to_load)
        self._batch_shape = (global_batch, length)

    def _install_loss(self, clip: tuple[float, float, float]) -> None:
        import jax.numpy as jnp

        from verl.workers.engine.maxtext.batch import next_token_logprobs

        low, high, clip_c = clip

        def forward_logits(model, config, data, is_train):
            # Use every row given. With gradient accumulation MaxText already hands
            # the loss one micro-batch, and scoring runs on the whole batch. The
            # eval micro-batch size is an unrelated setting and would drop rows.
            del config, is_train
            logits = model(
                decoder_input_tokens=data["inputs"],
                decoder_positions=data["inputs_position"],
                decoder_segment_ids=data["inputs_segmentation"],
                enable_dropout=False,
                decoder_target_tokens=data["targets"],
                decoder_target_mask=data["targets_segmentation"],
            )
            if isinstance(logits, tuple):
                logits = logits[0]
            return data, logits

        self._forward_logits = forward_logits

        def loss_fn(model, config, data, dropout_rng, params, sparsity_state=None, is_train=True):
            del dropout_rng, params, sparsity_state
            data, logits = forward_logits(model, config, data, is_train)
            token_log_probs = next_token_logprobs(logits, data["inputs"])
            if is_train:
                loss, _metrics = clipped_policy_loss(
                    jnp,
                    token_log_probs,
                    data["old_log_probs"],
                    data["advantages"],
                    data["response_mask"],
                    low,
                    high,
                    clip_c,
                )
            else:
                loss = jnp.zeros((), dtype=jnp.float32)
            weight = jnp.sum(data["response_mask"])
            # With gradient accumulation MaxText sums the micro-batch losses and
            # divides by the summed weights once, so the objective must be the
            # token sum. Scaling the token mean by its (constant) weight gives it.
            objective = loss * weight if getattr(config, "gradient_accumulation_steps", 1) > 1 and is_train else loss
            aux = {
                "intermediate_outputs": {},
                "xent_sum": loss * weight,
                "total_weights": weight,
                "moe_lb_loss": jnp.asarray(0.0, dtype=loss.dtype),
                "indexer_loss": jnp.asarray(0.0, dtype=loss.dtype),
                "mtp_loss": jnp.asarray(0.0, dtype=loss.dtype),
                "log_probs": token_log_probs,
            }
            return objective, aux

        self._loss_fn = loss_fn
        self._train_module.loss_fn = loss_fn

    def _compile(self) -> None:
        import jax
        from flax import nnx
        from maxtext.utils import train_utils

        self._install_loss(self._clip)
        graph, state = self._graph, self._state
        params_shardings, state_shardings = self._params_shardings()
        train_step, _eval = train_utils.jit_train_and_eval_step(
            self._config,
            graph,
            self._device_mesh,
            state,
            state_shardings,
            self._train_module.train_step,
            self._train_module.eval_step,
            params_shardings=params_shardings,
        )
        self._train_step = train_step

        def logprob_step(pure_state, batch):
            from verl.workers.engine.maxtext.batch import next_token_entropy, next_token_logprobs

            merged = nnx.merge(graph, pure_state)
            data, logits = self._forward_logits(merged.model, self._config, batch, False)
            return next_token_logprobs(logits, data["inputs"]), next_token_entropy(logits)

        self._logprob_step = jax.jit(
            logprob_step,
            in_shardings=(state_shardings, self._data_sharding),
        )
        del jax

    def _params_shardings(self):
        from maxtext.utils import sharding

        config = self._config
        if getattr(config, "enable_diloco", False):
            return self._state_shardings.params, self._state_shardings
        return sharding.maybe_update_params_sharding_with_opt(config, self._state_shardings)

    def _model_parts(self):
        from flax import nnx

        merged = nnx.merge(self._graph, self._state)
        graph, pure = nnx.split(merged.model)
        return graph, pure
