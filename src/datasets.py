from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import secrets
import tensorflow as tf

BaseSeedLike = Union[int, Tuple[int, int], None]
_MASK32_U64 = tf.constant(0xFFFFFFFF, dtype=tf.uint64)


def _stateless_fold_in(seed: tf.Tensor, data: tf.Tensor) -> tf.Tensor:
    """
    Fold `data` into `seed` to form a new stateless RNG seed.

    Uses tf.random.fold_in, which aliases tf.random.experimental.stateless_fold_in.
    NOTE: Keep `data` int32 so the returned seed stays int32, as expected by TF stateless RNG APIs
    (shape [2], dtype int32).
    """
    return tf.random.fold_in(seed, data)  # aliases stateless_fold_in


@dataclass(frozen=True)
class StatelessRngDataset:
    """
    Finite tf.data.Dataset factory that calls a user-provided batch generator with:
        (batch_indices, per_batch_stateless_seed) -> samples

    Semantics:
      - `num_batches` is authoritative for dataset length.
      - Each batch has exactly `batch_size` indices: [b*B, ..., b*B + (B-1)].
      - Per-batch seed is derived by folding in the batch id (64-bit mixed into the base seed).

    Generator contract:
      - Must be traceable under `tf.function` and should be pure TensorFlow ops.
      - Must not use stateful RNG (e.g., `tf.random.uniform`); use stateless RNG ops with `seed`.
      - Must return a stable nested structure (dtypes/keys/ranks) across calls.

    How determinism works:
      - RNG determinism comes from stateless RNG: for a fixed `(base_seed, batch_id)` mapping, the
        batch seed is deterministic, so generated samples are deterministic (for fixed shapes).
      - Emission order determinism is separate: when parallel mapping is enabled, elements may be
        produced out-of-order unless `deterministic_order=True`. This does not affect RNG results,
        only the order in which batches are yielded.
    """

    batch_generator: Callable[[tf.Tensor, tf.Tensor], Any]
    batch_size: int
    num_batches: int
    base_seed: BaseSeedLike

    # Controls whether tf.data may emit elements out-of-order when parallelism is enabled.
    # This affects only output order, not RNG determinism (which is handled via stateless RNG).
    deterministic_order: bool = False
    num_parallel_calls: Any = tf.data.AUTOTUNE

    # Prefetch is independent of mapping parallelism. Defaults to AUTOTUNE.
    # Set to an int for a fixed buffer size, or to tf.data.AUTOTUNE.
    prefetch_buffer: Any = tf.data.AUTOTUNE

    xla_compile_generator: bool = True

    # Safer default for tf.distribute / multi-GPU: let TF/strategy place ops.
    generator_device: Optional[str] = None

    # Optional threading controls (avoid CPU oversubscription).
    threading_private_threadpool_size: Optional[int] = None
    threading_max_intra_op_parallelism: Optional[int] = None

    assert_cardinality: bool = True

    _base_seed: tf.Tensor = field(init=False, repr=False)
    _resolved_seed_py: Tuple[int, int] = field(init=False, repr=False)
    _batch_generator_tf: Callable[[tf.Tensor, tf.Tensor], Any] = field(init=False, repr=False)
    _batch_inputs_tf: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        bs = int(self.batch_size)
        nb = int(self.num_batches)
        if bs <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if nb < 0:
            raise ValueError(f"num_batches must be >= 0, got {self.num_batches}")

        base_seed, seed_py = self._normalize_base_seed(self.base_seed)
        object.__setattr__(self, "_base_seed", base_seed)
        object.__setattr__(self, "_resolved_seed_py", seed_py)

        # Batch-id -> (indices, seed) in-graph. Traced once via input_signature.
        batch_id_spec = tf.TensorSpec(shape=[], dtype=tf.int64)
        indices_spec = tf.TensorSpec(shape=[bs], dtype=tf.int64)
        seed_spec = tf.TensorSpec(shape=[2], dtype=tf.int32)

        def _make_batch_inputs(batch_id: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            indices = self._indices_for_batch(batch_id, bs)
            batch_seed = self._mix_key_into_seed(base_seed, tf.cast(batch_id, tf.int64))
            indices = tf.ensure_shape(indices, [bs])
            batch_seed = tf.ensure_shape(batch_seed, [2])
            return indices, batch_seed

        batch_inputs_tf = tf.function(
            _make_batch_inputs,
            jit_compile=False,  # tiny integer ops; keep compilation overhead minimal
            input_signature=[batch_id_spec],
        )
        object.__setattr__(self, "_batch_inputs_tf", batch_inputs_tf)

        gen_tf = tf.function(
            self.batch_generator,
            jit_compile=bool(self.xla_compile_generator),
            input_signature=[indices_spec, seed_spec],
        )
        object.__setattr__(self, "_batch_generator_tf", gen_tf)

    @property
    def num_examples(self) -> int:
        return int(self.batch_size) * int(self.num_batches)

    @staticmethod
    def _python_random_uint32() -> int:
        return secrets.randbits(32)

    @classmethod
    def _normalize_base_seed(cls, seed: BaseSeedLike) -> Tuple[tf.Tensor, Tuple[int, int]]:
        """
        Returns:
          - base_seed: [2] int32 Tensor suitable for TF stateless RNG APIs
          - resolved_seed_py: (s0, s1) python ints (loggable)
        """
        if seed is None:
            s0 = cls._python_random_uint32()
            s1 = cls._python_random_uint32()
            return tf.constant([s0, s1], dtype=tf.int32), (int(s0), int(s1))

        if isinstance(seed, tuple):
            if len(seed) != 2:
                raise ValueError(f"tuple seed must have length 2, got {len(seed)}")
            s0, s1 = (int(seed[0]) & 0xFFFFFFFF), (int(seed[1]) & 0xFFFFFFFF)
            return tf.constant([s0, s1], dtype=tf.int32), (int(s0), int(s1))

        s = int(seed)
        s0 = s & 0xFFFFFFFF
        s1 = (s ^ 0x9E3779B9) & 0xFFFFFFFF
        return tf.constant([s0, s1], dtype=tf.int32), (int(s0), int(s1))

    @staticmethod
    def _indices_for_batch(batch_id: tf.Tensor, batch_size: int) -> tf.Tensor:
        """batch_id: scalar int64 -> indices: [B] int64"""
        batch_id = tf.cast(batch_id, tf.int64)
        start = batch_id * tf.cast(batch_size, tf.int64)
        return start + tf.range(batch_size, dtype=tf.int64)

    @staticmethod
    def _mix_key_into_seed(base_seed: tf.Tensor, key64: tf.Tensor) -> tf.Tensor:
        """
        Low-collision mixing for 64-bit keys:
          fold_in(low32), then fold_in(high32).

        This avoids compressing the key to 32-bit first.
        """
        # Use logical (zero-fill) right shift by operating on uint64.
        # This makes the function well-defined even if `key64` is negative.
        key_u64 = tf.cast(key64, tf.uint64)

        lo_u64 = key_u64 & _MASK32_U64
        hi_u64 = tf.bitwise.right_shift(key_u64, 32)

        # Fold-in expects int32 data to keep the returned seed int32.
        # Masking ensures the cast to int32 is simply a bit-cast of the low 32 bits.
        lo = tf.cast(lo_u64, tf.int32)
        hi = tf.cast(hi_u64 & _MASK32_U64, tf.int32)

        seed1 = _stateless_fold_in(base_seed, lo)  # keep int32 output by passing int32 data
        seed1 = tf.ensure_shape(seed1, [2])
        seed2 = _stateless_fold_in(seed1, hi)
        return tf.ensure_shape(seed2, [2])

    def as_dataset(self) -> tf.data.Dataset:
        bs = int(self.batch_size)
        nb = int(self.num_batches)

        # num_batches-driven dataset: each element is one batch id.
        ds = tf.data.Dataset.range(nb, output_type=tf.int64)

        if bool(self.assert_cardinality):
            # TF version robustness:
            # - Newer TF: Dataset.assert_cardinality(count)
            # - Older TF: tf.data.experimental.assert_cardinality(count) via apply(...)
            assert_card = getattr(ds, "assert_cardinality", None)
            if callable(assert_card):
                ds = assert_card(nb)
            else:
                ds = ds.apply(tf.data.experimental.assert_cardinality(nb))

        batch_inputs_tf = self._batch_inputs_tf
        batch_generator_tf = self._batch_generator_tf
        device = self.generator_device

        # Stage 1: batch_id -> (indices, batch_seed) (pure graph seed computation)
        ds = ds.map(
            batch_inputs_tf,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=bool(self.deterministic_order),
        )

        # Stage 2: (indices, seed) -> samples (compiled generator)
        def call_generator(indices: tf.Tensor, batch_seed: tf.Tensor):
            if device is not None:
                # Explicit placement is opt-in; may be undesirable under tf.distribute.
                with tf.device(device):
                    return batch_generator_tf(indices, batch_seed)
            return batch_generator_tf(indices, batch_seed)

        ds = ds.map(
            call_generator,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=bool(self.deterministic_order),
        )

        opts = tf.data.Options()
        # Controls whether transformations are allowed to yield elements out-of-order when parallel.
        opts.deterministic = bool(self.deterministic_order)

        if self.threading_private_threadpool_size is not None:
            opts.threading.private_threadpool_size = int(self.threading_private_threadpool_size)
        if self.threading_max_intra_op_parallelism is not None:
            opts.threading.max_intra_op_parallelism = int(self.threading_max_intra_op_parallelism)

        ds = ds.with_options(opts)
        return ds.prefetch(self.prefetch_buffer)

    def base_seed_tensor(self) -> tf.Tensor:
        """Resolved base seed used by this instance (stable for the instance)."""
        return self._base_seed

    def resolved_seed_tuple(self) -> Tuple[int, int]:
        """
        Python (s0, s1) seed actually used by this instance.
        Useful for logging when `base_seed=None` was passed.
        """
        return self._resolved_seed_py


# Example batch_generator (single RNG call per batch, fully vectorized, stable structure)
def batch_generator(indices: tf.Tensor, seed: tf.Tensor) -> dict[str, tf.Tensor]:
    """
    indices: [B] int64
    seed:    [2] int32 (one stateless seed per batch)
    returns a stable dict structure.

    stateless_* ops are deterministic for fixed seed/shape.
    """
    B = tf.shape(indices)[0]
    x = tf.random.stateless_uniform(
        shape=[B, 32],
        seed=seed,
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )
    return {"x": x, "x_sqrt": tf.sqrt(x)}
