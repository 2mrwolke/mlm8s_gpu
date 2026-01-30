# mlm8s_gpu

Small TensorFlow utilities for reproducible, high-throughput prototyping.

Currently included:
- **`StatelessRngDataset`**: a `tf.data.Dataset` factory for **finite**, **on-the-fly** batch generation using **TensorFlow stateless RNG**.

---

## Requirements

- TensorFlow **>= 2.12**
- Python **>= 3.9**

---

## What `StatelessRngDataset` does

It builds a dataset of **exactly `num_batches` elements**. Each element is one **batch** of size `batch_size`.

For batch id `b` (0-based), it computes:

- `indices`: `[b*B, b*B + 1, ..., b*B + (B-1)]` (shape `[B]`, dtype `int64`)
- `batch_seed`: stateless seed (shape `[2]`, dtype `int32`) derived from:
  - a resolved base seed (`base_seed`)
  - the batch id `b` (mixed as 64-bit into the seed via two fold-ins: low32 then high32)

Then it calls your generator:

```py
samples = batch_generator(indices, batch_seed)
```

---

## Enabled patterns

This class is a good fit for:

- **Synthetic training data**  
  Procedural features/labels generated on-the-fly with reproducibility from a single base seed.

- **Reproducible data augmentation**  
  Randomized transforms keyed by batch id, not by call order, so results are stable across reruns even with `num_parallel_calls`.

- **Simulation-in-the-loop**  
  TF-native simulators / rollouts per batch driven by deterministic per-batch seeds, making outputs traceable across experiments.

- **Deterministic evaluation / benchmarking**  
  Fixed-length “random” eval sets for apples-to-apples regression testing.

- **High-throughput input pipelines**  
  Turn up parallel mapping and prefetching without changing the generated random numbers (only order varies unless `deterministic_order=True`).

- **Compiled generators (`tf.function` / XLA)**  
  Your generator is wrapped in `tf.function` with an input signature and can be XLA compiled (`xla_compile_generator=True`).

- **Distributed-friendly generation (seed-explicit)**  
  Seeds are explicit tensors passed into the generator, which is the right basis for reproducibility under `tf.distribute`.  
  *(Note: shard/replica-aware folding isn’t built-in; it’s straightforward to add another fold-in key if needed.)*

- **Experiment provenance**  
  `base_seed=None` still supports full reproducibility by logging `resolved_seed_tuple()`.

---

## Install

### From source (recommended)

```bash
pip install -e .
```

### Non-editable install

```bash
pip install .
```

> Project name in `pyproject.toml`: `mlm8s-gpu`  
> Import name: `mlm8s_gpu`

---

## Minimal usage

```python
import tensorflow as tf
from mlm8s_gpu import StatelessRngDataset

def batch_generator(indices: tf.Tensor, seed: tf.Tensor) -> dict[str, tf.Tensor]:
    # indices: [B] int64, seed: [2] int32
    B = tf.shape(indices)[0]
    x = tf.random.stateless_uniform([B, 32], seed=seed, dtype=tf.float32)
    y = tf.cast(tf.reduce_sum(x, axis=-1) > 16.0, tf.int32)
    return {"x": x, "y": y, "idx": indices}

factory = StatelessRngDataset(
    batch_generator=batch_generator,
    batch_size=256,
    num_batches=1000,
    base_seed=(123, 456),             # int | (int,int) | None
    deterministic_order=True,         # stable output order
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch_buffer=tf.data.AUTOTUNE,
    xla_compile_generator=True,
)

ds = factory.as_dataset()

for batch in ds.take(1):
    print(batch["x"].shape, batch["y"].shape)
```

---

## Determinism contract

### What is deterministic

For a fixed `base_seed`, `batch_size`, `num_batches`, and a generator that:

1) uses **only stateless RNG** (e.g. `tf.random.stateless_uniform`) with the provided `seed`, and  
2) returns a **stable structure** (same keys/dtypes/ranks each call), and  
3) runs only deterministic kernels for its ops,

then **batch `b` always produces identical tensors** across reruns — regardless of `tf.data` parallelism.

### What can still vary

- **Iteration order** can vary when parallel mapping is enabled unless you set `deterministic_order=True`.
  - This affects only **the order batches are yielded**, not the values for a given batch id.

### What is not guaranteed

- If your generator uses **stateful RNG** (e.g. `tf.random.uniform`), reproducibility is not guaranteed.
- GPU / XLA / certain ops may be nondeterministic depending on your TensorFlow build and kernels used.

---

## Logging runs when `base_seed=None`

If `base_seed=None`, the factory picks a random 2x32-bit seed once. You can retrieve it and log it:

```python
factory = StatelessRngDataset(..., base_seed=None)
print("resolved seed:", factory.resolved_seed_tuple())
```

Re-run later using `base_seed=factory.resolved_seed_tuple()`.

---

## API

### Constructor fields (with defaults)

Required:
- `batch_generator: Callable[[indices, seed], Any]`
- `batch_size: int` (must be > 0)
- `num_batches: int` (must be >= 0)
- `base_seed: int | (int, int) | None`

Optional:
- `deterministic_order: bool = False`
- `num_parallel_calls = tf.data.AUTOTUNE`
- `prefetch_buffer = tf.data.AUTOTUNE`
- `xla_compile_generator: bool = True`
- `generator_device: str | None = None`
- `threading_private_threadpool_size: int | None = None`
- `threading_max_intra_op_parallelism: int | None = None`
- `assert_cardinality: bool = True`

### Methods

- `as_dataset() -> tf.data.Dataset`
- `resolved_seed_tuple() -> tuple[int, int]`  *(useful when `base_seed=None`)*
- `base_seed_tensor() -> tf.Tensor`  *(shape `[2]`, dtype `int32`)*
- `num_examples -> int`  *(equals `batch_size * num_batches`)*

