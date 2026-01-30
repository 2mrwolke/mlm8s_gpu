# mlm8s_gpu
GPU-Accelerated Prototyping

## StatelessRngDataset

`StatelessRngDataset` is a small `tf.data` dataset *factory* for building **finite**, **on-the-fly generated** datasets using **TensorFlow stateless RNG**.

Each dataset element is a *batch*. For batch `b`, the class derives a per-batch stateless seed from:
- a **base seed** (`base_seed`), and
- the **batch id** (`b`),

then calls your generator as:

```python
samples = batch_generator(batch_indices, batch_seed)
```

This makes randomness **reproducible and traceable**: for a fixed `(base_seed, batch_id)` mapping and fixed shapes, the generated tensors are identical across reruns and robust to `tf.data` parallelism. If you allow parallel execution, **only the emission order may vary** unless you enable deterministic ordering.

## Requirements

- TensorFlow **>= 2.12**

### What it guarantees

- **Deterministic per-batch RNG** (stateless): the same `(base_seed, batch_id)` produces the same `batch_seed`, and therefore the same random tensors (given the same generator logic and shapes).
- **Fixed dataset length**: `num_batches` is authoritative; optional cardinality assertion is supported.
- **Parallelism-safe randomness**: increasing `num_parallel_calls` / `prefetch` won’t change the random numbers, because seeds do not depend on call order.

### What it does *not* guarantee (important)

- If your generator uses **stateful RNG** (e.g. `tf.random.uniform`), determinism is not guaranteed. Use `tf.random.stateless_*`.
- If you run on GPU with non-deterministic kernels in your generator, you may still see nondeterminism unrelated to RNG.
- With `deterministic_order=False` (default), **iteration order can vary** under parallel mapping; values per batch id are still deterministic.

---

## Minimal usage

```python
import tensorflow as tf
from mlm8s_gpu import StatelessRngDataset

def batch_generator(indices: tf.Tensor, seed: tf.Tensor) -> dict[str, tf.Tensor]:
    # indices: [B] int64, seed: [2] int32
    B = tf.shape(indices)[0]
    x = tf.random.stateless_uniform([B, 32], seed=seed, dtype=tf.float32)
    y = tf.cast(tf.reduce_sum(x, axis=-1) > 16.0, tf.int32)  # example label
    return {"x": x, "y": y, "idx": indices}

ds = StatelessRngDataset(
    batch_generator=batch_generator,
    batch_size=256,
    num_batches=1000,
    base_seed=(123, 456),          # or int, or None
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch_buffer=tf.data.AUTOTUNE,
    deterministic_order=True,      # set True for stable iteration order
    xla_compile_generator=True,
).as_dataset()

for batch in ds.take(1):
    print(batch["x"].shape, batch["y"].shape)
```

### Logging “random-by-default” runs

If you pass `base_seed=None`, a cryptographically strong random seed is chosen. You can log it for full reproducibility:

```python
factory = StatelessRngDataset(..., base_seed=None)
print("resolved seed:", factory.resolved_seed_tuple())
ds = factory.as_dataset()
```

Re-run later with `base_seed=factory.resolved_seed_tuple()`.

---

## Why this exists

TensorFlow’s **stateless** RNG APIs are deterministic, but it’s easy to accidentally reintroduce nondeterminism via:
- `tf.data` parallel execution (call order changes),
- hidden RNG state in generators,
- per-step “randomness” that’s hard to trace back.

`StatelessRngDataset` forces the randomness boundary to be explicit: **seed in, tensors out**.

---

## Enabled patterns

This class is a good fit for:

- **Synthetic training data**  
  Procedural features/labels generated on-the-fly with reproducibility from a single base seed.

- **Reproducible data augmentation**  
  Randomized transforms keyed by **batch id**, not by call order, so results are stable across reruns even with `num_parallel_calls`.

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

## API quick reference

Constructor arguments (high-level):

- `batch_generator(indices, seed) -> nested structure of tensors`  
  Must be traceable under `tf.function` and should use **stateless RNG** ops.
- `batch_size`, `num_batches` define dataset shape/length.
- `base_seed`: `int | (int,int) | None`  
  If `None`, a random seed is generated and can be retrieved via `resolved_seed_tuple()`.
- `deterministic_order`: if `True`, `tf.data` will preserve deterministic ordering (slower sometimes).
- `num_parallel_calls`, `prefetch_buffer`: performance tuning knobs.
- `xla_compile_generator`: XLA compile generator for speed if beneficial.
