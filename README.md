# ComfyUI-LoRABlockSurgeon

Two nodes. Measure where a LoRA actually stores its learned change, then apply
only the blocks you want.

**Nothing is written to disk.** The `.safetensors` is opened read-only and the
block selection is applied to an in-memory copy of the state dict, so the file is
byte-identical after a run. There is no "save pruned copy" path by design.

## Nodes

**LoRA Block Profiler** — `lora_name`, `sort_by`, `top_n` → `report` (STRING).
Prints the Frobenius norm of the effective weight delta per transformer block,
its share of total energy, a bar profile, and how many blocks hold 90% of the
energy.

**LoRA Block Filter (Apply)** — `model`, `lora_name`, `strength_model`, `blocks`,
`mode` → `model`, `applied`. Drop-in for `LoraLoaderModelOnly` with a block
filter. `blocks` takes `31-35` or `0-2,31,35`. `mode` is `keep` or `drop`.

## What is measured, and why that specific quantity

The effective delta — the tensor actually added to the base weight — not the
norms of the stored factors. LoRA has a free scale: multiply `up` by 10 and
divide `down` by 10 and the delta is identical, so individual factor norms carry
no information and only their product does.

- **LoRA**: `‖(alpha/rank) · up @ down‖_F`, computed as
  `sqrt(sum((A Aᵀ) ⊙ (Bᵀ B)))` via the cyclic property of trace. Exact, and it
  never materialises the full delta — a rank-32 adapter on a 6144-wide layer is
  two 32×32 matrices instead of a 6144×6144 product.
- **LoKr**: `‖kron(w1, w2)‖_F = ‖w1‖_F · ‖w2‖_F`. Exact; no Kronecker product is
  built. Composition follows ComfyUI's own `weight_adapter/lokr.py`.
- **diff**: the norm of the stored tensor.
- **LoHa and anything else**: reported as NOT MEASURED and excluded from every
  number, never folded into a zero. A silent zero would make a block look
  prunable when it was merely not understood.

Per-block aggregation sums in **quadrature** (`sqrt(Σ nᵢ²)`), the norm of the
block's stacked deltas. A plain sum would over-rank blocks that simply contain
more adapted layers.

## Blocks that carry no index

Embedders, heads and final layers match no `blocks.N` pattern. They are grouped
as `unblocked` in the profile and are **always applied** by the filter, in both
modes.

## Tests

```bash
python -m pytest tests/ -q      # 39 tests, no network, no GPU, no ComfyUI import
```

39 tests. The load-bearing ones check the fast norms against explicitly materialised
products. If those drift, every number printed here is wrong.
