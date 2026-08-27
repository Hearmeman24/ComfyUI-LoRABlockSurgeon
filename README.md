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
`mode` → `model`, `applied`. Drop-in for `LoraLoaderModelOnly` with a main-block
filter. `blocks` takes `31-35` or `0-2,31,35`. `mode` is `keep` or `drop`.

Four optional Boolean inputs, all defaulting to `true`, independently control
the module groups present in MiniMax H3 LoRAs:

- `include_main_attention`
- `include_main_mlp`
- `include_token_refiner_attention`
- `include_token_refiner_mlp`

The numeric `blocks` selection applies **only to main transformer blocks**.
Token-refiner paths such as `token_refiner.blocks.0` have their own namespace and
do not collide with main block `0`; their tensors bypass numeric block selection
and respond only to the token-refiner toggles. Existing saved workflows can omit
all four optional inputs and retain the default all-groups-enabled behavior.

Module matching recognizes common attention tokens (`attn`, `attn1`,
`cross_attn`, and similar) and feed-forward tokens (`mlp`, `ff`, `ffn`,
`feed_forward`). An unfamiliar module group remains enabled by the group toggles
rather than being silently removed; if it belongs to a main block, it still obeys
the numeric main-block selection.

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

Embedders, heads and final layers match no supported block namespace. They are
grouped as `unblocked` in the profile and are **always applied** by the filter,
in both modes. The profiler keeps main and token-refiner groups separate; main
block `0` is labelled `0`, while token-refiner block `0` is labelled
`token_refiner.0`.

## Tests

```bash
python -m pytest tests/ -q      # no network, no GPU, no ComfyUI import
```

The suite currently reports **56 tests plus 16 subtests**. The load-bearing norm
tests check the fast calculations against explicitly materialised products. The
filter regressions cover namespace collisions, each independent module toggle,
main-only block selection, backwards-compatible defaults, and source-dict
immutability. If the norm identities drift, every number printed here is wrong.
