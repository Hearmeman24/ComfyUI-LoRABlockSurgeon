# Independently control MiniMax H3 LoRA module groups

**Type:** `feature/app` · **Full spec:** [`spec.claude.md`](./spec.claude.md)

## ✅ What you'll see when this is done

V9 users can independently enable or disable main attention, main MLP, token-refiner attention, and token-refiner MLP tensors. The existing block range controls only main transformer blocks, while saved workflows that omit the new optional inputs continue with every module group enabled.

## 🎲 Riding on these assumptions

- **Tests may use synthetic tensors with the exact inspected V9 key forms** — if production checkpoints use additional, materially different token-refiner paths, those paths will remain safely enabled but will need a later fixture-backed classifier extension. (couldn't confirm: the repository contains no real V9 checkpoint fixture)

## 🪤 Gotchas

- Token-refiner keys also contain `blocks.N`; token-refiner matching must happen before the generic main-block match.
- Unknown module groups must remain enabled so an unfamiliar LoRA layout is not silently pruned.
- A main block can be selected by the range while one of its module groups is disabled, so the report must not describe group-level drops as whole-block drops.

## Done when

- [ ] V9 users can independently toggle the four requested module groups, and block ranges affect only main transformer blocks; omitted toggles default to the current all-enabled behavior.
- [ ] Main block `0` and token-refiner block `0` are classified, filtered, and profiled independently.
- [ ] V8 MLP-only checkpoints and existing Krea/WAN/LTX main-block layouts retain their existing behavior.
- [ ] Filtering remains read-only and does not mutate the source state dict.
- [ ] The full test suite and configured static checks pass.

## The plan

1. Introduce a namespace-aware tensor location classifier while retaining the current generic main-block layouts.
2. Make profiling and block discovery distinguish main blocks, token-refiner blocks, and unblocked tensors.
3. Add the four optional Boolean node inputs and combine them with main-only block selection.
4. Add exact key-shape, filtering, compatibility, reporting, and immutability regressions; then update the README contract.
