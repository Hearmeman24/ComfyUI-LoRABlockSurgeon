# Independently control MiniMax H3 LoRA module groups

- **Work type:** `feature/app`
- **Status:** `draft` → proceed under task-scoped authority; no material decision is unresolved
- **Review surface:** [`spec.human.md`](./spec.human.md)

## 1. Problem / Context

The filter currently offers only a numeric block selection and `keep`/`drop` mode at the node boundary (`nodes.py:118-134`). MiniMax H3 V9 needs four independently selectable module groups: main attention, main MLP, token-refiner attention, and token-refiner MLP. Existing workflows must remain valid when they omit the new controls.

## 2. Approach & why

- The current block matcher searches any `blocks.N`-shaped fragment (`block_surgeon.py:31-36`), and `block_index` returns only that integer (`block_surgeon.py:85-88`). Therefore a token-refiner key containing `token_refiner.blocks.0` cannot be distinguished from main block `0` by the current representation.
- Block discovery is derived directly from `block_index` (`block_surgeon.py:91-98`), filtering applies selection directly to that integer (`block_surgeon.py:249-272`), and profiling aggregates directly by it (`block_surgeon.py:227-246`). The representation must be corrected at the shared classifier rather than patched independently at each caller.
- Introduce a tensor location containing namespace, optional block index, and recognized module group. Token-refiner block matching precedes generic main-block matching. Unknown groups stay enabled, preserving unfamiliar files.
- Preserve `block_index` as a compatibility helper for main blocks only. Use the richer location in block discovery, profiling, filtering, and reporting.

## 3. Acceptance Criteria

- [ ] V9 users can independently enable/disable main attention, main MLP, token-refiner attention, and token-refiner MLP while numeric block selection affects only main transformer blocks; omitted optional inputs default all groups to enabled. → (ask: "boolean toggles for removing/keeping certain block groups like token refiner")
- [ ] Exact location tests distinguish `diffusion_model.blocks.0.attn.qkv_proj`, main MLP, `diffusion_model.token_refiner.blocks.0.attn.qkv_proj`, and token-refiner MLP. → (ask: "certain block groups like token refiner")
- [ ] `blocks_present` reports main blocks only, and main-block keep/drop never controls token-refiner tensors. → (ask: "token refiner")
- [ ] Each new Boolean independently drops only its intended classified tensor group; unknown and unblocked tensors remain enabled. → (ask: "removing/keeping certain block groups")
- [ ] The profiler labels main block `0` and token-refiner block `0` separately. → (ask: "certain block groups like token refiner")
- [ ] Default/all-true filtering is equivalent, source dictionaries remain unmodified, and V8 MLP-only plus existing generic Krea/WAN/LTX layouts remain green. → (ask: "extend the node")

## 4. Scope & Non-Goals

**In scope:** shared classification, profiling, in-memory filtering and reporting in `block_surgeon.py:31-319`; optional node inputs and application report in `nodes.py:115-199`; pure tests in `tests/test_block_surgeon.py:33-356`; user contract in `README.md:10-56`.

**Non-goals (explicitly NOT doing):** writing pruned checkpoints, changing adapter math, retraining a LoRA, changing node registration, packaging `node.zip`, or changing project metadata.

## 5. Key Decisions & Constraints

- **Decided:** Namespace is classified before block number, with token refiner matched before generic main blocks, because the existing regex otherwise captures the nested token-refiner `blocks.N` fragment (`block_surgeon.py:31-36`).
- **Decided:** Existing numeric block range semantics apply only to main blocks; token-refiner module toggles bypass numeric selection.
- **Decided:** The new inputs are optional and default `True`; the Python method and pure filtering function also default them to `True` so API workflows that omit them remain valid (`nodes.py:118-147`).
- **Constraint / must-not-break:** Filtering returns a new dict and never mutates or writes the loaded state dict (`block_surgeon.py:249-272`, `nodes.py:1-10`).
- **Constraint / must-not-break:** Unknown groups and tensors with no supported block namespace remain enabled, following the current unconditional retention of unblocked tensors (`block_surgeon.py:252-265`).
- **Mirror existing:** Preserve the generic `blocks`, `transformer_blocks`, `single_blocks`, and `double_blocks` dot/underscore layouts accepted by `_BLOCK_RE` (`block_surgeon.py:31-36`) and covered by compatibility tests (`tests/test_block_surgeon.py:33-52`).

## 6. Code Surface Map

- `block_surgeon.py:31-98` — block/location classification and main-block discovery.
- `block_surgeon.py:64-88` — profiler group representation and labels.
- `block_surgeon.py:227-319` — aggregation, in-memory filtering, filter report, and human-readable profile.
- `nodes.py:115-199` — ComfyUI filter node inputs, defaults, filter invocation, and applied summary.
- `tests/test_block_surgeon.py:33-356` — pure classifier, profiler, filter, compatibility, and immutability regressions.
- `README.md:10-56` — public node and testing contract.

## 7. Ultracode Dispatch Notes

**Build first (sequential — freezes interfaces before any parallelism):**
- Freeze this namespace/group contract in `spec.human.md` and `spec.claude.md`.

**Parallel slices (independent — one agent each). Each slice DECLARES the files/state it writes):**
- **Single serialized slice** — implement the shared classifier, profiler, filter, node inputs, tests, and README because these files depend on one evolving report/API contract. Writes: `block_surgeon.py`, `nodes.py`, `tests/test_block_surgeon.py`, `README.md`.

**⛓ Collision audit:** There is one implementation slice; no shared file or symbol is concurrently written.

**Each agent must:** implement its slice + write and green its own tests + self-verify against §3.

```yaml
dispatch:
  frozen:
    - docs/specs/2026-08-27-module-group-toggles/spec.human.md
    - docs/specs/2026-08-27-module-group-toggles/spec.claude.md
    - __init__.py
    - pyproject.toml
    - node.zip
  slices:
    - key: module_group_contract
      writes:
        - block_surgeon.py
        - nodes.py
        - tests/test_block_surgeon.py
        - README.md
  testRunner: "/Users/avivkaplan/.venvs/torch312/bin/python -m pytest tests/ -q"
```

## 8. Assumptions & Open Questions

- **ASSUMPTION:** Synthetic tensors using the exact inspected V9 key forms are sufficient for this repository's regressions. The repository has no real V9 checkpoint fixture to verify additional path variants. Impact if wrong: unrecognized production variants remain safely enabled but will not respond to a requested group toggle until fixture-backed classification is added.
