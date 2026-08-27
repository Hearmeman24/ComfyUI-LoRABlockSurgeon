"""Per-block effective-delta measurement and in-memory block filtering for LoRA files.

WHY THIS EXISTS
---------------
A LoRA is not one adapter. It is one small adapter per targeted layer, and a
transformer has dozens of layers. The concept you trained usually lives in a
handful of blocks; the rest carry small, diffuse changes that add up to a global
style shift -- colour saturation, contrast, lighting -- because many tiny nudges
in the same direction compound through the network. Measuring per-block
magnitude lets you see that split and suppress the half you did not want.

WHAT IS MEASURED
----------------
The Frobenius norm of the EFFECTIVE weight delta -- the tensor actually added to
the base model's weight -- not the norms of the stored factors. That distinction
is load-bearing: LoRA has a free scale (multiply `up` by 10, divide `down` by 10,
identical delta), so the individual factor norms mean nothing and only their
product is well defined.

NOTHING IS EVER WRITTEN. Filtering happens on an in-memory state dict; the
.safetensors on disk is opened read-only and never modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

import torch

# Token-refiner paths also contain `blocks.N`, so this MUST be checked before the
# generic main-block expression below. Both dot- and underscore-separated paths
# occur in converted checkpoints.
_TOKEN_REFINER_BLOCK_RE = re.compile(
    r"(?:^|[._])token[._]refiner[._]blocks?[._](\d+)(?=[._]|$)")

# Matches `blocks.31.`, `transformer_blocks.31.`, `single_blocks.31.`,
# `double_blocks.31.` and the `_31_` variants Kohya-converted files use.
# Verified against three real reference files: Krea2 `diffusion_model.blocks.N`
# (28 blocks), WAN2.2 `diffusion_model.blocks.N` (40), LTX2.3
# `diffusion_model.transformer_blocks.N` (48).
_BLOCK_RE = re.compile(r"blocks?[._](\d+)[._]")

_ATTENTION_TOKEN_RE = re.compile(r"^(?:attn|attention)\d*$")
_MLP_TOKENS = frozenset({"mlp", "ff", "ffn", "feedforward"})

# Every up/down naming convention ComfyUI's LoRAAdapter.load recognises
# (comfy/weight_adapter/lora.py:159-188), as (up_suffix, down_suffix).
# AI-Toolkit-trained LoRA files use the `lora_B`/`lora_A` pair and carry NO alpha
# tensor (scale 1.0). Its LoKr checkpoints DO carry alpha. Handle both.
_LORA_PAIRS = (
    (".lora_up.weight", ".lora_down.weight"),
    ("_lora.up.weight", "_lora.down.weight"),
    (".lora_B.weight", ".lora_A.weight"),
    (".lora.up.weight", ".lora.down.weight"),
    (".lora_B", ".lora_A"),
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
    (".lora_B.default.weight", ".lora_A.default.weight"),
)

_SUPPORTED = ("lora", "lokr", "diff")


class UnsupportedAdapter(Exception):
    """Raised for a layer whose format we cannot measure exactly.

    Never swallowed into a zero. A silent zero would make a block look prunable
    when in fact it was simply not understood, which is the one failure mode that
    would make this tool actively dangerous.
    """


@dataclass
class LayerStat:
    prefix: str
    kind: str
    norm: float


@dataclass(frozen=True)
class TensorLocation:
    """Where a tensor lives and which optional module group controls it.

    `namespace` is `main`, `token_refiner`, or `unblocked`. `group` is
    `attention`, `mlp`, or `unknown`; unknown groups deliberately remain enabled.
    """

    namespace: str
    block: int | None
    group: str

    @property
    def label(self) -> str:
        if self.namespace == "unblocked":
            return "unblocked"
        if self.namespace == "main":
            return str(self.block)
        return f"{self.namespace}.{self.block}"


@dataclass
class BlockStat:
    """A namespace-qualified profiler group."""

    block: int | None
    norm: float
    layers: int
    kinds: set[str] = field(default_factory=set)
    namespace: str = "main"

    @property
    def label(self) -> str:
        if self.block is None:
            return "unblocked"
        return TensorLocation(self.namespace, self.block, "unknown").label


def _module_group(key: str) -> str:
    """Classify common attention/feed-forward path tokens, or return `unknown`.

    Matching is intentionally conservative. A false negative leaves a tensor
    enabled; a false positive could remove a tensor the operator never selected.
    """
    tokens = [token for token in re.split(r"[._]", key.lower()) if token]
    if any(_ATTENTION_TOKEN_RE.fullmatch(token) for token in tokens):
        return "attention"
    if any(token in _MLP_TOKENS for token in tokens):
        return "mlp"
    if any(a == "feed" and b == "forward" for a, b in pairwise(tokens)):
        return "mlp"
    return "unknown"


def tensor_location(key: str) -> TensorLocation:
    """Return a namespace-aware location for a tensor key.

    Token-refiner matching precedes the generic `blocks.N` matcher so its block
    indices never collide with main transformer blocks.
    """
    group = _module_group(key)
    token_refiner = _TOKEN_REFINER_BLOCK_RE.search(key)
    if token_refiner:
        return TensorLocation("token_refiner", int(token_refiner.group(1)), group)
    main = _BLOCK_RE.search(key)
    if main:
        return TensorLocation("main", int(main.group(1)), group)
    return TensorLocation("unblocked", None, group)


def block_index(key: str) -> int | None:
    """The numbered MAIN transformer block a tensor key belongs to, or None."""
    location = tensor_location(key)
    return location.block if location.namespace == "main" else None


def blocks_present(sd: dict) -> set[int]:
    """Every numbered MAIN block the file actually carries tensors for.

    Used to catch a spec naming blocks this LoRA does not have -- a typo, or a spec
    copied from a different base model. Ignoring that silently is how you render
    with the whole LoRA while believing you pruned it.
    """
    return {b for b in (block_index(k) for k in sd) if b is not None}


def parse_block_spec(spec: str) -> set[int]:
    """`"31-35"` / `"31,33"` / `"0-2, 31-35"` -> a set of block indices.

    An empty or whitespace-only spec is an empty set, which the caller must treat
    as "the user selected nothing" rather than as a wildcard.
    """
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            if not (lo.isdigit() and hi.isdigit()):
                raise ValueError(f"not a block range: {part!r}")
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                raise ValueError(f"reversed range: {part!r}")
            out.update(range(lo_i, hi_i + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f"not a block index: {part!r}")
    return out


def _fro_of_product(down: torch.Tensor, up: torch.Tensor) -> float:
    """||up @ down||_F without materialising the product.

    `down` is [rank, in] (possibly with trailing conv dims), `up` is [out, rank].

    ||BA||_F^2 = tr(A^T B^T B A) = tr(A A^T . B^T B) by the cyclic property of
    trace, and both A A^T and B^T B are rank x rank. For a rank-32 adapter on a
    6144-wide layer that is a pair of 32x32 matrices instead of a 6144x6144
    product -- exact, not an approximation.
    """
    d = down.reshape(down.shape[0], -1).to(torch.float32)  # [rank, in]
    u = up.reshape(up.shape[0], -1).to(torch.float32)      # [out, rank]
    if d.shape[0] != u.shape[1]:
        raise UnsupportedAdapter(
            f"inner dims disagree: down {tuple(down.shape)} vs up {tuple(up.shape)}")
    return float(torch.sqrt(torch.clamp(((d @ d.T) * (u.T @ u)).sum(), min=0.0)))


def _lora_norm(sd: dict, prefix: str, up_sfx: str, down_sfx: str, alpha: float | None) -> float:
    if prefix + ".lora_mid.weight" in sd:
        raise UnsupportedAdapter(f"{prefix}: lora_mid (CP-decomposed conv) not measured")
    down = sd[prefix + down_sfx]
    rank = down.shape[0]
    scale = 1.0 if alpha is None else (alpha / rank)
    return _fro_of_product(down, sd[prefix + up_sfx]) * abs(scale)


def _lokr_norm(sd: dict, prefix: str, alpha: float | None) -> float:
    """||kron(w1, w2)||_F = ||w1||_F * ||w2||_F -- exact, no Kronecker product built.

    Composition mirrors comfy/weight_adapter/lokr.py:59-86: w1 and w2 are each
    either stored whole or rebuilt from an (a @ b) pair scaled by alpha/rank, and
    the delta is torch.kron(w1, w2). Reshaping before kron does not change the
    norm, so the identity holds on the final delta.
    """

    def factor(tag: str) -> float:
        whole = sd.get(f"{prefix}.lokr_{tag}")
        if whole is not None:
            return float(torch.linalg.norm(whole.to(torch.float32)))
        a = sd.get(f"{prefix}.lokr_{tag}_a")
        b = sd.get(f"{prefix}.lokr_{tag}_b")
        if a is None or b is None:
            raise UnsupportedAdapter(f"{prefix}: lokr_{tag} is neither whole nor an a/b pair")
        t2 = sd.get(f"{prefix}.lokr_t2") if tag == "w2" else None
        rank = b.shape[0]
        scale = 1.0 if alpha is None else (alpha / rank)
        if t2 is not None:
            w = torch.einsum("i j k l, j r, i p -> p r k l", t2.to(torch.float32),
                             b.to(torch.float32), a.to(torch.float32))
            return float(torch.linalg.norm(w)) * abs(scale)
        return _fro_of_product(b, a) * abs(scale)

    return factor("w1") * factor("w2")


def _classify(sd: dict) -> dict[str, tuple[str, tuple]]:
    """prefix -> (kind, extra). One entry per adapted layer."""
    found: dict[str, tuple[str, tuple]] = {}
    for key in sd:
        for up_sfx, down_sfx in _LORA_PAIRS:
            if key.endswith(up_sfx):
                prefix = key[: -len(up_sfx)]
                if prefix + down_sfx in sd:
                    found[prefix] = ("lora", (up_sfx, down_sfx))
                break
        else:
            if ".lokr_" in key:
                prefix = key[: key.rindex(".lokr_")]
                found.setdefault(prefix, ("lokr", ()))
            elif key.endswith(".diff"):
                found[key[: -len(".diff")]] = ("diff", ())
            elif ".hada_" in key:
                prefix = key[: key.rindex(".hada_")]
                found.setdefault(prefix, ("loha", ()))
    return found


def layer_stats(sd: dict) -> tuple[list[LayerStat], list[str]]:
    """Effective-delta norm per adapted layer, plus the layers we could not measure."""
    stats: list[LayerStat] = []
    skipped: list[str] = []
    for prefix, (kind, extra) in sorted(_classify(sd).items()):
        alpha_t = sd.get(prefix + ".alpha")
        alpha = float(alpha_t.item()) if alpha_t is not None else None
        try:
            if kind == "lora":
                norm = _lora_norm(sd, prefix, extra[0], extra[1], alpha)
            elif kind == "lokr":
                norm = _lokr_norm(sd, prefix, alpha)
            elif kind == "diff":
                norm = float(torch.linalg.norm(sd[prefix + ".diff"].to(torch.float32)))
            else:
                raise UnsupportedAdapter(f"{prefix}: {kind} not supported (supported: {_SUPPORTED})")
        except UnsupportedAdapter as e:
            skipped.append(str(e))
            continue
        stats.append(LayerStat(prefix=prefix, kind=kind, norm=norm))
    return stats, skipped


def block_stats(sd: dict) -> tuple[list[BlockStat], list[str]]:
    """Per-block aggregate. Blocks are summed in QUADRATURE, not linearly.

    The per-layer deltas act on different weight matrices, so the honest
    aggregate of a block is the norm of its stacked deltas: sqrt(sum of squares).
    Summing the norms directly would over-weight blocks that simply have more
    adapted layers in them.
    """
    layers, skipped = layer_stats(sd)
    acc: dict[tuple[str, int | None], BlockStat] = {}
    for st in layers:
        location = tensor_location(st.prefix)
        key = (location.namespace, location.block)
        cur = acc.get(key)
        if cur is None:
            cur = acc[key] = BlockStat(
                block=location.block, norm=0.0, layers=0, namespace=location.namespace)
        cur.norm = float((cur.norm ** 2 + st.norm ** 2) ** 0.5)
        cur.layers += 1
        cur.kinds.add(st.kind)
    namespace_order = {"main": 0, "token_refiner": 1, "unblocked": 2}
    ordered = sorted(
        acc.values(),
        key=lambda s: (
            namespace_order.get(s.namespace, 1),
            s.block if s.block is not None else 0,
            s.namespace,
        ),
    )
    return ordered, skipped


_REPORT_GROUPS = (
    "main_attention",
    "main_mlp",
    "main_unknown",
    "token_refiner_attention",
    "token_refiner_mlp",
    "token_refiner_unknown",
    "unblocked",
)


def _report_group(location: TensorLocation) -> str:
    if location.namespace == "unblocked":
        return "unblocked"
    return f"{location.namespace}_{location.group}"


def filter_state_dict(
    sd: dict,
    selected: set[int],
    mode: str = "keep",
    *,
    include_main_attention: bool = True,
    include_main_mlp: bool = True,
    include_token_refiner_attention: bool = True,
    include_token_refiner_mlp: bool = True,
) -> tuple[dict, dict]:
    """A new dict holding only the tensors we want applied. `sd` is not mutated.

    `mode="keep"` retains selected MAIN blocks, `mode="drop"` removes them.
    Token-refiner tensors bypass numeric block selection and use their own group
    toggles. Unknown groups and unblocked tensors are always retained by group
    filtering; unknown main groups still obey main block selection.
    """
    if mode not in ("keep", "drop"):
        raise ValueError(f"mode must be 'keep' or 'drop', got {mode!r}")

    present = blocks_present(sd)
    if mode == "keep":
        selected_main = present & selected
        excluded_main = present - selected
    else:
        selected_main = present - selected
        excluded_main = present & selected

    group_enabled = {
        "main_attention": include_main_attention,
        "main_mlp": include_main_mlp,
        "token_refiner_attention": include_token_refiner_attention,
        "token_refiner_mlp": include_token_refiner_mlp,
    }
    group_tensors = {group: {"kept": 0, "dropped": 0} for group in _REPORT_GROUPS}
    report = {
        "selected_main_blocks": selected_main,
        "excluded_main_blocks": excluded_main,
        # Backwards-compatible aliases. These describe numeric block selection,
        # not whether every tensor group inside the block remained enabled.
        "kept_blocks": selected_main,
        "dropped_blocks": excluded_main,
        "group_tensors": group_tensors,
        "unblocked_tensors": 0,
    }
    out = {}
    for key, tensor in sd.items():
        location = tensor_location(key)
        report_group = _report_group(location)

        if location.namespace == "main":
            wanted = location.block in selected_main
            wanted = wanted and group_enabled.get(report_group, True)
        elif location.namespace == "token_refiner":
            wanted = group_enabled.get(report_group, True)
        else:
            wanted = True
            report["unblocked_tensors"] += 1

        if wanted:
            out[key] = tensor
            group_tensors[report_group]["kept"] += 1
        else:
            group_tensors[report_group]["dropped"] += 1
    return out, report


def format_report(sd: dict, sort_by: str = "block", top: int = 0) -> str:
    """The human-readable profile. Percentages are of total squared norm (energy)."""
    stats, skipped = block_stats(sd)
    if not stats:
        return ("No measurable adapter tensors found.\n"
                + ("\n".join(f"  skipped: {s}" for s in skipped) if skipped else
                   "  Nothing matched a known LoRA / LoKr / diff key pattern."))

    total_sq = sum(s.norm ** 2 for s in stats) or 1.0
    peak = max(s.norm for s in stats) or 1.0
    rows = sorted(stats, key=lambda s: -s.norm) if sort_by == "norm" else stats
    if top:
        rows = rows[:top]

    main_blocks = [s for s in stats if s.namespace == "main"]
    label_width = max(9, *(len(s.label) for s in stats))
    lines = [
        f"{len(stats)} groups | {sum(s.layers for s in stats)} adapted layers | "
        + f"formats: {','.join(sorted({k for s in stats for k in s.kinds}))}",
        f"main blocks present: {min((s.block for s in main_blocks), default='-')}"
        + f"..{max((s.block for s in main_blocks), default='-')}",
        "",
        f"{'block':>{label_width}} {'layers':>7} {'‖ΔW‖_F':>12} {'energy%':>9}  profile",
    ]
    for s in rows:
        bar = "#" * round(40 * s.norm / peak)
        lines.append(f"{s.label:>{label_width}} {s.layers:>7} {s.norm:>12.5f} "
                     f"{100 * s.norm ** 2 / total_sq:>8.2f}%  {bar}")

    profiled_blocks = [s for s in stats if s.namespace != "unblocked"]
    ranked = sorted(profiled_blocks, key=lambda s: -s.norm)
    if ranked:
        cum, keep = 0.0, []
        for s in ranked:
            if cum >= 0.90:
                break
            cum += s.norm ** 2 / total_sq
            keep.append(s)
        if all(s.namespace == "main" for s in keep):
            compact_keep = _compact(sorted(s.block for s in keep if s.block is not None))
        else:
            compact_keep = ",".join(s.label for s in keep)
        lines += ["", f"90% of the energy sits in {len(keep)} of "
                      + f"{len(profiled_blocks)} block groups: {compact_keep}"]
    if skipped:
        lines += ["", f"NOT MEASURED ({len(skipped)} layers) -- these are excluded from every "
                      + "number above, so treat their blocks as unknown, not as zero:"]
        lines += [f"  {s}" for s in skipped[:10]]
        if len(skipped) > 10:
            lines.append(f"  ... and {len(skipped) - 10} more")
    return "\n".join(lines)


def _compact(nums: list[int]) -> str:
    """[31,32,33,35] -> '31-33,35'"""
    if not nums:
        return "(none)"
    runs, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append((start, prev))
        start = prev = n
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
