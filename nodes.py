"""ComfyUI nodes. Mirrors nodes.py:709-768 (LoraLoader / LoraLoaderModelOnly).

Neither node writes to disk. The .safetensors is opened read-only via
comfy.utils.load_torch_file and the block selection is applied to an in-memory
copy of the state dict, so the original file is byte-identical afterwards.

Everything both nodes decide goes to the ComfyUI console under the
[LoRABlockSurgeon] prefix. That is deliberate: a block filter that quietly
applies the wrong set of blocks is indistinguishable from one that works, and the
render looks plausible either way.
"""

from __future__ import annotations

import logging
import os
import time

import comfy.sd
import comfy.utils
import folder_paths

from .block_surgeon import (
    _compact,
    blocks_present,
    filter_state_dict,
    format_report,
    parse_block_spec,
)

CATEGORY = "LoRA Block Surgeon"
logger = logging.getLogger("LoRABlockSurgeon")

_TAG = "[LoRABlockSurgeon]"


class _CachedLoad:
    """One-file cache, same shape as LoraLoader's (nodes.py:742-751), so flipping
    a block spec on a 300 MB LoRA does not re-read it from disk every run."""

    def __init__(self):
        self.loaded = None  # (path, state_dict, metadata)

    def _load(self, lora_name: str):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded is not None and self.loaded[0] == path:
            logger.info("%s cache hit: %s (%d tensors)", _TAG, lora_name, len(self.loaded[1]))
            return self.loaded[1], self.loaded[2]

        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
        except OSError:
            size_mb = float("nan")
        t0 = time.perf_counter()
        sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        dt = time.perf_counter() - t0

        blocks = blocks_present(sd)
        logger.info(
            "%s loaded %s | %.1f MB | %d tensors | blocks %s | %.2fs | READ-ONLY, never rewritten",
            _TAG, lora_name, size_mb, len(sd),
            f"{min(blocks)}..{max(blocks)}" if blocks else "(none found)", dt)
        if not blocks:
            logger.warning(
                "%s no numbered MAIN blocks matched in %s -- the numeric block filter will "
                "be a no-op. Recognized token-refiner groups can still be controlled by their "
                "toggles; inspect the namespace-qualified profiler output before applying.",
                _TAG, lora_name)

        self.loaded = (path, sd, meta)
        return sd, meta


class LoRABlockProfiler(_CachedLoad):
    """Read-only. Prints where a LoRA actually stores its learned change."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),
                              {"tooltip": "Opened read-only. Never modified."}),
                "sort_by": (["block", "norm"],
                            {"tooltip": "'block' reads as a profile across depth; "
                                        "'norm' ranks the heaviest blocks first."}),
            },
            "optional": {
                "top_n": ("INT", {"default": 0, "min": 0, "max": 512,
                                  "tooltip": "0 shows every block."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "profile"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Frobenius norm of the EFFECTIVE weight delta per transformer block. Measures "
        "up@down (or the Kronecker product for LoKr), not the stored factors -- LoRA has "
        "a free scale, so factor norms alone are meaningless. Read-only; writes nothing."
    )

    def profile(self, lora_name, sort_by, top_n=0):
        sd, _ = self._load(lora_name)
        t0 = time.perf_counter()
        body = format_report(sd, sort_by=sort_by, top=top_n)
        dt = time.perf_counter() - t0
        report = f"{lora_name}\n{body}"
        # One multi-line record rather than a line per block: the console interleaves
        # output from other nodes, and a split profile is unreadable.
        logger.info("%s profile (%.2fs)\n%s", _TAG, dt, report)
        return {"ui": {"text": [report]}, "result": (report,)}


class LoRABlockFilter(_CachedLoad):
    """Apply only some of a LoRA's blocks. The file on disk is untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),
                              {"tooltip": "Opened read-only. Never modified."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0,
                                             "step": 0.01}),
                "blocks": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "e.g. 31-35  or  0-2,31,35. "
                                                 "Empty + keep mode applies no main blocks."}),
                "mode": (["keep", "drop"],
                         {"tooltip": "keep = apply only these main transformer blocks. "
                                     "drop = apply every main transformer block except these. "
                                     "Token-refiner blocks use the optional group toggles."}),
            },
            "optional": {
                "include_main_attention": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Apply attention tensors in selected main transformer blocks."}),
                "include_main_mlp": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Apply MLP/feed-forward tensors in selected main blocks."}),
                "include_token_refiner_attention": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Apply token-refiner attention tensors. Main block ranges do not "
                                "control token-refiner blocks."}),
                "include_token_refiner_mlp": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Apply token-refiner MLP tensors. Main block ranges do not control "
                                "token-refiner blocks."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "applied")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "LoraLoaderModelOnly with main-block filtering plus independent main-attention, "
        "main-MLP, token-refiner-attention, and token-refiner-MLP controls. Tensors in "
        "unknown groups and tensors belonging to no supported block namespace are retained "
        "by default. Nothing is written to disk."
    )

    def apply(
        self,
        model,
        lora_name,
        strength_model,
        blocks,
        mode,
        include_main_attention=True,
        include_main_mlp=True,
        include_token_refiner_attention=True,
        include_token_refiner_mlp=True,
    ):
        if strength_model == 0:
            logger.info("%s strength_model=0 -> %s not applied, model passed through unchanged",
                        _TAG, lora_name)
            return (model, "strength_model=0, LoRA not applied")

        sd, meta = self._load(lora_name)

        try:
            selected = parse_block_spec(blocks)
        except ValueError as e:
            logger.error("%s could not parse blocks=%r: %s", _TAG, blocks, e)
            raise

        present = blocks_present(sd)
        logger.info("%s spec %r (mode=%s) parsed to %d blocks: %s",
                    _TAG, blocks, mode, len(selected), _compact(sorted(selected)))

        # A spec naming blocks this file does not have is almost always a typo or a
        # spec copied from a different base model. Silently ignoring it is how you
        # end up rendering with the full LoRA and believing you pruned it.
        phantom = sorted(selected - present)
        if phantom:
            logger.warning(
                "%s blocks %s are NOT in %s (it has %s). They are ignored -- in 'keep' mode "
                "that means fewer blocks applied than you asked for, in 'drop' mode it means "
                "nothing was dropped for them. Check the spec against the profiler.",
                _TAG, _compact(phantom), lora_name,
                f"{min(present)}..{max(present)}" if present else "(none)")

        if mode == "keep" and not selected:
            logger.warning(
                "%s blocks is empty in 'keep' mode: NO main transformer block will be applied. "
                "Token-refiner and unblocked tensors remain governed by their separate/default "
                "rules. Set a spec like 31-35, or switch mode to 'drop'.", _TAG)

        filtered, report = filter_state_dict(
            sd,
            selected,
            mode,
            include_main_attention=include_main_attention,
            include_main_mlp=include_main_mlp,
            include_token_refiner_attention=include_token_refiner_attention,
            include_token_refiner_mlp=include_token_refiner_mlp,
        )
        selected_main = sorted(report["selected_main_blocks"])
        excluded_main = sorted(report["excluded_main_blocks"])
        group_tensors = report["group_tensors"]
        controlled_groups = (
            ("main attention", "main_attention", include_main_attention),
            ("main MLP", "main_mlp", include_main_mlp),
            ("token-refiner attention", "token_refiner_attention",
             include_token_refiner_attention),
            ("token-refiner MLP", "token_refiner_mlp", include_token_refiner_mlp),
        )
        group_lines = []
        for label, key, enabled in controlled_groups:
            counts = group_tensors[key]
            group_lines.append(
                f"  {label}: {'enabled' if enabled else 'disabled'} | "
                f"{counts['kept']} kept, {counts['dropped']} dropped")
        other_kept = sum(group_tensors[key]["kept"] for key in (
            "main_unknown", "token_refiner_unknown"))
        other_dropped = sum(group_tensors[key]["dropped"] for key in (
            "main_unknown", "token_refiner_unknown"))
        group_summary = "\n".join(group_lines)
        applied = (f"{lora_name} @ {strength_model} | mode={mode}\n"
                   f"selected main blocks:  {_compact(selected_main)}\n"
                   f"excluded main blocks:  {_compact(excluded_main)}\n"
                   f"module-group tensors:\n{group_summary}\n"
                   f"  other classified/unknown: {other_kept} kept, "
                   f"{other_dropped} dropped by main-block selection\n"
                   f"always-applied unblocked tensors: {report['unblocked_tensors']}\n"
                   f"tensors passed to the patcher: {len(filtered)} of {len(sd)}")
        logger.info("%s applying\n%s", _TAG, applied)

        t0 = time.perf_counter()
        patched, _ = comfy.sd.load_lora_for_models(
            model, None, filtered, strength_model, 0, lora_metadata=meta)
        logger.info(
            "%s patched model in %.2fs (%d/%d tensors, %d main blocks selected, %d excluded)",
            _TAG, time.perf_counter() - t0, len(filtered), len(sd),
            len(selected_main), len(excluded_main))
        # comfy.sd.load_lora_for_models logs "NOT LOADED <key>" for anything that did
        # not map onto the model (comfy/sd.py:129-131). With a filter in play those
        # warnings are the signal that the key naming, not the block spec, is wrong.
        return (patched, applied)


NODE_CLASS_MAPPINGS = {
    "LoRABlockProfiler": LoRABlockProfiler,
    "LoRABlockFilter": LoRABlockFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRABlockProfiler": "LoRA Block Profiler",
    "LoRABlockFilter": "LoRA Block Filter (Apply)",
}
