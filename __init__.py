"""ComfyUI-LoRABlockSurgeon — measure and filter a LoRA per transformer block.

Read-only against the .safetensors on disk. Nothing is ever written or rewritten.
"""

import logging

logger = logging.getLogger("LoRABlockSurgeon")

if __package__:
    # Normal path: ComfyUI imports this directory as a package.
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    logger.info("[LoRABlockSurgeon] registered %d nodes: %s",
                len(NODE_CLASS_MAPPINGS), ", ".join(sorted(NODE_CLASS_MAPPINGS)))
else:
    # Imported as a top-level module with no parent package — which is what pytest
    # does when it treats this directory as the tests' parent. `comfy.sd` is not
    # importable there. Checking __package__ rather than catching ImportError keeps
    # a genuinely broken import inside nodes.py loud under ComfyUI.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
