"""Load block_surgeon.py by explicit file path.

ComfyUI requires an __init__.py at the pack root, and that __init__ imports
comfy.sd -- which does not exist outside a ComfyUI tree. Any import mechanism
that infers a parent package from the directory layout ends up importing that
__init__ and dying. Loading the one module under test by path sidesteps the whole
question, and is honest about the boundary: these tests cover the pure-math core
and never touch ComfyUI.
"""

import importlib.util
import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("block_surgeon", _root / "block_surgeon.py")
_mod = importlib.util.module_from_spec(_spec)
# Registered BEFORE exec_module: @dataclass resolves sys.modules[cls.__module__]
# while the class body executes, and gets None if the module is not there yet.
sys.modules["block_surgeon"] = _mod
_spec.loader.exec_module(_mod)
