"""Locate a ComfyUI checkout and import it on the CPU (tests and bench).

Lookup order: $COMFYUI_PATH, then ../.. (the normal install,
ComfyUI/custom_nodes/DDRK-Omega-Sampler), then ../ComfyUI.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))


def find_comfy():
    candidates = [os.environ.get("COMFYUI_PATH"),
                  os.path.dirname(os.path.dirname(REPO)),
                  os.path.join(os.path.dirname(REPO), "ComfyUI")]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "comfy", "samplers.py")):
            return os.path.abspath(c)
    return None


def bootstrap():
    """Put ComfyUI and the repo on sys.path and force ComfyUI onto the CPU.

    Returns the ComfyUI path, or None if no checkout was found.
    """
    comfy = find_comfy()
    if comfy is None:
        return None
    for p in (comfy, REPO, TESTS):
        if p not in sys.path:
            sys.path.insert(0, p)
    argv = sys.argv
    sys.argv = [argv[0], "--cpu"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.model_management  # noqa: F401  (parses --cpu once)
    finally:
        sys.argv = argv
    return comfy
