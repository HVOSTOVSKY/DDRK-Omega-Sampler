"""Test bootstrap: run the sampler against a real ComfyUI checkout, on CPU.

The sampler imports comfy.* at module level, so every test needs ComfyUI; see
comfy_env.py for where it is looked up. Without one the suite is skipped with
a message saying how to point it at one. ComfyUI is forced onto the CPU so
results are deterministic and comparable between machines.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_env import bootstrap  # noqa: E402

COMFY = bootstrap()


def pytest_collection_modifyitems(config, items):
    if COMFY is not None:
        return
    skip = pytest.mark.skip(reason="ComfyUI not found: set COMFYUI_PATH to a "
                                   "ComfyUI checkout to run these tests")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def S():
    """The sampler module."""
    import ddrk_omega.sampler as sampler
    return sampler
