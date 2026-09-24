"""Node registration and UI schema: what ComfyUI sees when it loads the pack."""

import inspect
import json
import os

import pytest

NODE_WIDGET_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN")


def _inputs(cls):
    spec = cls.INPUT_TYPES()
    out = {}
    for section in ("required", "optional"):
        for name, val in spec.get(section, {}).items():
            out[name] = (section, val)
    return out


def test_package_exports_every_node(S):
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "ddrk_pkg", os.path.join(root, "__init__.py"),
        submodule_search_locations=[root])
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["ddrk_pkg"] = mod          # what ComfyUI's loader does too
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("ddrk_pkg", None)
    assert set(mod.NODE_CLASS_MAPPINGS) == set(S.NODE_CLASS_MAPPINGS)
    assert set(mod.NODE_DISPLAY_NAME_MAPPINGS) == set(S.NODE_CLASS_MAPPINGS)
    assert len(S.NODE_CLASS_MAPPINGS) == 7


@pytest.mark.parametrize("key", ["DDRKFluxConditioning", "DDRKOmegaSchedulerNode",
                                 "DDRKOmegaSamplerNode", "DDRKOmegaUnifiedKSamplerNode",
                                 "DDRKOmegaLiteKSamplerNode", "DDRKOmegaSmartConfigNode",
                                 "DDRKOmegaAutoNode"])
def test_inputs_match_function_signature(S, key):
    cls = S.NODE_CLASS_MAPPINGS[key]
    fn = getattr(cls, cls.FUNCTION)
    params = inspect.signature(fn).parameters
    inputs = _inputs(cls)
    for name, (section, _) in inputs.items():
        assert name in params, f"{key}: input {name!r} is not a parameter of {cls.FUNCTION}"
    for name, p in params.items():
        if name == "self":
            continue
        if p.default is inspect.Parameter.empty:
            assert name in inputs and inputs[name][0] == "required", \
                f"{key}: {name!r} has no default but is not a required input"
        if name in inputs and inputs[name][0] == "optional":
            assert p.default is not inspect.Parameter.empty, \
                f"{key}: optional input {name!r} needs a default in {cls.FUNCTION}"


@pytest.mark.parametrize("key", ["DDRKFluxConditioning", "DDRKOmegaSchedulerNode",
                                 "DDRKOmegaSamplerNode", "DDRKOmegaUnifiedKSamplerNode",
                                 "DDRKOmegaLiteKSamplerNode", "DDRKOmegaAutoNode"])
def test_widget_defaults_are_in_range(S, key):
    cls = S.NODE_CLASS_MAPPINGS[key]
    fn = getattr(cls, cls.FUNCTION)
    params = inspect.signature(fn).parameters
    for name, (section, val) in _inputs(cls).items():
        kind, opts = val[0], (val[1] if len(val) > 1 else {})
        if isinstance(kind, list):
            assert opts.get("default", kind[0]) in kind, f"{key}.{name}"
            continue
        if kind in ("INT", "FLOAT") and "default" in opts:
            assert opts.get("min", -float("inf")) <= opts["default"] <= opts.get("max", float("inf")), \
                f"{key}.{name} default outside [min, max]"
        # Python-side default must match the widget default, or a workflow
        # that omits the optional input behaves differently from the UI.
        if section == "optional" and "default" in opts and name in params:
            assert params[name].default == opts["default"], f"{key}.{name}"


def test_scheduler_dropdowns_are_all_known(S):
    for key in ("DDRKOmegaSchedulerNode", "DDRKOmegaUnifiedKSamplerNode"):
        choices = _inputs(S.NODE_CLASS_MAPPINGS[key])["scheduler_type"][1][0]
        assert set(choices) <= S._KNOWN_SCHEDULERS
        assert "ddrk_anima" not in choices


def _widgets(cls):
    """Widget names in saved order, with the frontend's extra seed control."""
    out = []
    for name, (_, v) in _inputs(cls).items():
        kind, opts = v[0], (v[1] if len(v) > 1 else {})
        if isinstance(kind, list) or kind in NODE_WIDGET_TYPES:
            out.append(name)
            if kind == "INT" and (opts.get("control_after_generate")
                                  or name in ("seed", "noise_seed")):
                out.append(None)          # "fixed" / "randomize" / ...
    return out


def _workflows():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, "example_workflows")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json"))


def test_example_workflows_exist():
    names = [os.path.basename(p) for p in _workflows()]
    assert any("Auto" in n for n in names), names


@pytest.mark.parametrize("path", _workflows(), ids=os.path.basename)
def test_example_workflows_match_the_nodes(S, path):
    """Saved widgets_values are positional; a stale count shifts every value."""
    with open(path) as f:
        wf = json.load(f)
    ids = {n["id"]: n for n in wf["nodes"]}
    for lid, src, sslot, dst, dslot, typ in wf["links"]:
        assert lid in (ids[src]["outputs"][sslot].get("links") or [])
        assert ids[dst]["inputs"][dslot]["link"] == lid
    checked = 0
    for node in wf["nodes"]:
        cls = S.NODE_CLASS_MAPPINGS.get(node["type"])
        if cls is None:
            continue
        spec = _inputs(cls)
        widgets = _widgets(cls)
        assert len(node["widgets_values"]) == len(widgets), node["type"]
        for name, value in zip(widgets, node["widgets_values"]):
            if name is None:
                assert value in ("fixed", "increment", "decrement", "randomize")
                continue
            kind, opts = spec[name][1][0], (spec[name][1][1] if len(spec[name][1]) > 1 else {})
            if isinstance(kind, list):
                assert value in kind, (node["type"], name, value)
            elif kind in ("INT", "FLOAT"):
                assert opts.get("min", -1e30) <= value <= opts.get("max", 1e30), (name, value)
            elif kind == "BOOLEAN":
                assert isinstance(value, bool), (name, value)
        checked += 1
    assert checked >= 1


def test_every_node_is_findable(S):
    """One menu folder, a description and a display name for every node."""
    for key, cls in S.NODE_CLASS_MAPPINGS.items():
        assert cls.CATEGORY.startswith("sampling/DDRK Omega"), key
        assert getattr(cls, "DESCRIPTION", ""), key
        assert S.NODE_DISPLAY_NAME_MAPPINGS[key].startswith("DDRK"), key
