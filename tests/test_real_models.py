"""The nodes on real ComfyUI models: SD 1.5 and Flux architectures, shrunk to a
few hundred thousand random weights so they run on a CPU in seconds.

Nothing here is faked: comfy.supported_models builds the model, ModelPatcher
wraps it, and sampling goes through comfy.sample.sample_custom - CFGGuider
with real cond/uncond batching, conditioning processing, model management,
KSAMPLER, noise scaling. The images are noise; what is checked is that every
integration path runs, stays finite, spends the model calls it says it does,
and ends up on the right family's code path.
"""

import contextlib
import io

import pytest
import torch

CPU = torch.device("cpu")


def _patcher(model):
    import comfy.model_patcher
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    return comfy.model_patcher.ModelPatcher(model, load_device=CPU, offload_device=CPU)


def _build(make):
    # Model construction is ComfyUI's internal API and changes between
    # releases; a change there is not a DDRK failure, so skip loudly.
    try:
        return make()
    except TypeError as e:
        pytest.skip(f"tiny model construction no longer matches this ComfyUI: {e}")


def tiny_sd15():
    import comfy.supported_models as sm
    unet = {"use_checkpoint": False, "image_size": 32, "out_channels": 4,
            "use_spatial_transformer": True, "legacy": False, "adm_in_channels": None,
            "in_channels": 4, "model_channels": 32, "num_res_blocks": [1, 1],
            "transformer_depth": [1, 1], "channel_mult": [1, 2],
            "transformer_depth_middle": 1, "use_linear_in_transformer": False,
            "context_dim": 32, "num_heads": 2, "num_head_channels": -1,
            "transformer_depth_output": [1, 1, 1, 1], "use_temporal_resblock": False,
            "use_temporal_attention": False}
    return _patcher(sm.SD15(unet).get_model({}, device=CPU))


def tiny_flux(schnell=False):
    import comfy.supported_models as sm
    unet = {"image_model": "flux", "in_channels": 16, "patch_size": 2, "out_channels": 16,
            "vec_in_dim": 16, "context_in_dim": 32, "hidden_size": 64, "mlp_ratio": 2.0,
            "num_heads": 2, "depth": 1, "depth_single_blocks": 1, "axes_dim": [8, 12, 12],
            "theta": 10000, "qkv_bias": True, "guidance_embed": not schnell,
            "txt_ids_dims": []}
    return _patcher((sm.FluxSchnell if schnell else sm.Flux)(unet).get_model({}, device=CPU))


class _Counter:
    """Counts diffusion-model forward passes (cond and uncond batched together)."""

    def __init__(self, patcher):
        self.calls = 0
        dm = patcher.model.diffusion_model
        orig = dm.forward

        def fwd(*a, **k):
            self.calls += 1
            return orig(*a, **k)
        dm.forward = fwd


def _sd_conds():
    g = torch.Generator().manual_seed(1)
    return ([[torch.randn(1, 77, 32, generator=g), {}]],
            [[torch.randn(1, 77, 32, generator=g), {}]])


def _flux_conds():
    g = torch.Generator().manual_seed(1)
    return ([[torch.randn(1, 8, 32, generator=g), {"pooled_output": torch.randn(1, 16, generator=g)}]],
            [[torch.randn(1, 8, 32, generator=g), {"pooled_output": torch.randn(1, 16, generator=g)}]])


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def test_auto_on_sd15_uses_real_cfg(S):
    model = _build(tiny_sd15)
    calls = _Counter(model)
    pos, neg = _sd_conds()
    out, info = _quiet(S.DDRKOmegaAutoNode().sample_auto, model, pos, neg,
                       {"samples": torch.zeros(1, 4, 16, 16)}, 3, "fast", "auto", 0, 0.0)
    assert "SD 1.5" in info and "CFG 7" in info and "15 steps" in info
    assert out["samples"].shape == (1, 4, 16, 16)
    assert bool(torch.isfinite(out["samples"]).all())
    assert calls.calls == 15            # cond+uncond share one batched call per step


def test_auto_on_flux_best_and_schnell(S):
    pos, neg = _flux_conds()
    model = _build(tiny_flux)
    calls = _Counter(model)
    out, info = _quiet(S.DDRKOmegaAutoNode().sample_auto, model, pos, neg,
                       {"samples": torch.zeros(1, 16, 16, 16)}, 3, "best", "auto", 0, 0.0)
    assert "Flux" in info and "second pass" in info
    assert tuple(out["samples"].shape) == (1, 16, 22, 22)
    assert bool(torch.isfinite(out["samples"]).all()) and calls.calls == 20 + 8

    schnell = _build(lambda: tiny_flux(schnell=True))
    out, info = _quiet(S.DDRKOmegaAutoNode().sample_auto, schnell, pos, neg,
                       {"samples": torch.zeros(1, 16, 16, 16)}, 3, "balanced", "auto", 0, 0.0)
    assert "schnell" in info and "6 steps" in info and "CFG 1" in info


def test_unified_img2img_on_edm_takes_the_edm_path(S, tmp_path, monkeypatch):
    """Real SD1.5 at denoise 0.4 starts below sigma 5: still EDM (1.10.0 fix)."""
    import folder_paths
    import json
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    model = _build(tiny_sd15)
    pos, neg = _sd_conds()
    init = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(2))
    out = _quiet(S.DDRKOmegaUnifiedKSamplerNode().sample, model, pos, neg,
                 {"samples": init}, 4, 10, 6.0, 0.4, "ddrk_auto", 3.0, False, "hc3",
                 0.3, 0.3, 0, False, False, debug_mode=True, debug_tag="img2img")[0]
    assert bool(torch.isfinite(out["samples"]).all())
    meta = json.loads(next(p for p in tmp_path.iterdir() if p.suffix == ".json").read_text())["meta"]
    assert meta["is_edm"] is True and meta["family_source"] == "model_sampling"
    assert meta["sigma_start"] < 5.0


def test_lite_and_inpaint_mask_on_real_models(S):
    model = _build(tiny_sd15)
    pos, neg = _sd_conds()
    init = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(3))
    mask = torch.zeros(1, 1, 128, 128)
    mask[..., :64, :] = 1.0                   # redraw the top half only
    out = _quiet(S.DDRKOmegaLiteKSamplerNode().sample_lite, model, pos, neg,
                 {"samples": init, "noise_mask": mask}, 5, 8, 5.0, "balanced", "neutral")[0]
    assert bool(torch.isfinite(out["samples"]).all())
    # The masked-out half is handed back untouched.
    assert torch.allclose(out["samples"][..., 8:, :], init[..., 8:, :], atol=1e-4)
    assert not torch.allclose(out["samples"][..., :8, :], init[..., :8, :], atol=1e-2)
