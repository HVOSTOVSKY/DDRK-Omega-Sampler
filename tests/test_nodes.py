"""Node-level tests: Unified, Lite, Sampler, Scheduler and Smart Config.

comfy.sample.sample_custom is replaced by analytic.fake_sample_custom, which
skips conditioning and model loading but runs ComfyUI's real KSAMPLER, so
noise scaling, the inpaint wrapper and inverse noise scaling are the
production code paths.
"""

import io
import contextlib
import itertools

import pytest
import torch

import analytic
from analytic import FakePatcher

SHAPE = (1, 4, 16, 16)


@pytest.fixture(autouse=True)
def fake_sampling(monkeypatch):
    import comfy.sample
    monkeypatch.setattr(comfy.sample, "sample_custom", analytic.fake_sample_custom)


def _latent(shape=SHAPE, fill=0.0):
    return {"samples": torch.full(shape, fill)}


def _unified(S, model, latent, **kw):
    args = dict(seed=11, steps=10, cfg=1.0, denoise=1.0, scheduler_type="ddrk_auto",
                flow_shift=3.0, auto_flow_shift=False, integrator="hc2",
                sde_strength=0.0, sharpness=0.0, warmup_steps=0, auto_optimize=False,
                smart_defaults=False, saber_fusion=0.0, momentum_beta=0.0)
    args.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return S.DDRKOmegaUnifiedKSamplerNode().sample(model, [], [], latent, **args)[0]


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
def test_unified_end_to_end(S, flow):
    model = FakePatcher(flow, SHAPE)
    out = _unified(S, model, _latent())
    assert out["samples"].shape == SHAPE
    assert bool(torch.isfinite(out["samples"]).all())
    assert model.calls == 10


def test_unified_is_reproducible_from_the_seed(S):
    a = _unified(S, FakePatcher(True, SHAPE), _latent(), sde_strength=0.2)["samples"]
    b = _unified(S, FakePatcher(True, SHAPE), _latent(), sde_strength=0.2)["samples"]
    c = _unified(S, FakePatcher(True, SHAPE), _latent(), sde_strength=0.2, seed=12)["samples"]
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_unified_low_denoise_on_edm(S):
    """EDM img2img at denoise 0.4: FM-only features must stay off."""
    init = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))
    a = _unified(S, FakePatcher(False, SHAPE), {"samples": init.clone()},
                 denoise=0.4, integrator="heun")["samples"]
    b = _unified(S, FakePatcher(False, SHAPE), {"samples": init.clone()},
                 denoise=0.4, integrator="heun", sde_strength=0.3,
                 sharpness=0.5)["samples"]
    assert torch.equal(a, b)
    assert bool(torch.isfinite(a).all())


def test_unified_passes_batch_index_to_the_noise(S, monkeypatch):
    import comfy.sample
    seen = {}
    real = comfy.sample.prepare_noise

    def spy(latent, seed, noise_inds=None):
        seen["inds"] = noise_inds
        return real(latent, seed, noise_inds)

    monkeypatch.setattr(comfy.sample, "prepare_noise", spy)
    shape = (2,) + SHAPE[1:]
    _unified(S, FakePatcher(True, shape), {"samples": torch.zeros(shape),
                                           "batch_index": [3, 5]})
    assert list(seen["inds"]) == [3, 5]


@pytest.mark.parametrize("scale,shape,expect", [
    (1.33, (1, 4, 16, 16), (22, 22)),
    (1.5, (1, 4, 12, 20), (18, 30)),
    (1.33, (1, 16, 3, 16, 16), (22, 22)),
])
def test_second_pass_canvas(S, scale, shape, expect):
    model = FakePatcher(True, shape)
    out = _unified(S, model, _latent(shape), refine_scale=scale, refine_steps=4)
    assert tuple(out["samples"].shape[-2:]) == expect
    assert out["samples"].shape[:-2] == torch.Size(shape[:-2])
    assert model.calls == 10 + 4
    assert all(s[-2] % 2 == 0 and s[-1] % 2 == 0 for s in model.seen_shapes)


def test_second_pass_keeps_the_inpaint_mask(S):
    """noise_mask = 0 everywhere: nothing may be redrawn, in either pass."""
    init = torch.randn(SHAPE, generator=torch.Generator().manual_seed(3))
    latent = {"samples": init.clone(), "noise_mask": torch.zeros(1, 1, 16, 16)}
    out = _unified(S, FakePatcher(True, SHAPE), latent, refine_scale=1.5,
                   refine_steps=4)["samples"]
    expected = S._upscale_latent(init, 1.5)
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
def test_lite_is_exactly_the_unified_node(S, flow):
    for quality, character in itertools.product(("fast", "balanced", "best"),
                                                ("neutral", "sharp", "smooth")):
        with contextlib.redirect_stdout(io.StringIO()):
            lite = S.DDRKOmegaLiteKSamplerNode().sample_lite(
                FakePatcher(flow, SHAPE), [], [], _latent(), 11, 10, 1.0,
                quality, character)[0]["samples"]
        p = S._lite_sampler_params(not flow, quality, character, steps=10)
        full = _unified(S, FakePatcher(flow, SHAPE), _latent(),
                        integrator=p["integrator"], sharpness=p["sharpness"],
                        saber_fusion=p["saber_fusion"], sigma_adapt=p["sigma_adapt"],
                        hc2_max_order=p.get("hc2_max_order", 2),
                        refine_scale=p.get("refine_scale", 1.0),
                        refine_denoise=p.get("refine_denoise", 0.35),
                        refine_steps=p.get("refine_steps", 5))["samples"]
        assert torch.equal(lite, full), (quality, character)


def test_lite_rejects_unknown_dials(S):
    with pytest.raises(ValueError):
        S._lite_sampler_params(False, "ultra", "neutral")


def test_sampler_node_options(S):
    node = S.DDRKOmegaSamplerNode()
    spec = node.INPUT_TYPES()
    kwargs = {k: (v[1].get("default", v[0][0] if isinstance(v[0], list) else None))
              for k, v in spec["required"].items()}
    sampler = node.get_sampler(**kwargs, hc2_space="flow", hc2_free_corrector=True)[0]
    assert sampler.extra_options["hc2_space"] == "flow"
    assert sampler.extra_options["hc2_free_corrector"] is True
    assert sampler.extra_options["sde_seed"] is None


def test_sampler_node_runs_through_ksampler(S):
    """The SamplerCustom path: our SAMPLER object in ComfyUI's KSAMPLER."""
    import comfy.samplers
    model = FakePatcher(True, SHAPE)
    sampler = comfy.samplers.KSAMPLER(S.sample_ddrk_omega, extra_options=dict(
        integrator="hc2", sde_strength=0.0, sharpness=0.0, saber_fusion=0.0,
        momentum_beta=0.0, auto_optimize=False))
    sig = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"),
                                          "simple", 8)
    noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))
    out = analytic.fake_sample_custom(model, noise, 1.0, sampler, sig, [], [],
                                      torch.zeros(SHAPE), seed=0)
    assert out.shape == SHAPE and model.calls == 8


def test_scheduler_node(S):
    for flow in (False, True):
        sig = S.DDRKOmegaSchedulerNode().get_sigmas(
            FakePatcher(flow, SHAPE), 12, "ddrk_auto", 3.0, 0, True)[0]
        assert len(sig) == 13 and float(sig[-1]) == 0.0


def test_smart_config_profiles(S):
    node = S.DDRKOmegaSmartConfigNode()
    with contextlib.redirect_stdout(io.StringIO()):
        fm = node.detect(FakePatcher(True, SHAPE, image_model="flux"), _latent())
        edm = node.detect(FakePatcher(False, SHAPE), _latent())
    assert fm[0] == "flux" and fm[5] == "hc2"
    assert edm[0] == "sd15" and edm[5] == "auto" and edm[7] == 0.0


def test_flux_conditioning_passthrough_and_scaling(S):
    node = S.DDRKFluxConditioning()
    emb = torch.ones(1, 6, 8)
    cond = [[emb, {"attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]])}]]
    with contextlib.redirect_stdout(io.StringIO()):
        same = node.apply(cond, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, False)[0]
        out = node.apply(cond, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, False)[0]
    assert same is cond
    t = out[0][0]
    assert torch.equal(t[0, :4], torch.full((4, 8), 2.0))
    assert torch.equal(t[0, 4:], torch.zeros(2, 8))
    assert torch.equal(emb, torch.ones(1, 6, 8))           # input untouched
