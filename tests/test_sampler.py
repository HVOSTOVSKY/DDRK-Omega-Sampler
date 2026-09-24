"""sample_ddrk_omega behaviour: determinism, shapes, call counts, and the
1.10.0 fixes (family detection, churn, final clamp, noise_scale)."""

import io
import json
import math
import contextlib

import pytest
import torch

from analytic import (ChainModel, DirectModel, MixtureDenoiser,
                      make_model_sampling)

CPU = torch.device("cpu")
PLAIN = dict(sde_strength=0.0, sharpness=0.0, saber_fusion=0.0,
             momentum_beta=0.0, auto_optimize=False, disable=True)


def _sched(S, flow, n):
    if flow:
        return S.get_ddrk_sigmas("ddrk_flow_linear", n, 0.001, 1.0, CPU, flow_shift=3.0)
    return S.get_ddrk_sigmas("ddrk_edm_karras", n, 0.0292, 14.6146, CPU)


def _x(shape, sigma, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * sigma


def _run(S, model, x, sig, **kw):
    args = dict(PLAIN)
    args.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return S.sample_ddrk_omega(model, x.clone(), sig, **args)


ENHANCED = dict(integrator="auto", sde_strength=0.1, sde_seed=7, sharpness=0.12,
                saber_fusion=0.2, momentum_beta=0.25, sigma_adapt=0.1)


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
@pytest.mark.parametrize("shape", [(1, 4, 16, 16), (2, 16, 8, 8), (1, 16, 3, 8, 8)])
def test_runs_finite_and_reproducible(S, flow, shape):
    den = MixtureDenoiser(shape, flow)
    sig = _sched(S, flow, 12)
    x = _x(shape, float(sig[0]))
    a = _run(S, DirectModel(den), x, sig, **ENHANCED)
    b = _run(S, DirectModel(den), x, sig, **ENHANCED)
    assert a.shape == x.shape and a.dtype == x.dtype
    assert bool(torch.isfinite(a).all())
    assert torch.equal(a, b)


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
def test_hc2_option_combinations(S, flow):
    shape = (1, 16, 3, 8, 8)
    den = MixtureDenoiser(shape, flow)
    sig = _sched(S, flow, 12)
    x = _x(shape, float(sig[0]))
    for kw in (dict(hc2_free_corrector=True, hc2_max_order=3),
               dict(hc2_free_corrector=True, hc2_space="flow", sigma_adapt=0.2),
               dict(hc2_free_corrector=True, hc2_corrector=0.05, limiter_kappa=0.5),
               dict(hc2_free_corrector=True, sde_strength=0.2, sde_seed=1,
                    saber_fusion=0.3, sharpness=0.2, s_churn=5.0,
                    restart_repeats=1, restart_t_min=0.4 if flow else 0.1,
                    restart_t_max=0.7 if flow else 0.35)):
        out = _run(S, DirectModel(den), x, sig, integrator="hc2", **kw)
        assert out.shape == x.shape and bool(torch.isfinite(out).all()), kw


def test_sde_seed_minus_one_follows_the_global_seed(S):
    shape = (1, 4, 16, 16)
    den = MixtureDenoiser(shape, True)
    sig = _sched(S, True, 10)
    x = _x(shape, 1.0)
    outs = []
    for seed in (5, 5, 6):
        torch.manual_seed(seed)
        outs.append(_run(S, DirectModel(den), x, sig, integrator="hc2",
                         sde_strength=0.3, sde_seed=None))
    assert torch.equal(outs[0], outs[1])
    assert not torch.equal(outs[0], outs[2])


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
def test_batch_items_do_not_leak(S, flow):
    """An item sampled in a batch of two matches the same item sampled alone."""
    shape = (2, 4, 16, 16)
    den = MixtureDenoiser(shape, flow)
    sig = _sched(S, flow, 10)
    x = _x(shape, float(sig[0]))
    kw = dict(integrator="hc2", sharpness=0.12, saber_fusion=0.2)
    both = _run(S, DirectModel(den), x, sig, **kw)
    alone = _run(S, DirectModel(den), x[:1], sig, **kw)
    assert torch.allclose(both[:1], alone, atol=1e-5)


@pytest.mark.parametrize("integrator,expected", [
    ("euler", lambda n: n), ("hc2", lambda n: n), ("heun", lambda n: 2 * n - 1),
    ("rk4", lambda n: 4 * (n - 1) + 1)])
def test_model_calls_per_integrator(S, integrator, expected):
    shape = (1, 4, 8, 8)
    for flow in (False, True):
        n = 12
        m = DirectModel(MixtureDenoiser(shape, flow))
        sig = _sched(S, flow, n)
        _run(S, m, _x(shape, float(sig[0])), sig, integrator=integrator)
        assert m.calls == expected(n), (integrator, flow, m.calls)


def test_hc2_options_call_counts(S):
    shape = (1, 4, 8, 8)
    for flow in (False, True):
        sig = _sched(S, flow, 12)
        x = _x(shape, float(sig[0]))
        for kw, calls in (({"hc2_max_order": 3}, 13), ({"hc2_free_corrector": True}, 12)):
            m = DirectModel(MixtureDenoiser(shape, flow))
            _run(S, m, x, sig, integrator="hc2", **kw)
            assert m.calls == calls, (kw, m.calls)


def test_few_steps_replace_non_hc2_integrators(S, capsys):
    shape = (1, 4, 8, 8)
    m = DirectModel(MixtureDenoiser(shape, True))
    sig = _sched(S, True, 5)
    S.sample_ddrk_omega(m, _x(shape, 1.0), sig, **dict(PLAIN, integrator="rk4"))
    assert m.calls == 5
    assert "REPLACED" in capsys.readouterr().out


def test_unknown_names_raise(S):
    shape = (1, 4, 8, 8)
    m = DirectModel(MixtureDenoiser(shape, True))
    sig = _sched(S, True, 8)
    with pytest.raises(ValueError):
        _run(S, m, _x(shape, 1.0), sig, integrator="hc-2")
    with pytest.raises(ValueError):
        _run(S, m, _x(shape, 1.0), sig, integrator="hc2", hc2_space="vp")


# ---------------------------------------------------------------- 1.10 fixes

def test_family_comes_from_the_model_not_the_schedule(S):
    """EDM img2img at low denoise starts below sigma 5: it is still EDM."""
    ms_edm, ms_fm = make_model_sampling(False), make_model_sampling(True)
    tail = S.get_ddrk_sigmas("ddrk_edm_karras", 40, 0.0292, 14.6146, CPU)[-21:]
    assert float(tail[0]) < 5.0
    den = MixtureDenoiser((1, 4, 8, 8), False)
    smax, is_edm, _, src = S._sampling_family(ChainModel(den, ms_edm), tail)
    assert is_edm and src == "model_sampling" and abs(smax - 14.6146) < 1e-3
    smax, is_edm, _, _ = S._sampling_family(ChainModel(den, ms_fm), tail)
    assert not is_edm and abs(smax - 1.0) < 1e-6
    # Without a ComfyUI model the old schedule heuristic remains.
    assert S._sampling_family(DirectModel(den), tail)[1] is False


def test_low_denoise_edm_run_stays_on_the_edm_path(S):
    """1.9.0 put this run through the FM SDE (which scales by 1 - sigma,
    negative above 1) and FM sharpening. On EDM both are off by design."""
    shape = (1, 4, 16, 16)
    den = MixtureDenoiser(shape, False)
    ms = make_model_sampling(False)
    tail = S.get_ddrk_sigmas("ddrk_edm_karras", 40, float(ms.sigma_min),
                             float(ms.sigma_max), CPU)[-21:]
    x = _x(shape, float(tail[0]))
    base = _run(S, ChainModel(den, ms), x, tail, integrator="heun")
    fm_only = _run(S, ChainModel(den, ms), x, tail, integrator="heun",
                   sde_strength=0.3, sde_seed=1, sharpness=0.5)
    assert torch.equal(base, fm_only)


def test_final_clamp_leaves_noisy_handoff_alone(S):
    """A schedule split at sigma ~5 hands on a latent with std ~5; the old
    fixed +-7 bound clipped it."""
    shape = (1, 4, 32, 32)
    den = MixtureDenoiser(shape, False)
    full = S.get_ddrk_sigmas("ddrk_edm_karras", 20, 0.0292, 14.6146, CPU)
    head = full[:6]
    assert float(head[-1]) > 4.0
    out = _run(S, DirectModel(den), _x(shape, 14.6146), head, integrator="euler")
    assert float(out.abs().max()) > 10.0


def _debug_json(S, tmp_path, monkeypatch, model, x, sig, **kw):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    _run(S, model, x, sig, debug_mode=True, debug_tag="t", **kw)
    path = next(p for p in tmp_path.iterdir() if p.suffix == ".json")
    return json.loads(path.read_text())


def test_churn_follows_karras_algorithm_2(S, tmp_path, monkeypatch):
    shape = (1, 4, 8, 8)
    n, churn = 20, 5.0
    sig = _sched(S, False, n)
    log = _debug_json(S, tmp_path, monkeypatch, DirectModel(MixtureDenoiser(shape, False)),
                      _x(shape, float(sig[0])), sig, integrator="heun",
                      s_churn=churn, sde_seed=3)
    gammas = [s["churn_gamma"] for s in log["steps"] if s.get("churn_fired")]
    assert len(gammas) == n                     # every step, as in k-diffusion
    assert all(abs(g - churn / n) < 1e-5 for g in gammas)


def test_debug_log_contents(S, tmp_path, monkeypatch):
    shape = (1, 4, 8, 8)
    ms = make_model_sampling(True)
    sig = _sched(S, True, 8)
    log = _debug_json(S, tmp_path, monkeypatch, ChainModel(MixtureDenoiser(shape, True), ms),
                      _x(shape, 1.0), sig, integrator="hc2", hc2_free_corrector=True)
    meta, steps = log["meta"], log["steps"]
    assert meta["family_source"] == "model_sampling" and meta["is_edm"] is False
    assert meta["hc2_free_corrector"] is True
    assert len(steps) == 8 and sum(s["model_calls"] for s in steps) == 8
    assert any("hc2_free_corr_delta" in s for s in steps)
    assert not any(s["has_nan"] for s in steps)


def test_nonfinite_output_raises_in_debug_and_warns_otherwise(S, tmp_path, monkeypatch, capsys):
    shape = (1, 4, 8, 8)

    def nan_model(x, sigma, **kw):
        return torch.full_like(x, float("nan"))

    sig = _sched(S, True, 6)
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    with pytest.raises(RuntimeError):
        _run(S, nan_model, _x(shape, 1.0), sig, integrator="hc2", debug_mode=True)
    S.sample_ddrk_omega(nan_model, _x(shape, 1.0), sig, **dict(PLAIN, integrator="hc2"))
    assert "non-finite" in capsys.readouterr().out


def test_fm_noise_follows_model_noise_scale(S):
    """HiDream-O1 declares noise_scale 8: injected noise must scale with it."""
    shape = (1, 4, 16, 16)
    ms = make_model_sampling(True)
    assert S._sampling_family(ChainModel(MixtureDenoiser(shape, True), ms),
                              _sched(S, True, 4))[2] == 1.0
    ms.set_noise_scale(8.0)
    assert S._sampling_family(ChainModel(MixtureDenoiser(shape, True), ms),
                              _sched(S, True, 4))[2] == 8.0


def test_fm_restart_lands_on_the_marginal(S):
    """x = (1-s) x0 + s eps  ->  (1-s') x0 + s' eps' (signal and noise std)."""
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(400_000, generator=g) * 0.5 + 1.0
    s, s2 = 0.2, 0.6
    x = (1 - s) * x0 + s * torch.randn(400_000, generator=g)
    y = S._renoise_flow(x, s, s2, torch.randn(400_000, generator=g))
    assert abs(float(y.mean()) - (1 - s2) * 1.0) < 5e-3
    expected_std = math.sqrt(((1 - s2) * 0.5) ** 2 + s2 ** 2)
    assert abs(float(y.std()) - expected_std) < 5e-3


def test_fm_ancestral_split_matches_marginal(S):
    for sigma, nxt, eta in ((0.9, 0.7, 1.0), (0.5, 0.3, 0.4), (0.2, 0.05, 1.0)):
        down, ratio, renoise = S._ancestral_split_flow(sigma, nxt, eta)
        assert 0 < down <= nxt
        assert abs(ratio * (1 - down) - (1 - nxt)) < 1e-9       # signal
        assert abs(math.hypot(down * ratio, renoise) - nxt) < 1e-9   # noise
