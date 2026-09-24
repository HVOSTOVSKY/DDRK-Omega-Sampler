"""HC3: HC2's predictor with trust damping plus the zero-cost corrector.

Claims checked here, all on the exactly solvable problem (tests/analytic.py):
third order at one model call per step; more accurate than HC2 at equal
calls at CFG 1 and 4-6; far more robust than HC2 at high CFG with few steps;
closer to the data distribution at few steps.
"""

import contextlib
import io
import json
import math

import pytest
import torch

from analytic import (DirectModel, GuidedDenoiser, MixtureDenoiser,
                      distribution_error, lambda_schedule, make_model_sampling,
                      reference_solution, rmse)

PLAIN = dict(sde_strength=0.0, sharpness=0.0, saber_fusion=0.0,
             momentum_beta=0.0, auto_optimize=False, disable=True)
CPU = torch.device("cpu")


def _run(S, den, x, sig, **kw):
    args = dict(PLAIN)
    args.update(kw)
    m = DirectModel(den)
    with contextlib.redirect_stdout(io.StringIO()):
        out = S.sample_ddrk_omega(m, x.clone(), sig, **args)
    return out, m.calls


def _sched(S, flow, n):
    import comfy.samplers
    ms = make_model_sampling(flow, 3.0)
    if flow:
        return comfy.samplers.calculate_sigmas(ms, "simple", n).float()
    return S.get_ddrk_sigmas("ddrk_edm_karras", n, float(ms.sigma_min),
                             float(ms.sigma_max), CPU)


_CACHE = {}


def _guided(flow, w):
    key = (flow, w)
    if key not in _CACHE:
        shape = (2, 4, 24, 24)
        den = GuidedDenoiser(shape, flow, w)
        smax = 1.0 if flow else float(make_model_sampling(False).sigma_max)
        g = torch.Generator().manual_seed(1)
        x0 = torch.randn(shape, generator=g, dtype=torch.float64) * smax
        _CACHE[key] = (den, x0, reference_solution(den, x0, smax, 0.0, n=3000))
    return _CACHE[key]


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
def test_hc3_is_third_order(S, flow):
    shape = (2, 4, 16, 16)
    den = MixtureDenoiser(shape, flow, stds=(0.25, 0.5))
    smax = 1.0 if flow else 14.6146
    x0 = torch.randn(shape, generator=torch.Generator().manual_seed(1),
                     dtype=torch.float64) * smax
    ref = reference_solution(den, x0, smax, 0.01, n=3000)
    errs = [rmse(_run(S, den, x0, lambda_schedule(smax, 0.01, n),
                      integrator="hc3")[0], ref) for n in (32, 64)]
    assert math.log2(errs[0] / errs[1]) > 2.6, errs


@pytest.mark.parametrize("flow,w,calls,factor", [
    (False, 1.0, (10, 16, 25), 0.7),     # EDM, no guidance
    (False, 6.0, (6, 8, 10), 0.65),      # EDM, high CFG, few steps
    (True, 4.0, (16, 25), 0.9),          # FM, CFG 4
])
def test_hc3_beats_hc2_at_equal_calls(S, flow, w, calls, factor):
    den, x0, ref = _guided(flow, w)
    for n in calls:
        sig = _sched(S, flow, n)
        e3, c3 = _run(S, den, x0.float(), sig, integrator="hc3")
        e2, c2 = _run(S, den, x0.float(), sig, integrator="hc2")
        assert c3 == c2 == n
        assert rmse(e3, ref) < factor * rmse(e2, ref), (n, rmse(e3, ref), rmse(e2, ref))


def test_hc3_stays_close_to_euler_where_multistep_overshoots(S):
    """CFG 6 at 5-8 calls: HC2 was 2-2.4x worse than Euler; HC3 within 30%."""
    den, x0, ref = _guided(False, 6.0)
    for n in (6, 8):
        sig = _sched(S, False, n)
        e3 = rmse(_run(S, den, x0.float(), sig, integrator="hc3")[0], ref)
        eu = rmse(_run(S, den, x0.float(), sig, integrator="euler")[0], ref)
        assert e3 < 1.3 * eu, (n, e3, eu)


def test_hc3_closer_to_the_data_distribution_at_few_steps(S):
    shape = (8, 4, 32, 32)
    den = MixtureDenoiser(shape, False, stds=(0.08, 0.2))
    x0 = torch.randn(shape, generator=torch.Generator().manual_seed(2)) * 14.6146
    for n in (6, 10):
        sig = _sched(S, False, n)
        d3 = distribution_error(den, _run(S, den, x0, sig, integrator="hc3")[0])
        d2 = distribution_error(den, _run(S, den, x0, sig, integrator="hc2")[0])
        assert d3 < 0.7 * d2, (n, d3, d2)


def test_hc3_trust_is_logged_and_sane(S, tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    shape = (1, 4, 16, 16)
    for w, lo, hi in ((1.0, 0.8, 1.0), (6.0, 0.0, 0.8)):
        den = GuidedDenoiser(shape, False, w)
        sig = _sched(S, False, 8 if w > 1 else 40)
        x0 = torch.randn(shape, generator=torch.Generator().manual_seed(0)) * float(sig[0])
        with contextlib.redirect_stdout(io.StringIO()):
            S.sample_ddrk_omega(DirectModel(den), x0, sig, debug_mode=True,
                                debug_tag=f"w{w}", integrator="hc3", **{
                                    k: v for k, v in PLAIN.items() if k != "disable"})
        path = next(p for p in tmp_path.iterdir()
                    if p.suffix == ".json" and f"_w{w}_" in p.name)
        trust = [s["hc3_trust"] for s in json.loads(path.read_text())["steps"]
                 if "hc3_trust" in s]
        assert trust and all(0.0 <= t <= 1.0 for t in trust)
        mean = sum(trust) / len(trust)
        assert lo <= mean <= hi, (w, mean)


def test_hc3_keeps_hc2_untouched(S):
    """hc2 without the new options is the 1.10.0 integrator, bit for bit."""
    shape = (1, 4, 16, 16)
    den = GuidedDenoiser(shape, True, 4.0)
    sig = _sched(S, True, 12)
    x0 = torch.randn(shape, generator=torch.Generator().manual_seed(0))
    a, _ = _run(S, den, x0, sig, integrator="hc2")
    b, _ = _run(S, den, x0, sig, integrator="hc2", hc2_free_corrector=False)
    assert torch.equal(a, b)
    c, _ = _run(S, den, x0, sig, integrator="hc3")
    assert not torch.equal(a, c)
