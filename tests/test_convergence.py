"""Integrator accuracy on the exactly solvable mixture problem (tests/analytic.py).

These are numerical-analysis facts, not image-quality claims: the measured
order of each integrator, and the equal-call comparisons the README relies on.
"""

import io
import math
import contextlib

import pytest
import torch

from analytic import (DirectModel, MixtureDenoiser, lambda_schedule,
                      make_model_sampling, reference_solution, rmse)

SHAPE = (2, 4, 16, 16)
SMOOTH = (0.25, 0.5)        # component stds: no mode flips at these step counts
PLAIN = dict(sde_strength=0.0, sharpness=0.0, saber_fusion=0.0,
             momentum_beta=0.0, auto_optimize=False, disable=True)


def _problem(flow, stds=SMOOTH, shape=SHAPE):
    den = MixtureDenoiser(shape, flow, stds=stds)
    smax = 1.0 if flow else 14.6146
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(shape, generator=g, dtype=torch.float64) * smax
    return den, smax, x0


_REF = {}


def _reference(flow, sigma_end):
    key = (flow, sigma_end)
    if key not in _REF:
        den, smax, x0 = _problem(flow)
        _REF[key] = reference_solution(den, x0, smax, sigma_end, n=3000)
    return _REF[key]


def _run(S, den, x0, sig, **kw):
    args = dict(PLAIN)
    args.update(kw)
    m = DirectModel(den)
    with contextlib.redirect_stdout(io.StringIO()):
        out = S.sample_ddrk_omega(m, x0.clone(), sig, **args)
    return out, m.calls


def _order(S, flow, integrator, ns, **kw):
    den, smax, x0 = _problem(flow)
    ref = _reference(flow, 0.01)
    errs = [rmse(_run(S, den, x0, lambda_schedule(smax, 0.01, n),
                      integrator=integrator, **kw)[0], ref) for n in ns]
    return math.log2(errs[-2] / errs[-1]), errs


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
@pytest.mark.parametrize("integrator,ns,lo,hi", [
    ("euler", (32, 64), 0.85, 1.2),
    ("heun", (32, 64), 1.8, 2.3),
    ("rk4", (16, 32), 3.5, 4.6),
    ("hc2", (32, 64), 1.8, 2.3),
])
def test_convergence_order(S, flow, integrator, ns, lo, hi):
    order, errs = _order(S, flow, integrator, ns)
    assert lo <= order <= hi, (integrator, order, errs)


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
def test_hc2_beats_euler_at_equal_calls(S, flow):
    den, smax, x0 = _problem(flow)
    ref = _reference(flow, 0.01)
    for n in (8, 16, 32):
        sig = lambda_schedule(smax, 0.01, n)
        e_hc2, c_hc2 = _run(S, den, x0, sig, integrator="hc2")
        e_eul, c_eul = _run(S, den, x0, sig, integrator="euler")
        assert c_hc2 == c_eul == n
        assert rmse(e_hc2, ref) < 0.5 * rmse(e_eul, ref)


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
@pytest.mark.parametrize("integrator", ["heun", "rk4"])
def test_momentum_does_not_touch_higher_order_integrators(S, flow, integrator):
    """1.9.0 applied AB2 momentum on top of Heun/RK4 and made them first order."""
    den, smax, x0 = _problem(flow)
    sig = lambda_schedule(smax, 0.01, 16)
    a, _ = _run(S, den, x0, sig, integrator=integrator, momentum_beta=0.0)
    b, _ = _run(S, den, x0, sig, integrator=integrator, momentum_beta=0.5)
    assert torch.equal(a, b)


@pytest.mark.parametrize("flow", [False, True], ids=["EDM", "FM"])
def test_full_momentum_makes_euler_second_order(S, flow):
    order, errs = _order(S, flow, "euler", (32, 64), momentum_beta=1.0)
    assert order > 1.7, errs


def test_free_corrector_is_free_and_more_accurate_on_edm(S):
    """Zero-cost corrector: same model calls, lower error (EDM, Karras)."""
    den, smax, x0 = _problem(False)
    ms = make_model_sampling(False)
    ref = reference_solution(den, x0, smax, 0.0, n=3000)
    for n in (12, 20, 30):
        sig = S.get_ddrk_sigmas("ddrk_edm_karras", n, float(ms.sigma_min), smax,
                                torch.device("cpu"))
        plain, c0 = _run(S, den, x0, sig, integrator="hc2")
        corr, c1 = _run(S, den, x0, sig, integrator="hc2", hc2_free_corrector=True)
        assert c0 == c1 == n
        assert rmse(corr, ref) < 0.6 * rmse(plain, ref), n


def test_free_corrector_raises_hc2_order(S):
    for flow in (False, True):
        o2, _ = _order(S, flow, "hc2", (32, 64))
        oc, errs = _order(S, flow, "hc2", (32, 64), hc2_free_corrector=True)
        assert oc > o2 + 0.5, (flow, o2, oc, errs)


def test_auto_mode_is_not_less_accurate_than_1_9(S):
    """HC2 steps in auto mode now extrapolate from the previous step."""
    den, smax, x0 = _problem(False)
    ms = make_model_sampling(False)
    ref = reference_solution(den, x0, smax, 0.0, n=3000)
    sig = S.get_ddrk_sigmas("ddrk_edm_karras", 24, float(ms.sigma_min), smax,
                            torch.device("cpu"))
    out, calls = _run(S, den, x0, sig, integrator="auto")
    hc2, _ = _run(S, den, x0, sig, integrator="hc2")
    assert rmse(out, ref) < rmse(hc2, ref)       # auto spends more calls
    assert calls > 24
