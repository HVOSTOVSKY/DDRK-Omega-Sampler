"""Sigma schedules: shape, monotonicity, endpoints, and the ddrk_model path."""

import pytest
import torch

from analytic import make_model_sampling

CPU = torch.device("cpu")
EDM_MIN, EDM_MAX = 0.0292, 14.6146
FM_NAMES = ["ddrk_auto", "ddrk_cosine", "ddrk_beta", "ddrk_flow_linear",
            "ddrk_flow_cosmos", "ddrk_fewstep", "ddrk_anima"]
EDM_NAMES = ["ddrk_auto", "ddrk_edm_karras", "ddrk_edm_poly", "ddrk_edm_simple"]


def _check(sig, steps):
    assert sig.dim() == 1 and len(sig) == steps + 1
    assert float(sig[-1]) == 0.0
    assert bool(((sig[:-1] - sig[1:]) > 0).all()), sig.tolist()
    assert bool(torch.isfinite(sig).all())


@pytest.mark.parametrize("name", FM_NAMES)
@pytest.mark.parametrize("steps", [1, 2, 5, 8, 20, 50])
def test_fm_schedules(S, name, steps):
    ms = make_model_sampling(True, 3.0)
    sig = S.get_ddrk_sigmas(name, steps, float(ms.sigma_min), 1.0, CPU,
                            flow_shift=3.0, model_sampling=ms)
    _check(sig, steps)
    assert abs(float(sig[0]) - 1.0) < 1e-3


@pytest.mark.parametrize("name", EDM_NAMES)
@pytest.mark.parametrize("steps", [1, 2, 5, 8, 20, 50])
def test_edm_schedules(S, name, steps):
    sig = S.get_ddrk_sigmas(name, steps, EDM_MIN, EDM_MAX, CPU)
    _check(sig, steps)
    assert abs(float(sig[0]) - EDM_MAX) < 1e-3
    assert float(sig[-2]) >= EDM_MIN - 1e-6


def test_fm_auto_is_the_models_own_schedule(S):
    import comfy.samplers
    ms = make_model_sampling(True, 3.0)
    for steps in (4, 10, 25):
        ours = S.get_ddrk_sigmas("ddrk_auto", steps, float(ms.sigma_min), 1.0, CPU,
                                 flow_shift=1.0, model_sampling=ms)
        ref = comfy.samplers.calculate_sigmas(ms, "simple", steps)
        assert torch.allclose(ours, ref.float())


def test_ddrk_model_falls_back_loudly(S, capsys):
    sig = S.get_ddrk_sigmas("ddrk_model", 8, 0.001, 1.0, CPU, flow_shift=3.0,
                            model_sampling=None)
    lin = S.get_ddrk_sigmas("ddrk_flow_linear", 8, 0.001, 1.0, CPU, flow_shift=3.0)
    assert torch.equal(sig, lin)
    assert "unavailable" in capsys.readouterr().out


def test_unknown_scheduler_raises(S):
    with pytest.raises(ValueError):
        S.get_ddrk_sigmas("ddrk_cosin", 8, 0.001, 1.0, CPU)


def test_restart_insertion(S):
    sig = S.get_ddrk_sigmas("ddrk_edm_karras", 20, EDM_MIN, EDM_MAX, CPU)
    out, jumps = S._insert_restarts(sig, repeats=2, k_steps=3, t_min_frac=0.10,
                                    t_max_frac=0.35)
    assert jumps is not None and sum(jumps) == 2
    assert len(out) == len(sig) + 2 * (1 + 3)
    assert float(out[-1]) == 0.0
    # Restart window that no schedule point reaches: disabled, not silent.
    fm = S.get_ddrk_sigmas("ddrk_flow_linear", 8, 0.001, 1.0, CPU, flow_shift=3.0)
    out2, jumps2 = S._insert_restarts(fm, 1, 3, 0.01, 0.35)
    assert jumps2 is None and torch.equal(out2, fm)


def test_sigma_adapt_keeps_its_guarantees(S):
    ref = S.get_ddrk_sigmas("ddrk_flow_linear", 12, 0.001, 1.0, CPU, flow_shift=3.0)
    for activity in (0.01, 0.2, 5.0):
        work = ref.clone()
        S._adapt_remaining_sigmas(work, ref, 2, 1.0, activity, 0.1, 0.3)
        assert float(work[-1]) == 0.0 and torch.equal(work[:4], ref[:4])
        assert float(work[-2]) <= float(ref[-2]) + 1e-7          # last sigma never raised
        gaps, ref_gaps = work[:-1] - work[1:], ref[:-1] - ref[1:]
        assert bool((gaps[:-1] >= 0.5 * ref_gaps[:-1] - 1e-6).all())
        assert bool((gaps > 0).all())
