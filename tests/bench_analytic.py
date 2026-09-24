"""Accuracy bench on the exactly solvable mixture problem (tests/analytic.py).

    COMFYUI_PATH=/path/to/ComfyUI python tests/bench_analytic.py

Prints RMSE to the exact ODE solution at equal model calls for the
integrators and options discussed in the README, on EDM (Karras schedule)
and Flow Matching (the model's own schedule, shift 3), for a smooth and a
sharp mixture. CPU only, a few minutes. These are numerical results on a
known problem, not image-quality measurements.
"""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_env import bootstrap  # noqa: E402

if bootstrap() is None:
    sys.exit("ComfyUI not found: set COMFYUI_PATH to a ComfyUI checkout.")

import torch  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.k_diffusion.sampling as kds  # noqa: E402
import ddrk_omega.sampler as S  # noqa: E402
from analytic import (DirectModel, MixtureDenoiser, make_model_sampling,  # noqa: E402
                      reference_solution, rmse)

PLAIN = dict(sde_strength=0.0, sharpness=0.0, saber_fusion=0.0,
             momentum_beta=0.0, auto_optimize=False, disable=True)
CALLS = (8, 12, 20, 30)


def ddrk(**kw):
    def run(model, x, sig):
        args = dict(PLAIN)
        args.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return S.sample_ddrk_omega(model, x.clone(), sig, **args)
    return run


def dpmpp_2m(model, x, sig):
    return kds.sample_dpmpp_2m(model, x.clone(), sig, disable=True)


# name, runner, model calls per step (the schedule gets calls // this)
ROWS = [
    ("euler", ddrk(integrator="euler"), 1),
    ("heun", ddrk(integrator="heun"), 2),
    ("heun, momentum 0.25 (1.9.0 behaviour)", None, 2),
    ("rk4", ddrk(integrator="rk4"), 4),
    ("dpmpp_2m (ComfyUI)", dpmpp_2m, 1),
    ("hc2", ddrk(integrator="hc2"), 1),
    ("hc2, hc2_space=flow", ddrk(integrator="hc2", hc2_space="flow"), 1),
    ("hc2, sigma_adapt=0.10", ddrk(integrator="hc2", sigma_adapt=0.10), 1),
    ("hc2 + free corrector", ddrk(integrator="hc2", hc2_free_corrector=True), 1),
    ("auto", ddrk(integrator="auto"), 0),
]


def heun_momentum_190(model, x, sig):
    """Heun with AB2 applied on top of it, as 1.9.0 did (for comparison)."""
    s_in = x.new_ones([x.shape[0]])
    d_prev = dt_prev = None
    x = x.clone()
    for i in range(len(sig) - 1):
        s, sn = sig[i], sig[i + 1]
        dt = float(sn - s)
        d = (x - model(x, s * s_in)) / s
        if float(sn) == 0.0:
            x = x + d * dt
            break
        x2 = x + d * dt
        d_avg = 0.5 * (d + (x2 - model(x2, sn * s_in)) / sn)
        use = d_avg
        if d_prev is not None:
            ratio = max(-1.0, min(1.0, 0.25 * dt / (2.0 * dt_prev) * 0.5))
            use = d_avg + ratio * (d_avg - d_prev)
        d_prev, dt_prev = d_avg, dt
        x = x + use * dt
    return x


def main():
    for stds in ((0.25, 0.5), (0.08, 0.2)):
        for flow in (False, True):
            shape = (2, 4, 24, 24)
            den = MixtureDenoiser(shape, flow, stds=stds)
            ms = make_model_sampling(flow, 3.0)
            smax = float(ms.sigma_max)

            def sched(n):
                n = max(n, 1)
                if flow:
                    return comfy.samplers.calculate_sigmas(ms, "simple", n).float()
                return S.get_ddrk_sigmas("ddrk_edm_karras", n, float(ms.sigma_min),
                                         smax, torch.device("cpu"))

            g = torch.Generator().manual_seed(1)
            x0 = torch.randn(shape, generator=g, dtype=torch.float64) * smax
            ref = reference_solution(den, x0, smax, 0.0, n=6000)
            title = "Flow Matching, model schedule (shift 3)" if flow else "EDM, Karras"
            print(f"\n{title}; mixture stds {stds}; RMSE to the exact solution "
                  f"(calls:error)")
            for name, fn, per_step in ROWS:
                if name.startswith("heun, momentum"):
                    fn = heun_momentum_190
                if flow and name == "rk4":
                    continue
                cells = []
                for calls in CALLS:
                    m = DirectModel(den)
                    steps = calls // per_step if per_step else calls // 2
                    out = fn(m, x0, sched(steps))
                    cells.append(f"{m.calls:3d}:{rmse(out, ref):.2e}")
                print(f"  {name:38s}" + "  ".join(cells))


if __name__ == "__main__":
    main()
