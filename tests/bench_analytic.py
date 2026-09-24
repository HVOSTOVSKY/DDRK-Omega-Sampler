"""DDRK against ComfyUI's own samplers on the exactly solvable problem.

    COMFYUI_PATH=/path/to/ComfyUI python tests/bench_analytic.py

Three tables, all at EQUAL MODEL CALLS on the schedule each family normally
uses (EDM: Karras; Flow Matching: the model's own, shift 3):

  1. ODE accuracy at CFG 1: RMSE to the exact solution of the sampling ODE.
  2. ODE accuracy at high CFG (EDM 6, FM 4): the regime where multistep
     methods overshoot.
  3. Distribution fidelity: how far the samples are from the true data
     distribution (tests/analytic.distribution_error, x1000; an exact
     sampler scores ~1.5). The analogue of FID on this problem, and the only
     fair table for the stochastic samplers.

CPU only, a few minutes. These are numerics on a known problem, not image
quality; see the README for what they do and do not say.
"""

import contextlib
import io
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_env import bootstrap  # noqa: E402

if bootstrap() is None:
    sys.exit("ComfyUI not found: set COMFYUI_PATH to a ComfyUI checkout.")

import torch  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.k_diffusion.sampling as kds  # noqa: E402
import ddrk_omega.sampler as S  # noqa: E402
from analytic import (GuidedDenoiser, MixtureDenoiser, distribution_error,  # noqa: E402
                      make_model_sampling, reference_solution, rmse)

CALLS = (6, 10, 16, 25)


class KModel:
    """What ComfyUI's k-diffusion samplers expect of `model`."""

    def __init__(self, den, ms):
        self.den, self.calls = den, 0
        patcher = SimpleNamespace(get_model_object=lambda name: ms)
        self.inner_model = SimpleNamespace(
            inner_model=SimpleNamespace(model_sampling=ms), model_patcher=patcher)

    def __call__(self, x, sigma, **kw):
        self.calls += 1
        return self.den(x, sigma)


def ddrk(**kw):
    def run(m, x, sig):
        args = dict(sde_strength=0.0, sharpness=0.0, saber_fusion=0.0,
                    momentum_beta=0.0, auto_optimize=False, disable=True)
        args.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return S.sample_ddrk_omega(m, x.clone(), sig, extra_args={"seed": 0}, **args)
    return run


def comfy_sampler(name):
    def run(m, x, sig):
        sampler = comfy.samplers.sampler_object(name)
        with contextlib.redirect_stdout(io.StringIO()):
            return sampler.sampler_function(m, x.clone(), sig, extra_args={"seed": 0},
                                            callback=None, disable=True,
                                            **sampler.extra_options)
    return run


ODE = [("euler", comfy_sampler("euler")), ("dpmpp_2m", comfy_sampler("dpmpp_2m")),
       ("ipndm", comfy_sampler("ipndm")), ("deis", comfy_sampler("deis")),
       ("uni_pc_bh2", comfy_sampler("uni_pc_bh2")),
       ("res_multistep", comfy_sampler("res_multistep")),
       ("DDRK hc2", ddrk(integrator="hc2")), ("DDRK hc3", ddrk(integrator="hc3"))]
SDE = [("euler_ancestral", comfy_sampler("euler_ancestral")),
       ("dpmpp_2m_sde", comfy_sampler("dpmpp_2m_sde")),
       ("dpmpp_3m_sde", comfy_sampler("dpmpp_3m_sde")),
       ("er_sde", comfy_sampler("er_sde")), ("sa_solver", comfy_sampler("sa_solver")),
       ("DDRK hc2 + sde 0.3 (FM)", ddrk(integrator="hc2", sde_strength=0.3, sde_seed=1))]


def schedule(ms, flow, n):
    return comfy.samplers.calculate_sigmas(ms, "simple" if flow else "karras", n).float()


def accuracy(flow, w):
    shape = (2, 4, 24, 24)
    ms = make_model_sampling(flow, 3.0)
    smax = float(ms.sigma_max)
    den = GuidedDenoiser(shape, flow, w)
    x0 = torch.randn(shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * smax
    ref = reference_solution(den, x0, smax, 0.0, n=4000)
    print(f"\n{'Flow Matching' if flow else 'EDM'}, CFG {w:g}: RMSE to the exact solution")
    for name, fn in ODE:
        if flow and name == "uni_pc_bh2":
            continue                          # VP-only in ComfyUI
        cells = []
        for calls in CALLS:
            m = KModel(den, ms)
            out = fn(m, x0.float(), schedule(ms, flow, calls))
            cells.append(f"{m.calls:3d}: {rmse(out, ref):.2e}")
        print(f"  {name:26s}" + "   ".join(cells))


def fidelity(flow):
    shape = (16, 4, 32, 32)
    ms = make_model_sampling(flow, 3.0)
    den = MixtureDenoiser(shape, flow, stds=(0.08, 0.2))
    x0 = torch.randn(shape, generator=torch.Generator().manual_seed(2)) * float(ms.sigma_max)
    print(f"\n{'Flow Matching' if flow else 'EDM'}: distance to the data distribution (x1000, exact ~1.5)")
    for name, fn in ODE + SDE:
        if flow and name == "uni_pc_bh2":
            continue
        cells = []
        for calls in CALLS:
            m = KModel(den, ms)
            out = fn(m, x0, schedule(ms, flow, calls))
            cells.append(f"{m.calls:3d}: {distribution_error(den, out) * 1e3:6.1f}")
        print(f"  {name:26s}" + "   ".join(cells))


def fm_schedules():
    """Flow Matching: the final jump to sigma 0 decides most of the error."""
    import comfy.samplers as cs
    shape = (16, 4, 32, 32)
    ms = make_model_sampling(True, 3.0)
    den = MixtureDenoiser(shape, True, stds=(0.08, 0.2))
    x0 = torch.randn(shape, generator=torch.Generator().manual_seed(2))
    hc3 = ddrk(integrator="hc3")
    print("\nFlow Matching, DDRK hc3: distance to the data distribution (x1000) by schedule")
    for label, name in (("ddrk_model (model 'simple')", "ddrk_model"),
                        ("ddrk_model_beta", "ddrk_model_beta")):
        cells = []
        for n in CALLS:
            with contextlib.redirect_stdout(io.StringIO()):
                sig = S.get_ddrk_sigmas(name, n, float(ms.sigma_min), 1.0,
                                        torch.device("cpu"), model_sampling=ms)
            out = hc3(KModel(den, ms), x0, sig)
            cells.append(f"{n:3d}: {distribution_error(den, out) * 1e3:6.1f} "
                         f"(last sigma {float(sig[-2]):.3f})")
        print(f"  {label:28s}" + "   ".join(cells))


def main():
    import warnings
    warnings.filterwarnings("ignore")
    accuracy(False, 1.0)
    accuracy(False, 6.0)
    accuracy(True, 1.0)
    accuracy(True, 4.0)
    fidelity(False)
    fidelity(True)
    fm_schedules()


if __name__ == "__main__":
    main()
