"""An exactly solvable diffusion problem, plus ComfyUI-shaped wrappers around it.

Every latent element is drawn independently from a two-component Gaussian
mixture whose means follow a smooth spatial pattern. For that data the
denoiser D(x, sigma) = E[x0 | x_sigma] has a closed form in both
parameterisations:

  EDM / variance exploding:  x = x0 + sigma * eps
  Flow Matching (CONST):     x = (1 - sigma) * x0 + sigma * eps

D is nonlinear in x (the mixture responsibilities are a softmax), so the
probability-flow ODE dx/dsigma = (x - D) / sigma has real curvature and a
sharp "snap" towards a mode at sigma ~ component std, like an image model
resolving detail. Its exact solution is computed with fine RK4 in
lambda = -log(sigma), where dx/dlambda = D - x is smooth.

Everything is float64 so that integrator orders can be measured well above
round-off.
"""

import math
from types import SimpleNamespace

import torch


class MixtureDenoiser:
    """Exact posterior mean for an elementwise 2-component Gaussian mixture."""

    def __init__(self, shape, flow: bool, seed: int = 0,
                 stds=(0.08, 0.20), weight: float = 0.6,
                 dtype=torch.float64):
        self.flow = flow
        g = torch.Generator().manual_seed(seed)
        elem = tuple(shape[1:])            # shared by every batch item
        c = elem[0]
        hw = elem[-2:]
        yy = torch.linspace(-1, 1, hw[0], dtype=dtype).view(-1, 1)
        xx = torch.linspace(-1, 1, hw[1], dtype=dtype).view(1, -1)
        pattern = torch.stack([torch.sin(2.5 * (k + 1) * xx + 1.7 * k * yy)
                               * torch.cos(1.9 * yy - 0.4 * k)
                               for k in range(c)])           # (C, H, W)
        pattern = pattern + 0.3 * torch.randn(pattern.shape, generator=g,
                                              dtype=dtype)
        if len(elem) == 4:                                   # (C, F, H, W)
            f = elem[1]
            drift = torch.linspace(0, 0.3, f, dtype=dtype).view(1, f, 1, 1)
            pattern = pattern.unsqueeze(1) + drift
        self.m1 = 0.9 * pattern
        self.m2 = -0.6 * pattern + 0.25
        self.s1, self.s2 = stds
        self.w1 = weight

    def __call__(self, x: torch.Tensor, sigma) -> torch.Tensor:
        dt = x.dtype
        x = x.to(torch.float64)
        sig = torch.as_tensor(sigma, dtype=torch.float64, device=x.device)
        sig = sig.reshape(-1, *([1] * (x.dim() - 1))) if sig.dim() else sig
        a = (1.0 - sig) if self.flow else torch.ones_like(sig)
        outs, logps = [], []
        for m, s, w in ((self.m1, self.s1, self.w1),
                        (self.m2, self.s2, 1.0 - self.w1)):
            m = m.to(x.device)
            var = a * a * s * s + sig * sig
            logps.append(math.log(w) - 0.5 * torch.log(var)
                         - 0.5 * (x - a * m) ** 2 / var)
            outs.append(m + a * s * s / var * (x - a * m))
        resp = torch.softmax(torch.stack(logps), dim=0)
        return (resp[0] * outs[0] + resp[1] * outs[1]).to(dt)


def reference_solution(den: MixtureDenoiser, x_init: torch.Tensor,
                       sigma_start: float, sigma_end: float = 0.0,
                       n: int = 3000) -> torch.Tensor:
    """Exact ODE solution from sigma_start to sigma_end (0 = the clean sample)."""
    x = x_init.to(torch.float64).clone()
    lam0 = -math.log(sigma_start)
    stop = max(sigma_end, 1e-6)
    lam1 = -math.log(stop)
    h = (lam1 - lam0) / n

    def f(xv, lam):
        return den(xv, math.exp(-lam)) - xv

    lam = lam0
    for _ in range(n):
        k1 = f(x, lam)
        k2 = f(x + 0.5 * h * k1, lam + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, lam + 0.5 * h)
        k4 = f(x + h * k3, lam + h)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        lam += h
    if sigma_end <= 0.0:
        x = den(x, stop)
    return x


class DirectModel:
    """model(x, sigma, **extra_args) -> denoised, as sample_ddrk_omega calls it."""

    def __init__(self, den):
        self.den = den
        self.calls = 0

    def __call__(self, x, sigma, **kwargs):
        self.calls += 1
        return self.den(x, sigma)


class ChainModel(DirectModel):
    """A DirectModel whose model_sampling is reachable the way ComfyUI's
    samplers reach it: model.inner_model.inner_model.model_sampling."""

    def __init__(self, den, model_sampling):
        super().__init__(den)
        self.inner_model = SimpleNamespace(
            inner_model=SimpleNamespace(model_sampling=model_sampling))


def lambda_schedule(sigma_max: float, sigma_end: float, n: int) -> torch.Tensor:
    """n steps uniform in log(sigma), no terminal zero."""
    return torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_end),
                                    n + 1, dtype=torch.float64))


def rmse(a, b) -> float:
    return float(((a.double() - b.double()) ** 2).mean().sqrt())


# --------------------------------------------------------------------------
# ComfyUI-shaped objects for node-level tests.
# --------------------------------------------------------------------------

def make_model_sampling(flow: bool, shift: float = 3.0):
    import comfy.model_sampling as cms
    if flow:
        class MS(cms.ModelSamplingDiscreteFlow, cms.CONST):
            pass
        ms = MS()
        ms.set_parameters(shift=shift)
        return ms

    class MS(cms.ModelSamplingDiscrete, cms.EPS):
        pass
    return MS()


class FakeBaseModel:
    def __init__(self, model_sampling, latent_format, unet_config):
        self.model_sampling = model_sampling
        self.latent_format = latent_format
        self.model_config = SimpleNamespace(unet_config=unet_config)

    def process_latent_in(self, x):
        return x

    def process_latent_out(self, x):
        return x

    def scale_latent_inpaint(self, x, sigma, noise, latent_image, **kwargs):
        # comfy.model_base.BaseModel.scale_latent_inpaint
        return self.model_sampling.noise_scaling(
            sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1)),
            noise, latent_image)


class FakePatcher:
    """Just enough of comfy.model_patcher.ModelPatcher for the DDRK nodes."""

    def __init__(self, flow: bool, shape, seed: int = 0, shift: float = 3.0,
                 image_model=None):
        self.flow = flow
        self.den = MixtureDenoiser(shape, flow, seed=seed)
        ms = make_model_sampling(flow, shift)
        # Channel count follows the test latent, so ComfyUI's
        # fix_empty_latent_channels leaves an all-zero latent alone.
        import comfy.latent_formats as lf
        fmt = lf.LatentFormat()
        fmt.latent_channels = shape[1]
        fmt.latent_dimensions = len(shape) - 2
        unet = {"image_model": image_model} if image_model else (
            {} if flow else {"context_dim": 768})
        self.model = FakeBaseModel(ms, fmt, unet)
        self.load_device = torch.device("cpu")
        self.model_options = {"transformer_options": {}}
        self.calls = 0
        self.seen_shapes = []

    def get_model_object(self, name):
        if name == "model_sampling":
            return self.model.model_sampling
        if name == "latent_format":
            return self.model.latent_format
        raise AttributeError(name)

    def denoise(self, x, sigma):
        self.calls += 1
        self.seen_shapes.append(tuple(x.shape))
        if tuple(x.shape[1:]) != tuple(self.den.m1.shape):
            # A resized canvas (second pass): rebuild the analytic data at
            # the new size so the denoiser still has a defined answer.
            full = (x.shape[0],) + tuple(x.shape[1:])
            self.den = MixtureDenoiser(full, self.flow)
        return self.den(x, sigma)


class _Guider:
    def __init__(self, patcher, cfg):
        self.model_patcher = patcher
        self.inner_model = patcher.model
        self.cfg = cfg

    def __call__(self, x, sigma, model_options=None, seed=None):
        return self.model_patcher.denoise(x, sigma)


def fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                       latent_image, noise_mask=None, callback=None,
                       disable_pbar=False, seed=None):
    """comfy.sample.sample_custom without conditioning or model loading.

    It still runs ComfyUI's real KSAMPLER.sample, so noise scaling, the
    KSamplerX0Inpaint wrapper and inverse noise scaling are the production
    code paths.
    """
    guider = _Guider(model, cfg)
    denoise_mask = None
    if noise_mask is not None:
        import comfy.sampler_helpers
        denoise_mask = comfy.sampler_helpers.prepare_mask(
            noise_mask, noise.shape, torch.device("cpu"))
    extra_args = {"model_options": model.model_options, "seed": seed}
    out = sampler.sample(guider, sigmas, extra_args, callback, noise,
                         latent_image, denoise_mask, disable_pbar)
    return out.to(torch.float32)
