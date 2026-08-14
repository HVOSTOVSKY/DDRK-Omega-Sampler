"""
DDRK Omega Sampler v1.6.0
ComfyUI | Flow Matching + EDM Universal Sampler
https://github.com/HVOSTOVSKY/DDRK-Omega-Sampler
"""

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import math
import time
import json
import csv
import os
from collections import OrderedDict
from tqdm.auto import trange
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class SamplerState:
    d_prev: Optional[torch.Tensor] = None
    step_count: int = 0
    total_steps: int = 0
    is_edm: bool = False
    prev_denoised: Optional[torch.Tensor] = None
    prev_sigma: float = 0.0


    curvature: float = 0.0


    prev_dt: Optional[float] = None


    hc2_D_prev: Optional[torch.Tensor] = None
    hc2_lambda_prev: Optional[float] = None

    hc2_D_prev2: Optional[torch.Tensor] = None
    hc2_lambda_prev2: Optional[float] = None


    hc2_order_used: int = 0
    hc2_err_ratio: float = -1.0


    hc2_activity: float = -1.0
    hc2_corrector_fired: bool = False


    hc2_limited_frac: float = -1.0


class DeviceDtypeGuard:
    def __init__(self, target_dtype, target_device):
        self.dtype = target_dtype
        self.device = target_device

    def __call__(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.to(device=self.device, dtype=self.dtype)


def _detect_model_profile(model, latent_samples=None) -> dict:
    profile = {
        "family": "unknown",
        "scheduler_type": "ddrk_auto",
        "flow_shift": 3.0,
        "integrator": "auto",
        "sde_strength": 0.08,
        "sharpness": 0.30,
        "saber_fusion": 0.30,
        "momentum_beta": 0.25,
        "guidance_embed": False,
        "hint": "Manual tuning required for steps/cfg. See checkpoint card or community docs.",
    }


    is_edm = True
    try:
        ms = model.get_model_object("model_sampling")
        is_edm = float(ms.sigma_max) > 5.0
    except Exception:
        pass


    latent_ch = 4
    if latent_samples is not None:
        try:
            latent_ch = int(latent_samples.shape[1])
        except Exception:
            pass


    family = "unknown"
    image_model = None
    guidance_embed = False
    try:

        inner_model = getattr(model, "model", None)
        cfg = getattr(inner_model, "model_config", {}) if inner_model is not None else {}
        unet = {}
        if isinstance(cfg, dict):
            unet = cfg.get("unet_config", {})
        elif hasattr(cfg, "unet_config"):
            unet_cfg = cfg.unet_config
            unet = unet_cfg() if callable(unet_cfg) else unet_cfg
            if not isinstance(unet, dict):
                unet = dict(unet) if hasattr(unet, '__dict__') else {}

        if isinstance(unet, dict):
            image_model = unet.get("image_model")
            guidance_embed = unet.get("guidance_embed", False)
            adm = unet.get("adm_in_channels", 0)
            ctx = unet.get("context_dim", 0)

            if image_model is not None:

                im = str(image_model).lower()
                if "flux" in im:
                    family = "flux"
                elif "sdxl" in im:
                    family = "sdxl"
                elif "sd1" in im or "sd15" in im:
                    family = "sd15"
                elif "sd2" in im:
                    family = "sd2"
                elif "sd3" in im:
                    family = "sd3"
                elif "qwen" in im:
                    family = "qwen"
                elif "krea" in im:
                    family = "krea"
                elif "hidream" in im or "hi_dream" in im or "hidd" in im:
                    family = "hidream"
                elif "chroma" in im:
                    family = "chroma"
                elif "lumina" in im:
                    family = "lumina"
                else:
                    family = "fm"
            else:

                if adm == 2816:
                    family = "sdxl"
                elif ctx == 768:
                    family = "sd15"
                elif ctx == 1024:
                    family = "sd2"
                elif ctx in (2048, 4096):
                    family = "flux"
    except Exception:
        pass


    if family == "unknown":
        if is_edm:
            family = "edm"
        elif latent_ch == 16:
            family = "flux"
        else:
            family = "fm"


    print(f"[DDRK Detect] raw_image_model={image_model!r}, family={family}, "
          f"guidance_embed={guidance_embed}, is_edm={is_edm}, latent_ch={latent_ch}")


    if family in ("sd15", "sd2"):
        profile.update(
            scheduler_type="ddrk_edm_karras",
            flow_shift=3.0,
            integrator="auto",
            sde_strength=0.0,


            sharpness=0.0,
            saber_fusion=0.20,
            momentum_beta=0.25,
            hint="SD1.5/SD2: EDM. Sharpen is EDM-disabled by design, so sharpness has no effect here. Steps/CFG depend on checkpoint (base 20-30 / 7-8). Use turbo/distilled LoRA for 4-8 steps / CFG 1-2.",
        )
    elif family == "sdxl":
        profile.update(
            scheduler_type="ddrk_edm_karras",
            flow_shift=3.0,
            integrator="auto",
            sde_strength=0.0,

            sharpness=0.0,
            saber_fusion=0.20,
            momentum_beta=0.20,
            hint="SDXL: EDM. Sharpen is EDM-disabled by design, so sharpness has no effect here. Steps/CFG depend on checkpoint (base 20-30 / 7-8). Use turbo/distilled for 4-8 steps / CFG 1-2.",
        )
    elif family in ("flux", "sd3", "qwen", "krea", "hidream", "chroma", "lumina"):
        profile.update(
            scheduler_type="ddrk_auto",
            flow_shift=1.0,


            integrator="auto",
            sde_strength=0.0,
            sharpness=0.10,
            saber_fusion=0.0,
            momentum_beta=0.0,
            hint=(f"{family.upper()}: Flow Matching. Steps/CFG vary wildly by checkpoint. "
                  f"{'Guidance embed detected — distilled variant, try CFG≈1.0, steps 4-8. ' if guidance_embed else ''}"
                  f"Check your model card. Integrator note: at a fixed model-call "
                  f"budget heun/rk4 beat euler on FM in testing — heun at N steps "
                  f"costs about the same as euler at 2N and looked better."),
        )
    elif family == "fm":
        profile.update(
            scheduler_type="ddrk_auto",
            flow_shift=1.5,

            integrator="auto",
            sde_strength=0.0,
            sharpness=0.12,
            saber_fusion=0.0,
            momentum_beta=0.0,
            hint=(f"Generic FM: Steps/CFG vary by checkpoint. Start with 8-20 steps, CFG 1-4. "
                  f"{'Guidance embed detected — distilled variant, try CFG≈1.0. ' if guidance_embed else ''}"
                  f"Check model card."),
        )
    elif family == "edm":
        profile.update(
            scheduler_type="ddrk_edm_karras",
            flow_shift=3.0,
            integrator="auto",
            sde_strength=0.0,

            sharpness=0.0,
            saber_fusion=0.20,
            momentum_beta=0.20,
            hint="Generic EDM: Steps/CFG depend on checkpoint (base 20-30 / 7-8). Use turbo/distilled for fewer steps / lower CFG. Sharpen is EDM-disabled by design, so sharpness has no effect here.",
        )

    profile["family"] = family
    profile["guidance_embed"] = guidance_embed
    return profile


def dynamic_threshold(denoised: torch.Tensor, sigma: float, sigma_max: float,
                      percentile: float = 0.995, min_val: float = 1.0,
                      edm_ratio: float = 0.30) -> torch.Tensor:
    threshold_ratio = 0.40 if sigma_max <= 5.0 else edm_ratio
    if sigma > threshold_ratio * sigma_max or denoised.numel() == 0:
        return denoised

    flat = denoised.reshape(denoised.shape[0], -1)
    abs_flat = flat.abs()
    max_val = abs_flat.max(dim=1, keepdim=True)[0]


    if max_val.max().item() <= min_val:
        return denoised

    s = torch.quantile(abs_flat, percentile, dim=1, keepdim=True)
    s = torch.clamp(s, min=min_val)
    s = s.reshape(denoised.shape[0], *[1] * (denoised.dim() - 1))
    ratio = s / denoised.abs().clamp_min(1e-8)
    scale = torch.where(ratio < 1.0, ratio, torch.ones_like(ratio))
    return denoised * scale


def _soft_clamp(t: torch.Tensor, bound: float = 5.0, softness: float = 0.25) -> torch.Tensor:
    core = torch.clamp(t, -bound, bound)
    excess = t - core
    return core + softness * excess


def _zscore_gate(t: torch.Tensor, gain: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    std, mean = torch.std_mean(t)
    z = (t - mean) / std.clamp_min(eps)
    return torch.sigmoid(z * gain)


class LoGMask:
    def __init__(self, max_cache: int = 4):
        self._kernels: OrderedDict = OrderedDict()
        self._max_cache = max_cache
        self._samplers: Dict[str, torch.Generator] = {}

    def _sample_gen(self, device) -> torch.Generator:
        key = str(device)
        gen = self._samplers.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(991127)
            self._samplers[key] = gen
        return gen

    def __call__(self, x: torch.Tensor, edge_sigma: float = 3.0) -> torch.Tensor:
        is_5d = x.dim() == 5
        if is_5d:
            b, c, f, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)

        key = (x.dtype, str(x.device))
        if key not in self._kernels:
            k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                             dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            self._kernels[key] = k
            if len(self._kernels) > self._max_cache:
                self._kernels.popitem(last=False)
        else:
            k = self._kernels[key]

        c = x.shape[1]
        kernel = k.repeat(c, 1, 1, 1)
        lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), kernel, groups=c)
        edge = lap.abs()


        flat_edge = edge.reshape(edge.shape[0], -1)
        n = flat_edge.shape[1]
        if n > 1_000_000:


            idx = torch.randint(0, n, (1_000_000,), device=edge.device,
                                generator=self._sample_gen(edge.device))
            sample = flat_edge[:, idx]
        else:
            sample = flat_edge
        med = sample.float().median(dim=1, keepdim=True)[0]
        s_bg = (med / 0.6745).clamp_min(1e-8)
        cutoff = (edge_sigma * s_bg).to(edge.dtype).view(-1, 1, 1, 1)
        mask = (edge > cutoff).float()
        mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)

        if is_5d:
            mask = mask.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
        return mask


def local_entropy_mask(x: torch.Tensor, window: int = 3) -> torch.Tensor:
    is_5d = x.dim() == 5
    if is_5d:
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    else:
        b = c = f = h = w = None

    pad = window // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    mu = F.avg_pool2d(x_pad, window, stride=1)
    mu_sq = F.avg_pool2d(x_pad ** 2, window, stride=1)
    var = (mu_sq - mu ** 2).clamp_min(0.0)
    flatness = 1.0 - _zscore_gate(var)

    if is_5d:
        flatness = flatness.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
    return flatness


class SABER2:
    def __init__(self, mode: str = "auto", buffer_size: int = 3,
                 fusion: float = 0.35, ema_decay: float = 0.7, use_ema: bool = True,
                 max_keys: int = 4, content_aware: bool = True):
        self.mode = mode
        self.buffer_size = buffer_size
        self.fusion = fusion
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self._max_keys = max_keys
        self.content_aware = content_aware
        self._buffers: OrderedDict[str, Any] = OrderedDict()

    def _is_video(self, x: torch.Tensor) -> bool:
        if self.mode == "video":
            return True
        if self.mode == "image":
            return False
        return x.dim() == 5 and x.shape[2] > 1

    def reset(self):
        self._buffers.clear()

    def fuse(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion <= 0:
            return x
        if x.dim() not in (4, 5):
            raise ValueError(f"SABER2 expects 4D or 5D input, got {x.dim()}D")

        is_vid = self._is_video(x)
        b = x.shape[0]
        dev_idx = x.device.index if x.device.index is not None else -1
        key = f"buf_{b}_{'_'.join(map(str, x.shape))}_{x.device.type}_{dev_idx}"

        if key not in self._buffers:
            self._buffers[key] = {"frames": [], "ema": None}
            if len(self._buffers) > self._max_keys:
                self._buffers.popitem(last=False)
        buf = self._buffers[key]

        buf["frames"].append(x.detach().clone())
        if len(buf["frames"]) > self.buffer_size:
            buf["frames"].pop(0)

        if is_vid and len(buf["frames"]) < 2:
            return x

        fusion_weight = self.fusion
        if self.content_aware and not is_vid:
            if x.dim() == 4:
                k3 = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
                k3 = k3.repeat(x.shape[1], 1, 1, 1)
                log3 = F.conv2d(F.pad(x, (1,1,1,1), mode='reflect'), k3, groups=x.shape[1]).abs()
                blur5 = F.avg_pool2d(F.pad(x, (2,2,2,2), mode='reflect'), 5, stride=1)
                edge5 = (x - blur5).abs()
                combined = (log3 + edge5) * 0.5
                edge_density = _zscore_gate(combined)
                fusion_weight = self.fusion * (1.0 - edge_density)
            else:
                if x.shape[2] == 1:
                    x_4d = x.squeeze(2)
                    k3 = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
                    k3 = k3.repeat(x_4d.shape[1], 1, 1, 1)
                    log3 = F.conv2d(F.pad(x_4d, (1,1,1,1), mode='reflect'), k3, groups=x_4d.shape[1]).abs()
                    blur5 = F.avg_pool2d(F.pad(x_4d, (2,2,2,2), mode='reflect'), 5, stride=1)
                    edge5 = (x_4d - blur5).abs()
                    combined = (log3 + edge5) * 0.5
                    edge_density = _zscore_gate(combined)
                    fusion_weight = self.fusion * (1.0 - edge_density)
                    fusion_weight = fusion_weight.unsqueeze(2)
                else:
                    blur3 = F.avg_pool3d(F.pad(x, (1,1,1,1,1,1), mode='reflect'), 3, stride=1)
                    edge3 = (x - blur3).abs()
                    edge_density = _zscore_gate(edge3)
                    fusion_weight = self.fusion * (1.0 - edge_density)

        if is_vid:
            stacked = torch.stack(buf["frames"], dim=0)
            var = torch.var(stacked, dim=0)
            chaos = _zscore_gate(var)
            if self.use_ema:
                if buf["ema"] is None:
                    buf["ema"] = x.detach().clone()
                else:
                    buf["ema"] = self.ema_decay * buf["ema"] + (1 - self.ema_decay) * x.detach()
                fused = x * (1.0 - chaos * fusion_weight) + buf["ema"] * (chaos * fusion_weight)
            else:
                avg = torch.mean(stacked, dim=0)
                fused = x * (1.0 - chaos * fusion_weight) + avg * (chaos * fusion_weight)
            return fused
        else:
            if x.dim() == 4:
                blurred = F.avg_pool2d(
                    F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, stride=1
                )
            else:
                if x.shape[2] == 1:
                    x_4d = x.squeeze(2)
                    blurred_4d = F.avg_pool2d(
                        F.pad(x_4d, (1, 1, 1, 1), mode='reflect'), 3, stride=1
                    )
                    blurred = blurred_4d.unsqueeze(2)
                else:
                    blurred = F.avg_pool3d(
                        F.pad(x, (1, 1, 1, 1, 1, 1), mode='reflect'), 3, stride=1
                    )
            if isinstance(fusion_weight, float):
                w = fusion_weight
                return x * (1.0 - w) + blurred * w
            else:
                return x * (1.0 - fusion_weight) + blurred * fusion_weight


def perceptual_sharpen(x: torch.Tensor, strength: float = 0.35,
                       is_final_step: bool = False) -> torch.Tensor:
    if strength <= 0:
        return x

    is_5d = x.dim() == 5
    if is_5d:
        b, c, f, h, w = x.shape
        x_4d = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    else:
        x_4d = x

    blur_3x3 = F.avg_pool2d(F.pad(x_4d, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
    detail_fine = x_4d - blur_3x3

    blur_5x5 = F.avg_pool2d(F.pad(x_4d, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
    detail_coarse = x_4d - blur_5x5

    if not is_final_step:
        mask = local_entropy_mask(x_4d, window=3)
        edge_boost = 1.0 - mask
        detail_fine = detail_fine * edge_boost
        detail_coarse = detail_coarse * edge_boost

    detail_fine = torch.clamp(detail_fine, min=-0.6, max=0.6)
    detail_coarse = torch.clamp(detail_coarse, min=-0.6, max=0.6)

    out = x_4d + strength * (0.6 * detail_fine + 0.4 * detail_coarse)

    if is_5d:
        out = out.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
    return out


def _safe_sigma(s: Union[float, torch.Tensor]) -> float:
    return max(float(s), 1e-8)


def euler_step(x, sigma, sigma_next, model_fn, state: SamplerState, momentum_beta: float = None):
    denoised = model_fn(x, sigma)
    d = (x - denoised) / _safe_sigma(sigma)
    dt = sigma_next - sigma
    d = _ab2_extrapolate(d, state, dt=float(dt), beta=momentum_beta)
    return x + d * dt, denoised, d


def heun_step(x, sigma, sigma_next, model_fn, state: SamplerState, momentum_beta: float = None):
    denoised = model_fn(x, sigma)
    d = (x - denoised) / _safe_sigma(sigma)
    dt = sigma_next - sigma
    x_next = x + d * dt

    if float(sigma_next) > 1e-7:
        denoised_2 = model_fn(x_next, sigma_next)
        d2 = (x_next - denoised_2) / _safe_sigma(sigma_next)
        d_avg = (d + d2) * 0.5
        d_avg = _ab2_extrapolate(d_avg, state, dt=float(dt), force=True, beta=momentum_beta)
        x_next = x + d_avg * dt
        return x_next, denoised_2, d_avg

    _ab2_extrapolate(d, state, dt=float(dt), beta=momentum_beta)
    return x_next, denoised, d


def rk4_step(x, sigma, sigma_next, model_fn, state: SamplerState, momentum_beta: float = None):
    dt = sigma_next - sigma
    s = _safe_sigma(sigma)

    if float(sigma_next) <= 1e-7:


        denoised = model_fn(x, sigma)
        d = (x - denoised) / s
        x_next = x + d * dt
        _ab2_extrapolate(d, state, dt=float(dt), beta=momentum_beta)
        return x_next, denoised, d

    s_mid = _safe_sigma(sigma + dt * 0.5)
    s_next = _safe_sigma(sigma_next)

    denoised_1 = model_fn(x, sigma)
    d1 = (x - denoised_1) / s

    x_k2 = x + d1 * (dt * 0.5)
    denoised_2 = model_fn(x_k2, sigma + dt * 0.5)
    d2 = (x_k2 - denoised_2) / s_mid

    x_k3 = x + d2 * (dt * 0.5)
    denoised_3 = model_fn(x_k3, sigma + dt * 0.5)
    d3 = (x_k3 - denoised_3) / s_mid

    x_k4 = x + d3 * dt
    denoised_4 = model_fn(x_k4, sigma_next)
    d4 = (x_k4 - denoised_4) / s_next

    d_final = (d1 + 2 * d2 + 2 * d3 + d4) / 6.0
    d_final = _ab2_extrapolate(d_final, state, dt=float(dt), force=True, beta=momentum_beta)
    return x + d_final * dt, denoised_4, d_final


def hc2_step(x, sigma, sigma_next, model_fn, state: SamplerState,
             momentum_beta: float = None, limiter_kappa: float = 1.0,
             max_order: int = 2, corrector_thresh: float = 0.0):
    s = _safe_sigma(sigma)
    sigma_next_f = float(sigma_next)

    denoised = model_fn(x, sigma)
    lam = -math.log(s)

    if sigma_next_f <= 1e-7:


        state.hc2_D_prev2, state.hc2_lambda_prev2 = state.hc2_D_prev, state.hc2_lambda_prev
        state.hc2_D_prev = denoised.detach().clone()
        state.hc2_lambda_prev = lam
        state.hc2_limited_frac = -1.0
        state.hc2_order_used = 1
        state.hc2_err_ratio = -1.0
        state.hc2_activity = -1.0
        state.hc2_corrector_fired = False
        d = (x - denoised) / s
        return denoised, denoised, d

    h = math.log(s / _safe_sigma(sigma_next))
    exp_neg_h = math.exp(-h)

    first_order = (1.0 - exp_neg_h) * (denoised - x)
    x_next = x + first_order
    order_used = 1
    state.hc2_limited_frac = -1.0
    state.hc2_err_ratio = -1.0
    state.hc2_activity = -1.0
    state.hc2_corrector_fired = False

    has_1 = state.hc2_D_prev is not None and state.hc2_lambda_prev is not None
    h_prev = (lam - state.hc2_lambda_prev) if has_1 else 0.0

    if max_order >= 3 and not has_1:


        denoised_2 = model_fn(x_next, sigma_next)
        d1 = (x - denoised) / s
        d2 = (x_next - denoised_2) / _safe_sigma(sigma_next)
        x_next = x + (d1 + d2) * 0.5 * (float(sigma_next) - float(sigma))
        order_used = 2

    elif has_1 and abs(h_prev) > 1e-8:
        r = (denoised - state.hc2_D_prev) / h_prev
        corr2 = r * ((h - 1.0) + exp_neg_h)
        correction = corr2
        order_used = 2

        has_2 = (max_order >= 3 and state.hc2_D_prev2 is not None
                 and state.hc2_lambda_prev2 is not None)
        if has_2:
            h_prev2 = state.hc2_lambda_prev - state.hc2_lambda_prev2
            if abs(h_prev2) > 1e-8:


                d_prev = (state.hc2_D_prev - state.hc2_D_prev2) / h_prev2
                dd = (r - d_prev) / (h_prev + h_prev2)
                corr3 = dd * ((h * h - 2.0 * h + 2.0) - 2.0 * exp_neg_h
                              + h_prev * ((h - 1.0) + exp_neg_h))


                try:
                    n3 = float(corr3.abs().mean().item())
                    n2 = float(corr2.abs().mean().item())
                    ratio = n3 / max(n2, 1e-12)
                except Exception:
                    ratio = float('inf')
                state.hc2_err_ratio = ratio
                if ratio < 1.0:
                    correction = corr2 + corr3
                    order_used = 3

        bound = limiter_kappa * first_order.abs()
        limited = torch.clamp(correction, -bound, bound)
        try:
            state.hc2_limited_frac = float(
                (correction.abs() > bound).float().mean().item())
            state.hc2_activity = float(
                correction.abs().mean().item()
                / max(float(first_order.abs().mean().item()), 1e-12))
        except Exception:
            state.hc2_limited_frac = -1.0
        x_next = x_next + limited

        if corrector_thresh > 0.0 and state.hc2_activity > corrector_thresh:


            denoised_c = model_fn(x_next, sigma_next)
            r_c = (denoised_c - denoised) / h
            corr_c = r_c * ((h - 1.0) + exp_neg_h)
            corr_c = torch.clamp(corr_c, -bound, bound)
            x_next = x + first_order + corr_c
            state.hc2_corrector_fired = True

    state.hc2_order_used = order_used
    state.hc2_D_prev2, state.hc2_lambda_prev2 = state.hc2_D_prev, state.hc2_lambda_prev
    state.hc2_D_prev = denoised.detach().clone()
    state.hc2_lambda_prev = lam

    d = (x - denoised) / s
    state.d_prev = d.detach().clone()
    state.prev_dt = float(sigma_next) - float(sigma)
    return x_next, denoised, d


def _ab2_extrapolate(d: torch.Tensor, state: SamplerState, dt: float,
                     force: bool = False, beta: float = 0.25,
                     momentum_beta: Optional[float] = None) -> torch.Tensor:
    if momentum_beta is not None:
        beta = momentum_beta

    if state.d_prev is None or state.prev_dt is None or abs(state.prev_dt) < 1e-8 or beta <= 0.0:
        state.d_prev = d.detach().clone()
        state.prev_dt = dt
        return d

    ratio = beta * (dt / (2.0 * state.prev_dt))
    if force:
        ratio *= 0.5
    ratio = max(-1.0, min(1.0, ratio))

    d_extrap = d + ratio * (d - state.d_prev)

    state.d_prev = d.detach().clone()
    state.prev_dt = dt
    return d_extrap


class AdaptivePhaseRouter:
    def __init__(self, total_steps: int, sigma_max: float, is_edm: bool,
                 p1_ratio: float = 0.65, p2_ratio: float = 0.25):
        self.total_steps = total_steps
        self.sigma_max = sigma_max
        self.is_edm = is_edm

        if total_steps <= 2:
            self.p1_end = total_steps
            self.p2_end = total_steps
            self.p3_start = total_steps
            return

        if is_edm:
            p1_ratio = max(p1_ratio, 0.60)
            p2_ratio = min(p2_ratio, 0.22)
        else:
            p1_ratio = min(p1_ratio, 0.55)
            p2_ratio = max(p2_ratio, 0.35)

        p1 = max(1, min(total_steps - 2, round(total_steps * p1_ratio)))
        p2 = max(1, min(total_steps - p1 - 1, round(total_steps * p2_ratio)))
        p3 = total_steps - p1 - p2
        if p3 < 1:
            deficit = 1 - p3
            p1 = max(1, p1 - (deficit + 1) // 2)
            p2 = max(1, p2 - deficit // 2)
            p3 = total_steps - p1 - p2
            if p3 < 1:
                p2 = max(1, total_steps - p1 - 1)
                p3 = total_steps - p1 - p2

        self.p1_end = p1
        self.p2_end = p1 + p2
        self.p3_start = self.p2_end

    def get_phase(self, step_idx: int) -> int:
        if step_idx < self.p1_end:
            return 1
        elif step_idx < self.p2_end:
            return 2
        return 3

    def pick_integrator(self, step_idx: int, sigma: float, cfg: str,
                        state: SamplerState = None) -> str:
        phase = self.get_phase(step_idx)
        if cfg == "auto":
            if phase == 1:
                if state is not None and state.prev_denoised is not None:
                    with torch.no_grad():


                        curvature = getattr(state, 'curvature', 0.0)
                        if curvature > 0.15 and self.total_steps >= 10:
                            return "rk4"
                        elif curvature > 0.05:
                            return "heun"
                        else:
                            return "euler"
                if not self.is_edm and self.total_steps >= 10:
                    if sigma > 0.3 * self.sigma_max:
                        return "rk4"
                    return "heun"
                elif self.is_edm:
                    return "heun"
                else:
                    return "heun"
            elif phase == 2:
                return "heun" if self.total_steps >= 8 else "euler"
            else:
                return "euler"
        return cfg


def _ancestral_split(sigma: float, sigma_next: float, eta: float) -> Tuple[float, float]:
    if sigma <= 1e-9 or sigma_next <= 1e-9 or eta <= 0.0:
        return sigma_next, 0.0
    eta = min(eta, 1.0)
    var_diff = max(sigma ** 2 - sigma_next ** 2, 0.0)
    sigma_up = min(sigma_next, eta * math.sqrt((sigma_next ** 2) * var_diff) / sigma)
    sigma_down = math.sqrt(max(sigma_next ** 2 - sigma_up ** 2, 0.0))
    return sigma_down, sigma_up


class AdaptiveSDE:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._gen: Optional[torch.Generator] = None

    def get_generator(self, device) -> torch.Generator:
        if self._gen is None or str(self._gen.device) != str(device):
            self._gen = torch.Generator(device=device)
            if self.seed is not None:
                self._gen.manual_seed(int(self.seed))
            else:
                derived = int(torch.randint(0, 2 ** 62, (1,)).item())
                self._gen.manual_seed(derived)
        return self._gen


    def _get_gen(self, device) -> torch.Generator:
        return self.get_generator(device)

    def __call__(self, x: torch.Tensor, sigma_up: float, mask: torch.Tensor) -> torch.Tensor:
        if sigma_up <= 1e-9:
            return torch.zeros_like(x)
        noise = torch.randn_like(x, generator=self.get_generator(x.device))
        return noise * sigma_up * mask


def _flow_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    if abs(shift - 1.0) < 1e-4:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def get_ddrk_sigmas(scheduler_type: str, steps: int, sigma_min: float,
                    sigma_max: float, device: torch.device,
                    flow_shift: float = 3.0, warmup_steps: int = 0,
                    beta_a: float = 2.0, beta_b: float = 1.0,
                    auto_optimize: bool = True) -> torch.Tensor:
    is_edm = sigma_max > 5.0

    if scheduler_type == "ddrk_anima":


        print("[DDRK] 'ddrk_anima' is not a real schedule and has been removed "
              "from the UI; using ddrk_flow_linear, which is what it always did.")
        scheduler_type = "ddrk_flow_linear"

    if scheduler_type == "ddrk_auto":
        if is_edm:
            scheduler_type = "ddrk_edm_karras"
        else:
            if auto_optimize and steps <= 10:
                scheduler_type = "ddrk_flow_linear"
                flow_shift = min(flow_shift, 1.0)
            elif auto_optimize and steps <= 20:
                scheduler_type = "ddrk_cosine"
                flow_shift = min(flow_shift, 2.0)
            else:
                scheduler_type = "ddrk_cosine"

    if scheduler_type.startswith("ddrk_") and scheduler_type not in (
        "ddrk_edm_karras", "ddrk_edm_simple", "ddrk_edm_poly"
    ):


        n_pts = steps + 1
        if scheduler_type == "ddrk_cosine":
            angles = torch.linspace(0, math.pi / 2, n_pts, device=device)
            t = torch.cos(angles)
            t_shifted = _flow_shift(t, flow_shift)
            sigmas = t_shifted
        elif scheduler_type == "ddrk_beta":


            t_raw = torch.linspace(0, 1, n_pts, device=device)
            a = max(float(beta_a), 1e-3)
            b = max(float(beta_b), 1e-3)
            t_kuma = (1.0 - (1.0 - t_raw) ** (1.0 / b)) ** (1.0 / a)
            t_beta = 1.0 - t_kuma
            t_shifted = _flow_shift(t_beta, flow_shift)
            sigmas = t_shifted
        elif scheduler_type == "ddrk_fewstep":
            shift_eff = max(flow_shift, 5.0)
            t = torch.linspace(1.0, 0.0, n_pts, device=device)
            sigmas = _flow_shift(t, shift_eff)
        elif scheduler_type == "ddrk_flow_cosmos":
            t = torch.linspace(1.0, 0.0, n_pts, device=device)
            t_shifted = _flow_shift(t, flow_shift)
            weight = torch.sigmoid((0.3 - t_shifted) * 20.0)
            sig = torch.sigmoid((t_shifted - 0.3) * -5.0)
            t_adj = t_shifted * (1.0 - weight * sig * 0.15)
            sigmas = t_adj
        else:
            t = torch.linspace(1.0, 0.0, n_pts, device=device)
            sigmas = _flow_shift(t, flow_shift)
            if steps > 4 and warmup_steps > 0:
                w = min(warmup_steps, max(1, steps // 8))
                for i in range(1, w + 1):
                    sigmas[i] *= 1.0 + 0.02 * (1.0 - (i - 1) / max(w, 1))
                for i in range(1, w + 1):
                    sigmas[i] = min(sigmas[i], sigmas[i - 1] - 1e-7)
            sigmas[0] = 1.0


        sigmas = sigmas.clone()
        sigmas[-1] = 0.0
        return sigmas

    elif scheduler_type.startswith("ddrk_edm"):
        if scheduler_type == "ddrk_edm_karras":
            rho = 7.0
            ramp = torch.linspace(0, 1, steps, device=device)
            sigmas = (sigma_max ** (1.0 / rho) +
                      ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
            sigmas = torch.clamp(sigmas, min=sigma_min)
            sigmas = torch.cat([sigmas, torch.tensor([0.0], device=device)])
            return sigmas
        elif scheduler_type == "ddrk_edm_poly":
            ramp = torch.linspace(0, 1, steps, device=device)
            sigmas = sigma_max * (1.0 - ramp ** 2) + sigma_min * (ramp ** 2)
            sigmas = torch.where(sigmas < sigma_min,
                                 torch.tensor(sigma_min, device=device), sigmas)
            sigmas = torch.cat([sigmas, torch.tensor([0.0], device=device)])
            return sigmas
        else:
            ramp = torch.linspace(0, 1, steps, device=device)
            sigmas = sigma_max * (sigma_min / sigma_max) ** ramp
            sigmas = torch.cat([sigmas, torch.tensor([0.0], device=device)])
            return sigmas

    if sigma_max > 5.0:
        ramp = torch.linspace(0, 1, steps + 1, device=device)
        return sigma_max * (sigma_min / sigma_max) ** ramp
    else:


        t = torch.linspace(1.0, 0.0, steps + 1, device=device)
        s = _flow_shift(t, flow_shift)
        s = s.clone()
        s[-1] = 0.0
        return s


def _resolve_effective_params(total_steps: int, is_edm: bool, auto_optimize: bool,
                              integrator: str, sde_strength: float, sharpness: float,
                              saber_fusion: float, momentum_beta: float
                              ) -> Tuple[str, float, float, float, float]:
    debug_lines = []

    if auto_optimize and not is_edm:
        if total_steps <= 10:
            saber_fusion = 0.0
            sde_strength = 0.0
            momentum_beta = 0.0
            sharpness = min(sharpness, 0.12)
            if integrator == "auto":
                integrator = "euler"
            debug_lines.append(
                f"[DDRK Auto] FM few-step ({total_steps} steps): "
                f"SABER=0, SDE=0, momentum=0, sharp={sharpness:.2f}, "
                f"integrator={integrator}, scheduler=linear, shift<=1.0")
        elif total_steps <= 20:
            saber_fusion = min(saber_fusion, 0.05)
            sde_strength = 0.0
            momentum_beta = min(momentum_beta, 0.10)
            sharpness = min(sharpness, 0.15)
            debug_lines.append(
                f"[DDRK Auto] FM mid-step ({total_steps} steps): "
                f"SABER<=0.05, SDE=0, momentum<=0.10, sharp<=0.15")

    if total_steps <= 6 and integrator != "hc2":


        if integrator != "euler":
            debug_lines.append(
                f"[DDRK Auto] {total_steps} steps (<=6): integrator '{integrator}' "
                f"overridden to 'euler'. Higher-order integrators need more steps "
                f"than this to pay for their extra model calls. Use >=7 steps to "
                f"keep your choice.")
        integrator = "euler"
        saber_fusion = min(saber_fusion, 0.15)
        sharpness = min(sharpness, 0.2)
        sde_strength = 0.0

    if not is_edm:


        calls_per_step = {"rk4": 4, "heun": 2}.get(integrator, 1)
        est_calls = total_steps * calls_per_step
        sharp_cap = 0.12 if est_calls <= 10 else 0.15
        if sharpness > sharp_cap:
            debug_lines.append(
                f"[DDRK Auto] FM at ~{est_calls} model calls: sharpness "
                f"{sharpness:.3f} capped to {sharp_cap:.2f}.")
        sharpness = min(sharpness, sharp_cap)
        if saber_fusion > 0.15:
            debug_lines.append(
                f"[DDRK Auto] FM: saber_fusion {saber_fusion:.3f} capped to 0.15.")
        saber_fusion = min(saber_fusion, 0.15)
        if momentum_beta > 0.15:
            debug_lines.append(
                f"[DDRK Auto] FM: momentum_beta {momentum_beta:.3f} capped to 0.15.")
        momentum_beta = min(momentum_beta, 0.15)
    elif total_steps <= 10 and integrator == "rk4":
        debug_lines.append(
            f"[DDRK Auto] EDM at {total_steps} steps (<=10): integrator 'rk4' "
            f"overridden to 'heun'.")
        integrator = "heun"

    for line in debug_lines:
        print(line)

    return integrator, sde_strength, sharpness, saber_fusion, momentum_beta


def _integrator_step(chosen_integrator: str, x: torch.Tensor, sigma_curr, sigma_target,
                     model_fn, state: SamplerState, momentum_beta: float,
                     limiter_kappa: float = 1.0, hc2_max_order: int = 2,
                     hc2_corrector: float = 0.0):
    if chosen_integrator == "hc2":
        return hc2_step(x, sigma_curr, sigma_target, model_fn, state,
                        momentum_beta=momentum_beta, limiter_kappa=limiter_kappa,
                        max_order=hc2_max_order, corrector_thresh=hc2_corrector)
    elif chosen_integrator == "rk4":
        return rk4_step(x, sigma_curr, sigma_target, model_fn, state, momentum_beta=momentum_beta)
    elif chosen_integrator == "heun":
        return heun_step(x, sigma_curr, sigma_target, model_fn, state, momentum_beta=momentum_beta)
    else:
        return euler_step(x, sigma_curr, sigma_target, model_fn, state, momentum_beta=momentum_beta)


class _DDRKDebugRecorder:

    def __init__(self, tag: str = ""):
        self.tag = tag
        self.meta: Dict[str, Any] = {}
        self.steps: List[Dict[str, Any]] = []
        self._t_last: Optional[float] = None

    def start_step(self):
        self._t_last = time.perf_counter()

    def log_step(self, **fields):
        try:
            if self._t_last is not None:
                fields["wall_ms"] = round((time.perf_counter() - self._t_last) * 1000, 2)
            self.steps.append(fields)
        except Exception as e:
            self.steps.append({"step": fields.get("step"), "log_error": str(e)})

    def _summary_text(self) -> str:
        lines = ["DDRK Omega Sampler — Debug Summary", "=" * 40]
        for k, v in self.meta.items():
            if k == "sigma_schedule":
                continue
            lines.append(f"{k}: {v}")
        n = len(self.steps)
        lines.append("")
        lines.append(f"Steps logged: {n}")
        if n:
            def count(key):
                return sum(1 for s in self.steps if s.get(key))
            lines.append(f"Steps with NaN: {count('has_nan')}")
            lines.append(f"Steps with Inf: {count('has_inf')}")
            lines.append(f"Churn fired: {count('churn_fired')} / {n}")
            lines.append(f"SDE noise fired: {count('sde_fired')} / {n}")
            lines.append(f"SABER fusion fired: {count('saber_fired')} / {n}")
            lines.append(f"Sharpen fired: {count('sharpen_fired')} / {n}")
            lines.append(f"Soft-clamp actually engaged (EDM): {count('soft_clamp_engaged')} / {n}")
            dt_fired_total = sum(s.get("dyn_thresh_fired_calls", 0) or 0 for s in self.steps)
            dt_calls_total = sum(s.get("model_calls", 0) or 0 for s in self.steps)
            lines.append(f"dynamic_threshold fired on {dt_fired_total} / {dt_calls_total} model calls")
            from collections import Counter
            ic = Counter(s.get("chosen_integrator") for s in self.steps)
            lines.append(f"Integrator choice counts: {dict(ic)}")
            curvatures = [s["curvature"] for s in self.steps if s.get("curvature") is not None]
            if curvatures:
                lines.append(
                    f"Curvature (auto mode only): min={min(curvatures):.4f} "
                    f"max={max(curvatures):.4f} avg={sum(curvatures)/len(curvatures):.4f}"
                )
            total_ms = sum(s.get("wall_ms", 0) or 0 for s in self.steps)
            lines.append(f"Total wall time logged: {total_ms/1000:.2f}s over {n} steps")
        return "\n".join(lines)

    def save(self) -> Optional[Tuple[str, str, str]]:
        try:
            try:
                import folder_paths
                out_dir = folder_paths.get_output_directory()
            except Exception:
                out_dir = os.path.join(os.getcwd(), "ddrk_debug")
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            base = f"ddrk_debug_{self.tag}_{stamp}" if self.tag else f"ddrk_debug_{stamp}"

            json_path = os.path.join(out_dir, base + ".json")
            with open(json_path, "w") as f:
                json.dump({"meta": self.meta, "steps": self.steps}, f, indent=2, default=str)

            cols: List[str] = []
            for s in self.steps:
                for k in s.keys():
                    if k not in cols:
                        cols.append(k)
            csv_path = os.path.join(out_dir, base + ".csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for s in self.steps:
                    w.writerow(s)

            txt_path = os.path.join(out_dir, base + "_summary.txt")
            with open(txt_path, "w") as f:
                f.write(self._summary_text())

            print(f"[DDRK Debug] Saved:\n  {json_path}\n  {csv_path}\n  {txt_path}")
            return json_path, csv_path, txt_path
        except Exception as e:
            print(f"[DDRK Debug] Failed to save debug log: {e}")
            return None


def _dbg_tensor_stats(t: torch.Tensor) -> dict:
    try:
        with torch.no_grad():
            flat = t.float()
            return {
                "mean": float(flat.mean().item()),
                "std": float(flat.std().item()),
                "min": float(flat.min().item()),
                "max": float(flat.max().item()),
                "abs_mean": float(flat.abs().mean().item()),
                "has_nan": bool(torch.isnan(flat).any().item()),
                "has_inf": bool(torch.isinf(flat).any().item()),
            }
    except Exception as e:
        return {"error": str(e)}


def _dbg_delta(before: torch.Tensor, after: torch.Tensor) -> Optional[float]:
    try:
        with torch.no_grad():
            return float((after - before).abs().mean().item())
    except Exception:
        return None


@torch.no_grad()
def sample_ddrk_omega(model, x, sigmas, extra_args=None, callback=None,
                      disable=False, **kwargs):
    extra_args = {} if extra_args is None else extra_args.copy()

    integrator = kwargs.get("integrator", "auto")
    sde_strength = kwargs.get("sde_strength", 0.08)
    sharpness = kwargs.get("sharpness", 0.30)
    saber_fusion = kwargs.get("saber_fusion", 0.30)
    saber_mode = kwargs.get("saber_mode", "auto")
    ema_decay = kwargs.get("ema_decay", 0.7)
    use_ema_saber = kwargs.get("use_ema_saber", True)
    sde_seed = kwargs.get("sde_seed", None)
    dyn_thresh_percentile = kwargs.get("dyn_thresh_percentile", 0.995)


    latent_rescale = kwargs.get("latent_rescale", kwargs.get("cfg_rescale", 0.0))
    limiter_kappa = kwargs.get("limiter_kappa", 1.0)
    hc2_max_order = int(kwargs.get("hc2_max_order", 2))
    hc2_corrector = float(kwargs.get("hc2_corrector", 0.0))
    sigma_adapt = float(kwargs.get("sigma_adapt", 0.0))
    momentum_beta = kwargs.get("momentum_beta", 0.25)
    s_churn = kwargs.get("s_churn", 0.0)


    s_tmin = kwargs.get("s_tmin", 0.0)
    s_tmax = kwargs.get("s_tmax", float('inf'))
    s_noise = kwargs.get("s_noise", 1.0)
    auto_optimize = kwargs.get("auto_optimize", True)
    debug_mode = kwargs.get("debug_mode", False)
    debug_tag = kwargs.get("debug_tag", "")

    work_device = comfy.model_management.get_torch_device()
    work_dtype = x.dtype
    guard = DeviceDtypeGuard(work_dtype, work_device)

    x = guard(x)
    sigmas = sigmas.to(device=work_device, dtype=torch.float32)
    total_steps = len(sigmas) - 1
    if total_steps < 1:
        return x

    sigma_max = float(sigmas.max())
    is_edm = sigma_max > 5.0

    integrator, sde_strength, sharpness, saber_fusion, momentum_beta = _resolve_effective_params(
        total_steps, is_edm, auto_optimize, integrator, sde_strength, sharpness,
        saber_fusion, momentum_beta,
    )

    rec = _DDRKDebugRecorder(tag=debug_tag) if debug_mode else None
    if rec is not None:
        rec.meta.update({
            "is_edm": is_edm,
            "total_steps": total_steps,
            "sigma_max": sigma_max,
            "integrator_param": integrator,
            "sde_strength": sde_strength,
            "sde_seed": sde_seed,
            "sharpness": sharpness,
            "saber_fusion": saber_fusion,
            "momentum_beta": momentum_beta,
            "s_churn": s_churn, "s_tmin": s_tmin, "s_tmax": s_tmax, "s_noise": s_noise,
            "dyn_thresh_percentile": dyn_thresh_percentile,
            "limiter_kappa": limiter_kappa,
            "hc2_max_order": hc2_max_order,
            "hc2_corrector": hc2_corrector,
            "sigma_adapt": sigma_adapt,
            "latent_rescale": latent_rescale,
            "saber_mode": saber_mode,
            "use_ema_saber": use_ema_saber, "ema_decay": ema_decay,
            "auto_optimize": auto_optimize,
            "latent_shape": list(x.shape),
            "latent_dtype": str(work_dtype),
            "work_device": str(work_device),
            "sigma_schedule": [round(float(s), 5) for s in sigmas.tolist()],
        })


        for label, key in (("cfg", "debug_cfg"), ("seed", "debug_seed"),
                           ("denoise", "debug_denoise"), ("scheduler_type", "debug_scheduler_type"),
                           ("flow_shift", "debug_flow_shift"), ("steps_requested", "debug_steps_requested")):
            val = kwargs.get(key, None)
            if val is not None:
                rec.meta[label] = val

    state = SamplerState(total_steps=total_steps, is_edm=is_edm)
    router = AdaptivePhaseRouter(total_steps, sigma_max, is_edm)
    content_aware = kwargs.get("content_aware", True)
    if rec is not None:
        rec.meta["content_aware"] = content_aware
    saber = SABER2(mode=saber_mode, buffer_size=3, fusion=saber_fusion,
                   ema_decay=ema_decay, use_ema=use_ema_saber,
                   content_aware=content_aware)
    sde = AdaptiveSDE(seed=sde_seed)
    log_mask = LoGMask()

    s_in = x.new_ones([x.shape[0]], dtype=work_dtype, device=work_device)


    debug_counters = {"model_calls": 0, "dt_fired": 0} if rec is not None else None

    def model_fn(latent_in, sigma_val):
        out = model(latent_in.to(work_dtype), sigma_val * s_in, **extra_args).to(work_dtype)
        if debug_counters is not None:
            debug_counters["model_calls"] += 1
        if dyn_thresh_percentile < 1.0:
            before_dt = out if debug_counters is not None else None
            out = dynamic_threshold(out, float(sigma_val), sigma_max, dyn_thresh_percentile)
            if debug_counters is not None and not torch.equal(before_dt, out):
                debug_counters["dt_fired"] += 1
        if latent_rescale > 0:
            dims = (2, 3) if out.dim() == 4 else (2, 3, 4)
            mean = out.mean(dim=dims, keepdim=True)
            std = out.std(dim=dims, keepdim=True).clamp_min(1e-8)
            deviation = out - mean
            scale_factor = 1.0 / (1.0 + latent_rescale * (deviation.abs() / std - 2.0).clamp_min(0.0))
            out = mean + deviation * scale_factor
        return out

    preview_denoised = None
    progress_bar = trange(total_steps, disable=disable)


    sigmas_work = sigmas.clone()
    sigmas_ref = sigmas.clone()
    adapt_scale = 1.0
    activity_sum, activity_n = 0.0, 0

    for i in progress_bar:
        sigma_curr = sigmas_work[i]
        sigma_next = sigmas_work[i + 1]
        state.step_count = i

        if float(sigma_curr) < 1e-7:
            break

        if rec is not None:
            rec.start_step()
            step_rec: Dict[str, Any] = {"step": i, "phase": None}
            debug_counters["model_calls"] = 0
            debug_counters["dt_fired"] = 0
            sigma_curr_pre_churn = float(sigma_curr)

        if is_edm and s_churn > 0 and s_tmin <= float(sigma_curr) <= s_tmax:
            gamma = min(s_churn / float(sigma_curr), math.sqrt(2) - 1)
            sigma_hat = float(sigma_curr) * (1.0 + gamma)


            noise = torch.randn_like(x, generator=sde.get_generator(x.device)) * s_noise
            x_before_churn = x if rec is not None else None
            x = x + noise * math.sqrt(sigma_hat ** 2 - float(sigma_curr) ** 2)
            sigma_curr = torch.tensor(sigma_hat, device=work_device, dtype=torch.float32)
            if rec is not None:
                step_rec["churn_fired"] = True
                step_rec["churn_gamma"] = round(gamma, 5)
                step_rec["churn_sigma_hat"] = round(sigma_hat, 5)
                step_rec["churn_delta"] = _dbg_delta(x_before_churn, x)
        elif rec is not None:
            step_rec["churn_fired"] = False

        phase = router.get_phase(i)
        chosen_integrator = router.pick_integrator(i, float(sigma_curr), integrator, state)
        if rec is not None:
            step_rec["phase"] = phase
            step_rec["chosen_integrator"] = chosen_integrator
            step_rec["sigma_curr"] = round(sigma_curr_pre_churn, 5)
            step_rec["sigma_curr_post_churn"] = round(float(sigma_curr), 5)
            step_rec["sigma_next"] = round(float(sigma_next), 5)

        if phase == 1:
            sde_active = (not is_edm) and sde_strength > 0 and float(sigma_next) > 1e-7
            if sde_active:


                sigma_down, sigma_up = _ancestral_split(float(sigma_curr), float(sigma_next), eta=sde_strength)
                sigma_down_t = torch.tensor(sigma_down, device=work_device, dtype=torch.float32)
            else:
                sigma_down_t = sigma_next

            x_next, preview_denoised, _ = _integrator_step(
                chosen_integrator, x, sigma_curr, sigma_down_t, model_fn, state, momentum_beta,
                limiter_kappa=limiter_kappa, hc2_max_order=hc2_max_order,
                hc2_corrector=hc2_corrector)

            if sde_active and sigma_up > 1e-9:
                edge_mask = log_mask(x_next)
                entropy = local_entropy_mask(x_next, window=5)
                flat_mask = ((1.0 - edge_mask) * entropy).clamp(0.0, 1.0)
                x_before_sde = x_next if rec is not None else None
                x_next = x_next + sde(x_next, sigma_up, flat_mask)
                if rec is not None:
                    step_rec["sde_fired"] = True
                    step_rec["sigma_up"] = round(float(sigma_up), 5)
                    step_rec["sde_delta"] = _dbg_delta(x_before_sde, x_next)
                    try:
                        step_rec["edge_mask_mean"] = round(float(edge_mask.mean().item()), 5)
                        step_rec["entropy_mean"] = round(float(entropy.mean().item()), 5)
                        step_rec["flat_mask_mean"] = round(float(flat_mask.mean().item()), 5)
                    except Exception:
                        pass
            elif rec is not None:
                step_rec["sde_fired"] = False


            if is_edm and saber_fusion > 0 and float(sigma_curr) > 0.55 * sigma_max:
                x_before_saber = x_next if rec is not None else None
                x_next = saber.fuse(x_next)
                if rec is not None:
                    step_rec["saber_fired"] = True
                    step_rec["saber_delta"] = _dbg_delta(x_before_saber, x_next)
            elif rec is not None:
                step_rec["saber_fired"] = False

        elif phase == 2:
            x_next, preview_denoised, _ = _integrator_step(
                chosen_integrator, x, sigma_curr, sigma_next, model_fn, state, momentum_beta,
                limiter_kappa=limiter_kappa, hc2_max_order=hc2_max_order,
                hc2_corrector=hc2_corrector)
            if not is_edm and saber_fusion > 0:
                x_before_saber = x_next if rec is not None else None
                x_next = saber.fuse(x_next)
                if rec is not None:
                    step_rec["saber_fired"] = True
                    step_rec["saber_delta"] = _dbg_delta(x_before_saber, x_next)
            elif rec is not None:
                step_rec["saber_fired"] = False

        else:
            x_next, preview_denoised, _ = _integrator_step(
                chosen_integrator, x, sigma_curr, sigma_next, model_fn, state, momentum_beta,
                limiter_kappa=limiter_kappa, hc2_max_order=hc2_max_order,
                hc2_corrector=hc2_corrector)
            if (saber_fusion > 0 and saber_mode in ("video", "auto")
                    and x_next.dim() == 5 and x_next.shape[2] > 1):
                x_before_saber = x_next if rec is not None else None
                x_next = saber.fuse(x_next)
                if rec is not None:
                    step_rec["saber_fired"] = True
                    step_rec["saber_delta"] = _dbg_delta(x_before_saber, x_next)
            elif rec is not None:
                step_rec["saber_fired"] = False


            is_final_step = float(sigma_next) <= 1e-7
            if is_final_step and sharpness > 0:
                if not is_edm:
                    x_before_sharpen = x_next if rec is not None else None
                    x_next = perceptual_sharpen(x_next, sharpness, is_final_step=True)
                    if rec is not None:
                        step_rec["sharpen_fired"] = True
                        step_rec["sharpen_delta"] = _dbg_delta(x_before_sharpen, x_next)
                elif rec is not None:
                    step_rec["sharpen_fired"] = False
            elif rec is not None:
                step_rec["sharpen_fired"] = False


        _CLAMP_BOUND = max(4.0 * float(sigma_curr), 10.0)
        if is_edm:
            if rec is not None:
                try:
                    step_rec["soft_clamp_bound"] = round(_CLAMP_BOUND, 3)
                    step_rec["soft_clamp_engaged"] = bool((x_next.abs() > _CLAMP_BOUND).any().item())
                    step_rec["soft_clamp_frac"] = float((x_next.abs() > _CLAMP_BOUND).float().mean().item())
                except Exception:
                    pass
            x_next = _soft_clamp(x_next, bound=_CLAMP_BOUND, softness=0.8)
        elif rec is not None:
            step_rec["soft_clamp_engaged"] = None


        if integrator == "auto" and phase == 1 and preview_denoised is not None and state.prev_denoised is not None:
            with torch.no_grad():
                diff = (preview_denoised - state.prev_denoised).abs().mean()
                base = state.prev_denoised.abs().mean().clamp_min(1e-8)


                state.curvature = float((diff / base).item())
        else:
            state.curvature = 0.0

        if (sigma_adapt > 0.0 and chosen_integrator == "hc2"
                and state.hc2_activity > 0.0 and i + 2 < len(sigmas_work)):


            activity_sum += state.hc2_activity
            activity_n += 1
            mean_act = activity_sum / activity_n
            if mean_act > 1e-12:
                f = (mean_act / state.hc2_activity) ** (1.0 / 3.0)
                f = min(max(f, 1.0 - sigma_adapt), 1.0 + sigma_adapt)
                adapt_scale = min(max(adapt_scale * f, 1.0 - sigma_adapt),
                                  1.0 + sigma_adapt)


                s_here = float(sigmas_work[i + 1])
                s_ref_next = float(sigmas_ref[i + 2])
                if s_ref_next > 0.0:
                    max_scale = (s_here * 0.98) / s_ref_next
                    adapt_scale = min(adapt_scale, max_scale)
                    sigmas_work[i + 2:] = sigmas_ref[i + 2:] * adapt_scale
                    if rec is not None:
                        step_rec["sigma_adapt_scale"] = round(float(adapt_scale), 5)

        if rec is not None:
            step_rec["curvature"] = state.curvature if integrator == "auto" and phase == 1 else None
            if state.hc2_limited_frac >= 0.0:
                step_rec["hc2_limiter_frac"] = round(state.hc2_limited_frac, 5)
            if state.hc2_order_used:
                step_rec["hc2_order"] = state.hc2_order_used
            if state.hc2_err_ratio >= 0.0:
                step_rec["hc2_err_ratio"] = round(state.hc2_err_ratio, 5)
            if state.hc2_activity >= 0.0:
                step_rec["hc2_activity"] = round(state.hc2_activity, 5)
            if state.hc2_corrector_fired:
                step_rec["hc2_corrector_fired"] = True
            step_rec["model_calls"] = debug_counters["model_calls"]
            step_rec["dyn_thresh_fired_calls"] = debug_counters["dt_fired"]
            stats = _dbg_tensor_stats(x_next)
            step_rec["x_mean"] = stats.get("mean")
            step_rec["x_std"] = stats.get("std")
            step_rec["x_min"] = stats.get("min")
            step_rec["x_max"] = stats.get("max")
            step_rec["x_abs_mean"] = stats.get("abs_mean")
            step_rec["has_nan"] = stats.get("has_nan")
            step_rec["has_inf"] = stats.get("has_inf")
            rec.log_step(**step_rec)

        state.prev_denoised = preview_denoised.detach().clone() if preview_denoised is not None else None
        state.prev_sigma = float(sigma_curr)

        x = x_next

        if callback is not None:
            callback({
            "x": x,
            "i": i,
            "sigma": sigma_curr,
            "denoised": preview_denoised,
            })

    if rec is not None:
        rec.meta["actual_steps_run"] = len(rec.steps)
        rec.meta["final_x_stats"] = _dbg_tensor_stats(x)
        rec.save()


    if is_edm:
        x = torch.clamp(x, -7.0, 7.0)
    else:
        x = torch.clamp(x, -20.0, 20.0)
    return x


class DDRKOmegaSchedulerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Auto-detects FM vs EDM by sigma_max"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "scheduler_type": ([
                    "ddrk_auto",
                    "ddrk_cosine",
                    "ddrk_beta",
                    "ddrk_flow_linear",
                    "ddrk_flow_cosmos",
                    "ddrk_fewstep",
                    "ddrk_edm_karras",
                    "ddrk_edm_poly",
                    "ddrk_edm_simple",
                ], {"default": "ddrk_auto"}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 5}),
                "beta_a": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Shape parameter A, used only by ddrk_beta. Higher A front-loads larger steps at high sigma and leaves finer steps near sigma 0. Ignored by every other scheduler."}),
                "beta_b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Shape parameter B, used only by ddrk_beta. Raising B above ~2 shifts resolution toward high sigma and leaves a large final step, which is usually undesirable. A=2, B=1 gives Karras-like monotonically shrinking steps."}),
                "auto_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-select scheduler & flow_shift for FM few-step. Disable for full manual control."
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_schedulers"

    def get_sigmas(self, model, steps, scheduler_type, flow_shift, warmup_steps,
                   auto_optimize, beta_a=2.0, beta_b=1.0):
        ms = model.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        device = ms.sigma_min.device
        sigmas = get_ddrk_sigmas(
            scheduler_type, steps, sigma_min, sigma_max,
            device=device, flow_shift=flow_shift, warmup_steps=warmup_steps,
            beta_a=beta_a, beta_b=beta_b, auto_optimize=auto_optimize
        )
        return (sigmas,)


class DDRKOmegaSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "integrator": (["auto", "hc2", "rk4", "heun", "euler"], {"default": "auto"}),
                "sde_strength": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Stochastic noise. 0 = deterministic. FM only."
                }),
                "sharpness": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01,
                    "tooltip": "Final sharpening. Auto-reduced for few-step."
                }),
                "saber_fusion": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stabilization. Auto-capped to 0.15 for FM photo models."
                }),
                "saber_mode": (["auto", "image", "video"], {
                    "default": "auto",
                    "tooltip": "Auto detects video by 5D latent with >1 frame."
                }),
                "use_ema_saber": ("BOOLEAN", {"default": True}),
                "ema_decay": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01}),
                "dyn_thresh_percentile": ("FLOAT", {
                    "default": 0.995, "min": 0.9, "max": 1.0, "step": 0.001,
                    "tooltip": "Dynamic thresholding percentile. 1.0 = disabled. Universal (FM + EDM)."
                }),
                "latent_rescale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Attenuates latent values far from the per-image mean (beyond ~2 std). 0 = disabled. NOTE: this is NOT classical CFG-rescale (Lin et al.), which compares conditional vs unconditional predictions — those are already combined before this sampler is called, so that method cannot be implemented here. Renamed from 'cfg_rescale' to stop implying otherwise."
                }),
                "hc2_corrector": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "HC2 selective corrector. 0 = off. Otherwise, on any step where the high-order correction exceeds this fraction of the first-order step, HC2 spends a SECOND model call to re-evaluate the denoiser at the step's endpoint and redo the step with an interpolated slope instead of an extrapolated one. Costs one extra call per firing. Small values (0.01-0.05) fire almost every step; larger values fire only on difficult steps. EXPERIMENTAL: strong on synthetic tests, unvalidated on images."
                }),
                "sigma_adapt": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "HC2 adaptive step placement. 0 = off. Otherwise the sampler may move the intermediate sigma values by up to this fraction to equalise estimated error across steps. Step COUNT, start and terminal zero are unchanged, so timing and the progress bar are unaffected. EXPERIMENTAL and the most speculative option here: schedules are already heavily tuned per model family, and nudging them may fight that tuning. Try 0.10-0.20."
                }),
                "hc2_max_order": ("INT", {
                    "default": 2, "min": 1, "max": 3,
                    "tooltip": "HC2 maximum order. 2 = second order, one model call per step (default). 3 = allow third order: uses two past denoiser evaluations, spends ONE extra model call on the first step to bootstrap (a multistep method's first step is otherwise first-order and caps the whole run's accuracy), and falls back to second order on any step where the expansion stops converging. Ignored by every other integrator."
                }),
                "limiter_kappa": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "HC2 slope limiter. Caps the 2nd-order correction at this multiple of the 1st-order step, per element. 1.0 = the correction may at most double or cancel the step, never reverse it. Lower it (0.5-0.7) if high CFG still blows out highlights; raise it toward 2-3 to let HC2 run closer to unlimited 2nd order on smooth content. Ignored by every other integrator."
                }),
                "momentum_beta": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 0.8, "step": 0.05,
                    "tooltip": "Velocity EMA. Reduces oscillation. 0 = disabled."
                }),
                "sde_seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": "SDE noise seed. -1 = random (non-deterministic)."
                }),
                "s_churn": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "EDM churn (Karras Alg 2). 0 = off. Try 5-15 for SDXL. EDM models only; silently ignored for Flow Matching."
                }),
                "s_tmin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 999999.0, "step": 0.1,
                    "tooltip": "EDM churn lower sigma bound (Karras Alg 2). Churn only fires while s_tmin <= sigma <= s_tmax. 0.0 = no lower bound."
                }),
                "s_tmax": ("FLOAT", {
                    "default": 999999.0, "min": 0.0, "max": 999999.0, "step": 1.0,
                    "tooltip": "EDM churn upper sigma bound. Default (max) means unbounded, matching Karras Alg 2's s_tmax=inf. Lower it to restrict churn to high-sigma steps only."
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EDM churn noise multiplier. 1.0 = standard."
                }),
                "content_aware": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Content-aware SABER — edge-gated fusion. Disable for pixel-art/flat styles."
                }),
                "auto_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-disable SABER/SDE/momentum and force euler for FM few-step. Does NOT set steps/cfg."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Write a per-step diagnostics log (JSON+CSV+summary) to your ComfyUI output folder. Off by default; adds overhead only when on."
                }),
                "debug_tag": ("STRING", {
                    "default": "",
                    "tooltip": "Optional label included in the debug log filename, e.g. 'test1'."
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_samplers"

    def get_sampler(self, integrator, sde_strength, sharpness, saber_fusion,
                    saber_mode, use_ema_saber, ema_decay, dyn_thresh_percentile,
                    latent_rescale, limiter_kappa, hc2_corrector, sigma_adapt, hc2_max_order, momentum_beta, sde_seed, s_churn, s_tmin, s_tmax,
                    s_noise, content_aware, auto_optimize, debug_mode=False, debug_tag=""):
        extra = {
            "integrator": integrator,
            "sde_strength": sde_strength,
            "sharpness": sharpness,
            "saber_fusion": saber_fusion,
            "saber_mode": saber_mode,
            "use_ema_saber": use_ema_saber,
            "ema_decay": ema_decay,
            "dyn_thresh_percentile": dyn_thresh_percentile,
            "limiter_kappa": limiter_kappa,
            "hc2_max_order": hc2_max_order,
            "hc2_corrector": hc2_corrector,
            "sigma_adapt": sigma_adapt,
            "latent_rescale": latent_rescale,
            "limiter_kappa": limiter_kappa,
            "hc2_max_order": hc2_max_order,
            "hc2_corrector": hc2_corrector,
            "sigma_adapt": sigma_adapt,
            "momentum_beta": momentum_beta,
            "sde_seed": sde_seed if sde_seed >= 0 else None,
            "s_churn": s_churn,
            "s_tmin": s_tmin,
            "s_tmax": s_tmax,
            "s_noise": s_noise,
            "content_aware": content_aware,
            "auto_optimize": auto_optimize,
            "debug_mode": debug_mode,
            "debug_tag": debug_tag,
        }
        sampler = comfy.samplers.KSAMPLER(sample_ddrk_omega, extra_options=extra)
        return (sampler,)


class DDRKOmegaSmartConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {}),
                "latent_image": ("LATENT", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "STRING", "FLOAT", "STRING",
                    "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("family", "hint", "guidance_embed", "scheduler_type", "flow_shift", "integrator",
                    "sde_strength", "sharpness", "saber_fusion", "momentum_beta")
    FUNCTION = "detect"
    CATEGORY = "sampling/custom_schedulers"

    def detect(self, model, latent_image):
        profile = _detect_model_profile(model, latent_image.get("samples"))
        print(f"[DDRK SmartConfig] Detected family: {profile['family'].upper()}")
        print(f"[DDRK SmartConfig] Hint: {profile['hint']}")
        return (profile["family"], profile["hint"], profile["guidance_embed"], profile["scheduler_type"],
                profile["flow_shift"], profile["integrator"],
                profile["sde_strength"], profile["sharpness"],
                profile["saber_fusion"], profile["momentum_beta"])


class DDRKOmegaUnifiedKSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "latent_image": ("LATENT", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise strength, as in the stock KSampler. Minimum is 0.01, not 0.0: denoise scales the step count via steps/denoise, so exactly 0.0 is a division by zero, and 0.0 denoise means 'do nothing' anyway."}),
                "scheduler_type": ([
                    "ddrk_auto", "ddrk_cosine", "ddrk_beta",
                    "ddrk_flow_linear", "ddrk_flow_cosmos", "ddrk_fewstep",
                    "ddrk_edm_karras", "ddrk_edm_poly", "ddrk_edm_simple",
                ], {"default": "ddrk_auto"}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "integrator": (["auto", "hc2", "rk4", "heun", "euler"], {"default": "auto"}),
                "sde_strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01}),
                "saber_fusion": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stabilization blur. 0 = disabled. Disable for text/graphics."
                }),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 5}),
                "beta_a": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Shape parameter A, used only by ddrk_beta. Higher A front-loads larger steps at high sigma and leaves finer steps near sigma 0. Ignored by every other scheduler."}),
                "beta_b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Shape parameter B, used only by ddrk_beta. Raising B above ~2 shifts resolution toward high sigma and leaves a large final step, which is usually undesirable. A=2, B=1 gives Karras-like monotonically shrinking steps."}),
                "auto_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-disable SABER/SDE/momentum and force euler for FM few-step. Does NOT set steps/cfg."
                }),
                "smart_defaults": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-detect architecture and set scheduler/integrator/shift/saber/sharpness only. Steps/CFG are checkpoint-specific and must be tuned manually."
                }),
            },
            "optional": {
                "saber_mode": (["auto", "image", "video"], {"default": "auto"}),
                "use_ema_saber": ("BOOLEAN", {"default": True}),
                "ema_decay": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01}),
                "dyn_thresh_percentile": ("FLOAT", {"default": 0.995, "min": 0.9, "max": 1.0, "step": 0.001}),
                "latent_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Attenuates latent values beyond ~2 std from the per-image mean. 0 = disabled. Not classical CFG-rescale; renamed from 'cfg_rescale'."}),
                "hc2_corrector": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "HC2 selective corrector. 0 = off. Otherwise, on any step where the high-order correction exceeds this fraction of the first-order step, HC2 spends a SECOND model call to re-evaluate the denoiser at the step's endpoint and redo the step with an interpolated slope instead of an extrapolated one. Costs one extra call per firing. Small values (0.01-0.05) fire almost every step; larger values fire only on difficult steps. EXPERIMENTAL: strong on synthetic tests, unvalidated on images."
                }),
                "sigma_adapt": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "HC2 adaptive step placement. 0 = off. Otherwise the sampler may move the intermediate sigma values by up to this fraction to equalise estimated error across steps. Step COUNT, start and terminal zero are unchanged, so timing and the progress bar are unaffected. EXPERIMENTAL and the most speculative option here: schedules are already heavily tuned per model family, and nudging them may fight that tuning. Try 0.10-0.20."
                }),
                "hc2_max_order": ("INT", {
                    "default": 2, "min": 1, "max": 3,
                    "tooltip": "HC2 maximum order. 2 = second order, one model call per step (default). 3 = allow third order: uses two past denoiser evaluations, spends ONE extra model call on the first step to bootstrap (a multistep method's first step is otherwise first-order and caps the whole run's accuracy), and falls back to second order on any step where the expansion stops converging. Ignored by every other integrator."
                }),
                "limiter_kappa": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "HC2 slope limiter. Caps the 2nd-order correction at this multiple of the 1st-order step, per element. 1.0 = the correction may at most double or cancel the step, never reverse it. Lower it (0.5-0.7) if high CFG still blows out highlights; raise it toward 2-3 to let HC2 run closer to unlimited 2nd order on smooth content. Ignored by every other integrator."
                }),
                "momentum_beta": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.05}),
                "sde_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "s_churn": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "EDM churn (Karras Alg 2). 0 = off. Try 5-15 for SDXL. EDM models only; silently ignored for Flow Matching."
                }),
                "s_tmin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 999999.0, "step": 0.1,
                    "tooltip": "EDM churn lower sigma bound (Karras Alg 2). Churn only fires while s_tmin <= sigma <= s_tmax. 0.0 = no lower bound."
                }),
                "s_tmax": ("FLOAT", {
                    "default": 999999.0, "min": 0.0, "max": 999999.0, "step": 1.0,
                    "tooltip": "EDM churn upper sigma bound. Default (max) means unbounded, matching Karras Alg 2's s_tmax=inf. Lower it to restrict churn to high-sigma steps only."
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EDM churn noise multiplier. 1.0 = standard."
                }),
                "content_aware": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Content-aware SABER — edge-gated fusion. Disable for pixel-art/flat styles."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Write a per-step diagnostics log (JSON+CSV+summary) to your ComfyUI output folder. Off by default; adds overhead only when on."
                }),
                "debug_tag": ("STRING", {
                    "default": "",
                    "tooltip": "Optional label included in the debug log filename, e.g. 'test1'."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/unified_samplers"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               denoise, scheduler_type, flow_shift, integrator, sde_strength,
               sharpness, warmup_steps, auto_optimize, smart_defaults,
               saber_fusion=0.30, saber_mode="auto", use_ema_saber=True,
               ema_decay=0.7, dyn_thresh_percentile=0.995, latent_rescale=0.0, limiter_kappa=1.0, hc2_corrector=0.0, sigma_adapt=0.0, hc2_max_order=2,
               momentum_beta=0.25, sde_seed=-1, s_churn=0.0, s_tmin=0.0,
               s_tmax=999999.0, s_noise=1.0, content_aware=True,
               beta_a=2.0, beta_b=1.0,
               debug_mode=False, debug_tag=""):

        latent = latent_image.copy()
        latent_samples = latent["samples"]
        noise_mask = latent.get("noise_mask", None)


        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)


        if smart_defaults:
            profile = _detect_model_profile(model, latent_samples)
            scheduler_type = profile["scheduler_type"]
            flow_shift = profile["flow_shift"]
            integrator = profile["integrator"]
            sde_strength = profile["sde_strength"]
            sharpness = profile["sharpness"]
            saber_fusion = profile["saber_fusion"]
            momentum_beta = profile["momentum_beta"]
            print(f"[DDRK Smart] {profile['family'].upper()} detected: "
                  f"scheduler={scheduler_type}, shift={flow_shift}, integrator={integrator}")
            print(f"[DDRK Smart] Hint: {profile['hint']}")


        denoise = float(min(max(denoise, 0.01), 1.0))


        steps_denoised = steps
        if denoise < 1.0:
            steps = max(1, int(steps / denoise))

        ms = model.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        device = ms.sigma_min.device

        sigmas = get_ddrk_sigmas(
            scheduler_type, steps, sigma_min, sigma_max,
            device=device, flow_shift=flow_shift, warmup_steps=warmup_steps,
            beta_a=beta_a, beta_b=beta_b, auto_optimize=auto_optimize
        )

        noise = comfy.sample.prepare_noise(latent_samples, seed, None)

        if denoise < 1.0:
            sigmas = sigmas[-(steps_denoised + 1):]


        actual_steps = len(sigmas) - 1
        callback = None
        try:
            import latent_preview
            callback = latent_preview.prepare_callback(model, actual_steps)
        except Exception:
            try:
                import comfy.latent_preview as lp
                callback = lp.prepare_callback(model, actual_steps)
            except Exception as e:
                print(f"[DDRK] Preview callback failed: {e}")

        extra = {
            "integrator": integrator,
            "sde_strength": sde_strength,
            "sharpness": sharpness,
            "saber_fusion": saber_fusion,
            "saber_mode": saber_mode,
            "use_ema_saber": use_ema_saber,
            "ema_decay": ema_decay,
            "dyn_thresh_percentile": dyn_thresh_percentile,
            "limiter_kappa": limiter_kappa,
            "hc2_max_order": hc2_max_order,
            "hc2_corrector": hc2_corrector,
            "sigma_adapt": sigma_adapt,
            "latent_rescale": latent_rescale,
            "limiter_kappa": limiter_kappa,
            "hc2_max_order": hc2_max_order,
            "hc2_corrector": hc2_corrector,
            "sigma_adapt": sigma_adapt,
            "momentum_beta": momentum_beta,
            "sde_seed": sde_seed if sde_seed >= 0 else None,
            "s_churn": s_churn,
            "s_tmin": s_tmin,
            "s_tmax": s_tmax,
            "s_noise": s_noise,
            "content_aware": content_aware,
            "auto_optimize": auto_optimize,
            "debug_mode": debug_mode,
            "debug_tag": debug_tag,


            "debug_cfg": cfg,
            "debug_seed": seed,
            "debug_denoise": denoise,
            "debug_scheduler_type": scheduler_type,
            "debug_flow_shift": flow_shift,
            "debug_steps_requested": steps_denoised,
        }
        sampler_obj = comfy.samplers.KSAMPLER(sample_ddrk_omega, extra_options=extra)

        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler_obj, sigmas, positive, negative,
            latent_image=latent_samples, noise_mask=noise_mask,
            callback=callback, disable_pbar=False, seed=seed
        )

        out = latent.copy()
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "DDRKOmegaSchedulerNode": DDRKOmegaSchedulerNode,
    "DDRKOmegaSamplerNode": DDRKOmegaSamplerNode,
    "DDRKOmegaUnifiedKSamplerNode": DDRKOmegaUnifiedKSamplerNode,
    "DDRKOmegaSmartConfigNode": DDRKOmegaSmartConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDRKOmegaSchedulerNode": "DDRK Omega Scheduler",
    "DDRKOmegaSamplerNode": "DDRK Omega Sampler",
    "DDRKOmegaUnifiedKSamplerNode": "DDRK Omega Unified KSampler",
    "DDRKOmegaSmartConfigNode": "DDRK Omega Smart Config",
}
