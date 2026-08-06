"""
================================================================================
DDRK Omega Sampler — v1.4.1 Adaptive Balance
ComfyUI | Flow Matching + EDM Universal Sampler

v1.4.1: Soft variance clamp in loop, adaptive DT (skip if no outliers),
      SABER for EDM only on very early noise. Targets both photoreal
      texture and anime cleanliness without plastic look.
================================================================================
"""

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import math
from collections import OrderedDict
from tqdm.auto import trange
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass


# =========================================================
# CONFIG & UTILITIES
# =========================================================

@dataclass
class SamplerState:
    d_prev: Optional[torch.Tensor] = None
    step_count: int = 0
    total_steps: int = 0
    is_edm: bool = False


class DeviceDtypeGuard:
    def __init__(self, target_dtype, target_device):
        self.dtype = target_dtype
        self.device = target_device

    def __call__(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.to(device=self.device, dtype=self.dtype)


def _detect_family(sigma_max: float) -> str:
    return "edm" if sigma_max > 5.0 else "flow"


# =========================================================
# DYNAMIC THRESHOLDING
# =========================================================

def dynamic_threshold(denoised: torch.Tensor, sigma: float, sigma_max: float,
                      percentile: float = 0.995, min_val: float = 1.0,
                      edm_ratio: float = 0.30) -> torch.Tensor:
    """
    v1.4.1: EDM ratio 0.30 — clamps early noisy steps only.
    ADAPTIVE: completely skipped if no actual outliers are present.
    This preserves photoreal micro-texture while catching CFG blowouts.
    """
    threshold_ratio = 0.40 if sigma_max <= 5.0 else edm_ratio
    if sigma > threshold_ratio * sigma_max or denoised.numel() == 0:
        return denoised

    # Adaptive skip: if max latent magnitude is sane, do nothing
    flat = denoised.reshape(denoised.shape[0], -1)
    max_val = flat.abs().max(dim=1, keepdim=True)[0]
    if max_val.max().item() < 3.5:
        return denoised

    abs_flat = flat.abs()
    s = torch.quantile(abs_flat, percentile, dim=1, keepdim=True)
    s = torch.clamp(s, min=min_val)
    s = s.reshape(denoised.shape[0], *[1] * (denoised.dim() - 1))
    ratio = s / denoised.abs().clamp_min(1e-8)
    scale = torch.where(ratio < 1.0, ratio, torch.ones_like(ratio))
    return denoised * scale


def _soft_clamp(t: torch.Tensor, bound: float = 5.0, softness: float = 0.25) -> torch.Tensor:
    """
    v1.4.1: Soft clamp — values inside [-bound, bound] are untouched.
    Tails beyond the bound are compressed, not hard-cut.
    Prevents latent explosion without destroying micro-contrast.
    """
    core = torch.clamp(t, -bound, bound)
    excess = t - core
    return core + softness * excess


# =========================================================
# EDGE & ENTROPY MASKS
# =========================================================

class LoGMask:
    def __init__(self, max_cache: int = 4):
        self._kernels: OrderedDict = OrderedDict()
        self._max_cache = max_cache

    def __call__(self, x: torch.Tensor, threshold: float = 0.025) -> torch.Tensor:
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
        emax = edge.max()
        if emax > 1e-8:
            edge = edge / emax
        mask = (edge > threshold).float()
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
    flatness = 1.0 - torch.sigmoid((var - var.mean()) * 30.0)

    if is_5d:
        flatness = flatness.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
    return flatness


# =========================================================
# SABER 2.0 — SPATIAL + TEMPORAL
# =========================================================

class SABER2:
    def __init__(self, mode: str = "auto", buffer_size: int = 3,
                 fusion: float = 0.35, ema_decay: float = 0.7, use_ema: bool = True,
                 max_keys: int = 4):
        self.mode = mode
        self.buffer_size = buffer_size
        self.fusion = fusion
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self._max_keys = max_keys
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

        if len(buf["frames"]) < 2:
            return x

        if is_vid:
            stacked = torch.stack(buf["frames"], dim=0)
            var = torch.var(stacked, dim=0)
            chaos = torch.sigmoid((var - var.mean()) * 10.0)
            if self.use_ema:
                if buf["ema"] is None:
                    buf["ema"] = x.detach().clone()
                else:
                    buf["ema"] = self.ema_decay * buf["ema"] + (1 - self.ema_decay) * x.detach()
                fused = x * (1.0 - chaos * self.fusion) + buf["ema"] * (chaos * self.fusion)
            else:
                avg = torch.mean(stacked, dim=0)
                fused = x * (1.0 - chaos * self.fusion) + avg * (chaos * self.fusion)
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
            edges = (x - blurred).abs()
            edge_weight = torch.sigmoid((edges - edges.mean()) * 20.0)
            w = self.fusion * (1.0 - edge_weight)
            return x * (1.0 - w) + blurred * w


# =========================================================
# SHARPENING
# =========================================================

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

    blurred = F.avg_pool2d(F.pad(x_4d, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
    detail = x_4d - blurred

    if not is_final_step:
        mask = local_entropy_mask(x_4d, window=3)
        detail = detail * mask

    detail = torch.clamp(detail, min=-0.6, max=0.6)
    out = x_4d + strength * detail

    if is_5d:
        out = out.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
    return out


# =========================================================
# INTEGRATORS
# =========================================================

def _safe_sigma(s: Union[float, torch.Tensor]) -> float:
    return max(float(s), 1e-8)


def euler_step(x, sigma, sigma_next, model_fn, state: SamplerState):
    denoised = model_fn(x, sigma)
    d = (x - denoised) / _safe_sigma(sigma)
    d = _apply_momentum(d, state)
    dt = sigma_next - sigma
    return x + d * dt, denoised, d


def heun_step(x, sigma, sigma_next, model_fn, state: SamplerState):
    denoised = model_fn(x, sigma)
    d = (x - denoised) / _safe_sigma(sigma)
    dt = sigma_next - sigma
    x_next = x + d * dt

    if float(sigma_next) > 1e-7:
        denoised_2 = model_fn(x_next, sigma_next)
        d2 = (x_next - denoised_2) / _safe_sigma(sigma_next)
        d_avg = (d + d2) * 0.5
        d_avg = _apply_momentum(d_avg, state, force=True)
        x_next = x + d_avg * dt
        return x_next, denoised_2, d_avg

    _apply_momentum(d, state)
    return x_next, denoised, d


def rk4_step(x, sigma, sigma_next, model_fn, state: SamplerState):
    dt = sigma_next - sigma
    s = _safe_sigma(sigma)
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
    d_final = _apply_momentum(d_final, state, force=True)
    return x + d_final * dt, denoised_4, d_final


def _apply_momentum(d: torch.Tensor, state: SamplerState, force: bool = False,
                    beta: float = 0.25) -> torch.Tensor:
    if state.d_prev is None:
        state.d_prev = d.detach().clone()
        return d
    t = state.step_count / max(state.total_steps, 1)
    adaptive_beta = beta * (1.0 - t * 0.5)
    if force:
        adaptive_beta *= 0.5
    if state.is_edm:
        adaptive_beta *= 0.6
    d_smooth = adaptive_beta * state.d_prev + (1.0 - adaptive_beta) * d
    state.d_prev = d_smooth.detach().clone()
    return d_smooth


# =========================================================
# PHASE ENGINE
# =========================================================

class AdaptivePhaseRouter:
    def __init__(self, total_steps: int, sigma_max: float, is_edm: bool,
                 p1_ratio: float = 0.65, p2_ratio: float = 0.25):
        self.total_steps = total_steps
        self.sigma_max = sigma_max
        self.is_edm = is_edm

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

    def pick_integrator(self, step_idx: int, sigma: float, cfg: str) -> str:
        phase = self.get_phase(step_idx)
        if cfg == "auto":
            if phase == 1 and self.total_steps >= 10 and not self.is_edm:
                if sigma > 0.3 * self.sigma_max:
                    return "rk4"
                return "heun"
            elif phase == 1 and self.is_edm:
                return "heun"
            elif phase == 2:
                return "heun" if self.total_steps >= 8 else "euler"
            else:
                return "euler"
        return cfg


# =========================================================
# SDE NOISE
# =========================================================

class AdaptiveSDE:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._gen: Optional[torch.Generator] = None

    def _get_gen(self, device):
        if self._gen is None or str(self._gen.device) != str(device):
            self._gen = torch.Generator(device=device)
            if self.seed is not None:
                self._gen.manual_seed(self.seed)
        return self._gen

    def __call__(self, x: torch.Tensor, sigma: float, sigma_next: float,
                 sigma_max: float, mask: torch.Tensor) -> torch.Tensor:
        if sigma <= 1e-7:
            return torch.zeros_like(x)
        dt = sigma_next - sigma
        scale = (abs(dt) ** 0.5) * (sigma ** 0.25) / (sigma_max ** 0.25)
        noise = torch.randn_like(x, generator=self._get_gen(x.device))
        return noise * scale * mask


# =========================================================
# UNIVERSAL SCHEDULERS
# =========================================================

def _flow_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    if abs(shift - 1.0) < 1e-4:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def get_ddrk_sigmas(scheduler_type: str, steps: int, sigma_min: float,
                    sigma_max: float, device: torch.device,
                    flow_shift: float = 3.0, warmup_steps: int = 0,
                    beta_a: float = 2.0, beta_b: float = 5.0) -> torch.Tensor:
    if scheduler_type == "ddrk_auto":
        scheduler_type = "ddrk_edm_karras" if sigma_max > 5.0 else "ddrk_cosine"

    if scheduler_type.startswith("ddrk_") and scheduler_type not in (
        "ddrk_edm_karras", "ddrk_edm_simple", "ddrk_edm_poly"
    ):
        if scheduler_type == "ddrk_cosine":
            angles = torch.linspace(0, math.pi / 2, steps, device=device)
            t = torch.cos(angles)
            t_shifted = _flow_shift(t, flow_shift)
            sigmas = t_shifted
        elif scheduler_type == "ddrk_beta":
            t_raw = torch.linspace(0, 1, steps, device=device)
            from torch.distributions import Beta
            beta_dist = Beta(torch.tensor(beta_a, device=device),
                             torch.tensor(beta_b, device=device))
            t_beta = 1.0 - beta_dist.icdf(t_raw.clamp(1e-6, 1 - 1e-6))
            t_shifted = _flow_shift(t_beta, flow_shift)
            sigmas = t_shifted
        elif scheduler_type == "ddrk_fewstep":
            shift_eff = max(flow_shift, 5.0)
            t = torch.linspace(1.0, 0.0, steps, device=device)
            sigmas = _flow_shift(t, shift_eff)
        elif scheduler_type == "ddrk_flow_cosmos":
            t = torch.linspace(1.0, 0.0, steps, device=device)
            t_shifted = _flow_shift(t, flow_shift)
            tail = (t_shifted < 0.3).float()
            sig = torch.sigmoid((t_shifted - 0.3) * -5.0)
            t_adj = t_shifted * (1.0 - tail * sig * 0.15)
            sigmas = t_adj
        else:
            t = torch.linspace(1.0, 0.0, steps, device=device)
            sigmas = _flow_shift(t, flow_shift)
            if steps > 4 and warmup_steps > 0:
                w = min(warmup_steps, max(1, steps // 8))
                for i in range(1, w + 1):
                    sigmas[i] *= 1.0 + 0.02 * (1.0 - (i - 1) / max(w, 1))
                for i in range(w, 0, -1):
                    sigmas[i] = min(sigmas[i], sigmas[i - 1] - 1e-7)
            sigmas[0] = 1.0

        sigmas = torch.cat([sigmas, torch.tensor([0.0], device=device)])
        return sigmas

    elif scheduler_type.startswith("ddrk_edm"):
        if scheduler_type == "ddrk_edm_karras":
            rho = 7.0
            ramp = torch.linspace(0, 1, steps + 1, device=device)
            sigmas = (sigma_max ** (1.0 / rho) +
                      ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
            sigmas = torch.clamp(sigmas, min=sigma_min)
            return sigmas
        elif scheduler_type == "ddrk_edm_poly":
            ramp = torch.linspace(0, 1, steps + 1, device=device)
            sigmas = sigma_max * (1.0 - ramp ** 2) + sigma_min * (ramp ** 2)
            sigmas = torch.where(sigmas < sigma_min,
                                 torch.tensor(sigma_min, device=device), sigmas)
            return sigmas
        else:
            ramp = torch.linspace(0, 1, steps + 1, device=device)
            sigmas = sigma_max * (sigma_min / sigma_max) ** ramp
            return sigmas

    if sigma_max > 5.0:
        ramp = torch.linspace(0, 1, steps + 1, device=device)
        return sigma_max * (sigma_min / sigma_max) ** ramp
    else:
        t = torch.linspace(1.0, 0.0, steps, device=device)
        s = _flow_shift(t, flow_shift)
        return torch.cat([s, torch.tensor([0.0], device=device)])


# =========================================================
# CORE SAMPLER
# =========================================================

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
    cfg_rescale = kwargs.get("cfg_rescale", 0.0)
    momentum_beta = kwargs.get("momentum_beta", 0.25)

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

    if total_steps <= 6:
        integrator = "euler"
        saber_fusion = min(saber_fusion, 0.15)
        sharpness = min(sharpness, 0.2)
        sde_strength = 0.0
    elif total_steps <= 10:
        if integrator == "rk4":
            integrator = "heun"

    state = SamplerState(total_steps=total_steps, is_edm=is_edm)
    router = AdaptivePhaseRouter(total_steps, sigma_max, is_edm)
    saber = SABER2(mode=saber_mode, buffer_size=3, fusion=saber_fusion,
                   ema_decay=ema_decay, use_ema=use_ema_saber)
    sde = AdaptiveSDE(seed=sde_seed)
    log_mask = LoGMask()

    s_in = x.new_ones([x.shape[0]], dtype=work_dtype, device=work_device)

    def model_fn(latent_in, sigma_val):
        out = model(latent_in.to(work_dtype), sigma_val * s_in, **extra_args).to(work_dtype)
        if is_edm and dyn_thresh_percentile < 1.0:
            out = dynamic_threshold(out, float(sigma_val), sigma_max, dyn_thresh_percentile)
        if cfg_rescale > 0:
            mean = out.mean(dim=(2, 3) if out.dim() == 4 else (2, 3, 4), keepdim=True)
            deviation = out - mean
            mask = deviation.abs() > 3.0
            scaled = deviation / (1.0 + cfg_rescale * deviation.abs())
            out = mean + torch.where(mask, scaled, deviation)
        return out

    preview_denoised = None
    progress_bar = trange(total_steps, disable=disable)

    for i in progress_bar:
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        state.step_count = i

        if float(sigma_curr) < 1e-7:
            break

        phase = router.get_phase(i)
        chosen_integrator = router.pick_integrator(i, float(sigma_curr), integrator)

        if phase == 1:
            if chosen_integrator == "rk4":
                x_next, preview_denoised, _ = rk4_step(x, sigma_curr, sigma_next, model_fn, state)
            elif chosen_integrator == "heun":
                x_next, preview_denoised, _ = heun_step(x, sigma_curr, sigma_next, model_fn, state)
            else:
                x_next, preview_denoised, _ = euler_step(x, sigma_curr, sigma_next, model_fn, state)

            if not is_edm and sde_strength > 0 and float(sigma_next) > 1e-7:
                edge_mask = log_mask(x_next)
                flat_mask = (1.0 - edge_mask) * sde_strength
                x_next = x_next + sde(x_next, float(sigma_curr), float(sigma_next), sigma_max, flat_mask)

            # v1.4.1: EDM gets very light SABER only on extremely early noise
            if is_edm and float(sigma_curr) > 0.55 * sigma_max:
                x_next = saber.fuse(x_next)

        elif phase == 2:
            x_next, preview_denoised, _ = euler_step(x, sigma_curr, sigma_next, model_fn, state)
            # v1.4.1: No SABER on phase 2 for EDM — mid-step blur causes artifacts
            if not is_edm:
                x_next = saber.fuse(x_next)

        else:
            x_next, preview_denoised, _ = euler_step(x, sigma_curr, sigma_next, model_fn, state)
            if saber_mode in ("video", "auto") and x_next.dim() == 5 and x_next.shape[2] > 1:
                x_next = saber.fuse(x_next)
            if i == total_steps - 1 and sharpness > 0:
                x_next = perceptual_sharpen(x_next, sharpness, is_final_step=True)

        # v1.4.1: Soft clamp for EDM — compresses extreme tails without killing micro-contrast
        if is_edm:
            x_next = _soft_clamp(x_next, bound=5.0, softness=0.25)

        x = x_next

        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma_curr,
                'sigma_next': sigma_next,
                'denoised': preview_denoised,
            })

    # v1.4.1: Wide final safety clamp — prevents VAE crash from rare explosions
    x = torch.clamp(x, -7.0, 7.0)
    return x


# =========================================================
# COMFYUI NODES
# =========================================================

class DDRKOmegaSchedulerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Auto-detects FM vs EDM by sigma_max"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "scheduler_type": ([
                    "ddrk_auto",
                    "ddrk_anima",
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
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_schedulers"

    def get_sigmas(self, model, steps, scheduler_type, flow_shift, warmup_steps):
        ms = model.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        device = ms.sigma_min.device
        sigmas = get_ddrk_sigmas(
            scheduler_type, steps, sigma_min, sigma_max,
            device=device, flow_shift=flow_shift, warmup_steps=warmup_steps
        )
        return (sigmas,)


class DDRKOmegaSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "integrator": (["auto", "rk4", "heun", "euler"], {"default": "auto"}),
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
                    "tooltip": "Stabilization. Auto-capped for EDM."
                }),
                "saber_mode": (["auto", "image", "video"], {
                    "default": "auto",
                    "tooltip": "Auto detects video by 5D latent with >1 frame."
                }),
                "use_ema_saber": ("BOOLEAN", {"default": True}),
                "ema_decay": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01}),
                "dyn_thresh_percentile": ("FLOAT", {
                    "default": 0.995, "min": 0.9, "max": 1.0, "step": 0.001,
                    "tooltip": "Dynamic thresholding percentile. 1.0 = disabled. EDM only."
                }),
                "cfg_rescale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Soft-rescale extreme latent values. 0 = disabled."
                }),
                "momentum_beta": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 0.8, "step": 0.05,
                    "tooltip": "Velocity EMA. Reduces oscillation. 0 = disabled."
                }),
                "sde_seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": "SDE noise seed. -1 = random (non-deterministic)."
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_samplers"

    def get_sampler(self, integrator, sde_strength, sharpness, saber_fusion,
                    saber_mode, use_ema_saber, ema_decay, dyn_thresh_percentile,
                    cfg_rescale, momentum_beta, sde_seed):
        extra = {
            "integrator": integrator,
            "sde_strength": sde_strength,
            "sharpness": sharpness,
            "saber_fusion": saber_fusion,
            "saber_mode": saber_mode,
            "use_ema_saber": use_ema_saber,
            "ema_decay": ema_decay,
            "dyn_thresh_percentile": dyn_thresh_percentile,
            "cfg_rescale": cfg_rescale,
            "momentum_beta": momentum_beta,
            "sde_seed": sde_seed if sde_seed >= 0 else None,
        }
        sampler = comfy.samplers.KSAMPLER(sample_ddrk_omega, extra_options=extra)
        return (sampler,)


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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scheduler_type": ([
                    "ddrk_auto", "ddrk_anima", "ddrk_cosine", "ddrk_beta",
                    "ddrk_flow_linear", "ddrk_flow_cosmos", "ddrk_fewstep",
                    "ddrk_edm_karras", "ddrk_edm_poly", "ddrk_edm_simple",
                ], {"default": "ddrk_auto"}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "integrator": (["auto", "rk4", "heun", "euler"], {"default": "auto"}),
                "sde_strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 5}),
            },
            "optional": {
                "saber_fusion": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "forceInput": True,
                }),
                "saber_mode": (["auto", "image", "video"], {"default": "auto"}),
                "use_ema_saber": ("BOOLEAN", {"default": True}),
                "ema_decay": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01}),
                "dyn_thresh_percentile": ("FLOAT", {"default": 0.995, "min": 0.9, "max": 1.0, "step": 0.001}),
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "momentum_beta": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.05}),
                "sde_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/unified_samplers"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               denoise, scheduler_type, flow_shift, integrator, sde_strength,
               sharpness, warmup_steps, saber_fusion=0.30, saber_mode="auto",
               use_ema_saber=True, ema_decay=0.7, dyn_thresh_percentile=0.995,
               cfg_rescale=0.0, momentum_beta=0.25, sde_seed=-1):
        latent = latent_image["samples"]
        noise_mask = latent_image.get("noise_mask", None)

        ms = model.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        device = comfy.model_management.get_torch_device()

        sigmas = get_ddrk_sigmas(
            scheduler_type, steps, sigma_min, sigma_max,
            device=device, flow_shift=flow_shift, warmup_steps=warmup_steps
        )

        if denoise < 1.0:
            steps_denoised = max(1, int(steps * denoise))
            sigmas = sigmas[-(steps_denoised + 1):]
            if sigmas[0] < sigma_max:
                sigmas = torch.cat([
                    torch.tensor([sigma_max], device=device, dtype=sigmas.dtype),
                    sigmas
                ])

        noise = comfy.sample.prepare_noise(latent, seed, None)
        if denoise < 1.0:
            latent = latent + noise * sigmas[0]

        extra = {
            "integrator": integrator,
            "sde_strength": sde_strength,
            "sharpness": sharpness,
            "saber_fusion": saber_fusion,
            "saber_mode": saber_mode,
            "use_ema_saber": use_ema_saber,
            "ema_decay": ema_decay,
            "dyn_thresh_percentile": dyn_thresh_percentile,
            "cfg_rescale": cfg_rescale,
            "momentum_beta": momentum_beta,
            "sde_seed": sde_seed if sde_seed >= 0 else None,
        }
        sampler_obj = comfy.samplers.KSAMPLER(sample_ddrk_omega, extra_options=extra)

        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler_obj, sigmas, positive, negative,
            latent_image=latent, denoise_mask=noise_mask,
            callback=None, disable_pbar=False, seed=seed
        )
        return ({"samples": samples},)


# =========================================================
# REGISTRATION
# =========================================================

NODE_CLASS_MAPPINGS = {
    "DDRKOmegaSchedulerNode": DDRKOmegaSchedulerNode,
    "DDRKOmegaSamplerNode": DDRKOmegaSamplerNode,
    "DDRKOmegaUnifiedKSamplerNode": DDRKOmegaUnifiedKSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDRKOmegaSchedulerNode": "DDRK Omega Scheduler",
    "DDRKOmegaSamplerNode": "DDRK Omega Sampler",
    "DDRKOmegaUnifiedKSamplerNode": "DDRK Omega Unified KSampler",
}
