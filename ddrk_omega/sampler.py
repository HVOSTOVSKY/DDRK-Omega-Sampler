"""
================================================================================
DDRK Omega Sampler — v1.5.0
ComfyUI | Flow Matching + EDM Universal Sampler

v1.5.0: Removed per-step soft_clamp for FM models — fixes "plastic fur/hair".
      EDM still clamped per Karras. guidance_embed hints, debug logging.
      Auto-Optimize for FM few-step (≤10 steps): disables SABER/SDE/momentum,
      forces euler + linear scheduler. SmartConfig node for debugging.
      Fixed UnifiedKSamplerNode denoise/latent handling to match ComfyUI KSampler.
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
    prev_denoised: Optional[torch.Tensor] = None
    prev_sigma: float = 0.0


class DeviceDtypeGuard:
    def __init__(self, target_dtype, target_device):
        self.dtype = target_dtype
        self.device = target_device

    def __call__(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.to(device=self.device, dtype=self.dtype)


# =========================================================
# MODEL FAMILY DETECTION
# =========================================================

def _detect_model_profile(model, latent_samples=None) -> dict:
    """
    Detect model architecture family and recommend SAMPLER parameters only.

    IMPORTANT: steps and cfg are CHECKPOINT properties, not architecture properties.
    They depend on training recipe, LoRA, distillation, and turbo variants.
    This function does NOT guess steps/cfg — use your checkpoint card or manual tuning.
    """
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

    # --- EDM vs Flow detection (reliable) ---
    is_edm = True
    try:
        ms = model.get_model_object("model_sampling")
        is_edm = float(ms.sigma_max) > 5.0
    except Exception:
        pass

    # --- Latent channels (rough family hint) ---
    latent_ch = 4
    if latent_samples is not None:
        try:
            latent_ch = int(latent_samples.shape[1])
        except Exception:
            pass

    # --- Architecture introspection via model.model.model_config ---
    family = "unknown"
    image_model = None
    guidance_embed = False
    try:
        # ModelPatcher wraps the real model; config lives on model.model
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
                # ComfyUI uses image_model to dispatch model classes
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
                    family = "fm"  # generic flow-matching
            else:
                # Fallback to legacy heuristics
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

    # --- Final fallback ---
    if family == "unknown":
        if is_edm:
            family = "edm"
        elif latent_ch == 16:
            family = "flux"  # or any 16ch FM model
        else:
            family = "fm"

    # Log raw values for debugging (user can verify substring matching)
    print(f"[DDRK Detect] raw_image_model={image_model!r}, family={family}, "
          f"guidance_embed={guidance_embed}, is_edm={is_edm}, latent_ch={latent_ch}")

    # --- Architecture-based recommendations (NOT steps/cfg) ---
    if family in ("sd15", "sd2"):
        profile.update(
            scheduler_type="ddrk_edm_karras",
            flow_shift=3.0,
            integrator="auto",
            sde_strength=0.0,
            sharpness=0.25,
            saber_fusion=0.20,
            momentum_beta=0.25,
            hint="SD1.5/SD2: EDM. Steps/CFG depend on checkpoint (base 20-30 / 7-8). Use turbo/distilled LoRA for 4-8 steps / CFG 1-2.",
        )
    elif family == "sdxl":
        profile.update(
            scheduler_type="ddrk_edm_karras",
            flow_shift=3.0,
            integrator="auto",
            sde_strength=0.0,
            sharpness=0.25,
            saber_fusion=0.20,
            momentum_beta=0.20,
            hint="SDXL: EDM. Steps/CFG depend on checkpoint (base 20-30 / 7-8). Use turbo/distilled for 4-8 steps / CFG 1-2.",
        )
    elif family in ("flux", "sd3", "qwen", "krea", "hidream", "chroma", "lumina"):
        profile.update(
            scheduler_type="ddrk_auto",
            flow_shift=1.0,
            integrator="euler",
            sde_strength=0.0,
            sharpness=0.10,
            saber_fusion=0.0,
            momentum_beta=0.0,
            hint=(f"{family.upper()}: Flow Matching. Steps/CFG vary wildly by checkpoint. "
                  f"{'Guidance embed detected — distilled variant, try CFG≈1.0, steps 4-8. ' if guidance_embed else ''}"
                  f"Check your model card."),
        )
    elif family == "fm":
        profile.update(
            scheduler_type="ddrk_auto",
            flow_shift=1.5,
            integrator="euler",
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
            sharpness=0.25,
            saber_fusion=0.20,
            momentum_beta=0.20,
            hint="Generic EDM: Steps/CFG depend on checkpoint (base 20-30 / 7-8). Use turbo/distilled for fewer steps / lower CFG.",
        )

    profile["family"] = family
    profile["guidance_embed"] = guidance_embed
    return profile


# =========================================================
# DYNAMIC THRESHOLDING
# =========================================================

def dynamic_threshold(denoised: torch.Tensor, sigma: float, sigma_max: float,
                      percentile: float = 0.995, min_val: float = 1.0,
                      edm_ratio: float = 0.30) -> torch.Tensor:
    threshold_ratio = 0.40 if sigma_max <= 5.0 else edm_ratio
    if sigma > threshold_ratio * sigma_max or denoised.numel() == 0:
        return denoised

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
                edge_density = torch.sigmoid((combined - combined.mean()) * 15.0)
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
                    edge_density = torch.sigmoid((combined - combined.mean()) * 15.0)
                    fusion_weight = self.fusion * (1.0 - edge_density)
                    fusion_weight = fusion_weight.unsqueeze(2)
                else:
                    blur3 = F.avg_pool3d(F.pad(x, (1,1,1,1,1,1), mode='reflect'), 3, stride=1)
                    edge3 = (x - blur3).abs()
                    edge_density = torch.sigmoid((edge3 - edge3.mean()) * 20.0)
                    fusion_weight = self.fusion * (1.0 - edge_density)

        if is_vid:
            stacked = torch.stack(buf["frames"], dim=0)
            var = torch.var(stacked, dim=0)
            chaos = torch.sigmoid((var - var.mean()) * 10.0)
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


# =========================================================
# INTEGRATORS
# =========================================================

def _safe_sigma(s: Union[float, torch.Tensor]) -> float:
    return max(float(s), 1e-8)


def euler_step(x, sigma, sigma_next, model_fn, state: SamplerState, momentum_beta: float = None):
    denoised = model_fn(x, sigma)
    d = (x - denoised) / _safe_sigma(sigma)
    d = _apply_momentum(d, state, momentum_beta=momentum_beta)
    dt = sigma_next - sigma
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
        d_avg = _apply_momentum(d_avg, state, force=True, momentum_beta=momentum_beta)
        x_next = x + d_avg * dt
        return x_next, denoised_2, d_avg

    _apply_momentum(d, state, momentum_beta=momentum_beta)
    return x_next, denoised, d


def rk4_step(x, sigma, sigma_next, model_fn, state: SamplerState, momentum_beta: float = None):
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
    d_final = _apply_momentum(d_final, state, force=True, momentum_beta=momentum_beta)
    return x + d_final * dt, denoised_4, d_final


def _apply_momentum(d: torch.Tensor, state: SamplerState, force: bool = False,
                    beta: float = 0.25, momentum_beta: float = None) -> torch.Tensor:
    if momentum_beta is not None:
        beta = momentum_beta
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
                    beta_a: float = 2.0, beta_b: float = 5.0,
                    auto_optimize: bool = True) -> torch.Tensor:
    is_edm = sigma_max > 5.0

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
            weight = torch.sigmoid((0.3 - t_shifted) * 20.0)
            sig = torch.sigmoid((t_shifted - 0.3) * -5.0)
            t_adj = t_shifted * (1.0 - weight * sig * 0.15)
            sigmas = t_adj
        else:
            t = torch.linspace(1.0, 0.0, steps, device=device)
            sigmas = _flow_shift(t, flow_shift)
            if steps > 4 and warmup_steps > 0:
                w = min(warmup_steps, max(1, steps // 8))
                for i in range(1, w + 1):
                    sigmas[i] *= 1.0 + 0.02 * (1.0 - (i - 1) / max(w, 1))
                for i in range(1, w + 1):
                    sigmas[i] = min(sigmas[i], sigmas[i - 1] - 1e-7)
            sigmas[0] = 1.0

        sigmas = torch.cat([sigmas, torch.tensor([0.0], device=device)])
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
    s_churn = kwargs.get("s_churn", 0.0)
    s_tmin = kwargs.get("s_tmin", 0.0)
    s_tmax = float('inf')
    s_noise = kwargs.get("s_noise", 1.0)
    auto_optimize = kwargs.get("auto_optimize", True)

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

    if auto_optimize and not is_edm:
        if total_steps <= 10:
            saber_fusion = 0.0
            sde_strength = 0.0
            momentum_beta = 0.0
            sharpness = min(sharpness, 0.12)
            if integrator == "auto":
                integrator = "euler"
            print(f"[DDRK Auto] FM few-step ({total_steps} steps): "
                  f"SABER=0, SDE=0, momentum=0, sharp={sharpness:.2f}, "
                  f"integrator={integrator}, scheduler=linear, shift<=1.0")
        elif total_steps <= 20:
            saber_fusion = min(saber_fusion, 0.05)
            sde_strength = 0.0
            momentum_beta = min(momentum_beta, 0.10)
            sharpness = min(sharpness, 0.15)
            print(f"[DDRK Auto] FM mid-step ({total_steps} steps): "
                  f"SABER<=0.05, SDE=0, momentum<=0.10, sharp<=0.15")

    if total_steps <= 6:
        integrator = "euler"
        saber_fusion = min(saber_fusion, 0.15)
        sharpness = min(sharpness, 0.2)
        sde_strength = 0.0
    if not is_edm:
        if total_steps <= 10:
            sharpness = min(sharpness, 0.12)
        else:
            sharpness = min(sharpness, 0.15)
        saber_fusion = min(saber_fusion, 0.15)
        momentum_beta = min(momentum_beta, 0.15)
    elif total_steps <= 10:
        if integrator == "rk4":
            integrator = "heun"

    state = SamplerState(total_steps=total_steps, is_edm=is_edm)
    router = AdaptivePhaseRouter(total_steps, sigma_max, is_edm)
    content_aware = kwargs.get("content_aware", True)
    saber = SABER2(mode=saber_mode, buffer_size=3, fusion=saber_fusion,
                   ema_decay=ema_decay, use_ema=use_ema_saber,
                   content_aware=content_aware)
    sde = AdaptiveSDE(seed=sde_seed)
    log_mask = LoGMask()

    s_in = x.new_ones([x.shape[0]], dtype=work_dtype, device=work_device)

    def model_fn(latent_in, sigma_val):
        out = model(latent_in.to(work_dtype), sigma_val * s_in, **extra_args).to(work_dtype)
        if dyn_thresh_percentile < 1.0:
            out = dynamic_threshold(out, float(sigma_val), sigma_max, dyn_thresh_percentile)
        if cfg_rescale > 0:
            dims = (2, 3) if out.dim() == 4 else (2, 3, 4)
            mean = out.mean(dim=dims, keepdim=True)
            std = out.std(dim=dims, keepdim=True).clamp_min(1e-8)
            deviation = out - mean
            scale_factor = 1.0 / (1.0 + cfg_rescale * (deviation.abs() / std - 2.0).clamp_min(0.0))
            out = mean + deviation * scale_factor
        return out

    preview_denoised = None
    progress_bar = trange(total_steps, disable=disable)

    for i in progress_bar:
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        state.step_count = i

        if float(sigma_curr) < 1e-7:
            break

        if is_edm and s_churn > 0 and s_tmin <= float(sigma_curr) <= s_tmax:
            gamma = min(s_churn / float(sigma_curr), math.sqrt(2) - 1)
            sigma_hat = float(sigma_curr) * (1.0 + gamma)
            noise = torch.randn_like(x) * s_noise
            x = x + noise * math.sqrt(sigma_hat ** 2 - float(sigma_curr) ** 2)
            sigma_curr = torch.tensor(sigma_hat, device=work_device, dtype=torch.float32)

        phase = router.get_phase(i)
        chosen_integrator = router.pick_integrator(i, float(sigma_curr), integrator, state)

        if phase == 1:
            if chosen_integrator == "rk4":
                x_next, preview_denoised, _ = rk4_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)
            elif chosen_integrator == "heun":
                x_next, preview_denoised, _ = heun_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)
            else:
                x_next, preview_denoised, _ = euler_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)

            if not is_edm and sde_strength > 0 and float(sigma_next) > 1e-7:
                edge_mask = log_mask(x_next)
                entropy = local_entropy_mask(x_next, window=5)
                flat_mask = (1.0 - edge_mask) * entropy * sde_strength
                x_next = x_next + sde(x_next, float(sigma_curr), float(sigma_next), sigma_max, flat_mask)

            if is_edm and float(sigma_curr) > 0.55 * sigma_max:
                x_next = saber.fuse(x_next)

        elif phase == 2:
            if chosen_integrator == "rk4":
                x_next, preview_denoised, _ = rk4_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)
            elif chosen_integrator == "heun":
                x_next, preview_denoised, _ = heun_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)
            else:
                x_next, preview_denoised, _ = euler_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)
            if not is_edm:
                x_next = saber.fuse(x_next)

        else:
            if chosen_integrator == "rk4":
                x_next, preview_denoised, _ = rk4_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)
            elif chosen_integrator == "heun":
                x_next, preview_denoised, _ = heun_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)
            else:
                x_next, preview_denoised, _ = euler_step(x, sigma_curr, sigma_next, model_fn, state, momentum_beta=momentum_beta)
            if saber_mode in ("video", "auto") and x_next.dim() == 5 and x_next.shape[2] > 1:
                x_next = saber.fuse(x_next)
            if i == total_steps - 1 and sharpness > 0:
                x_next = perceptual_sharpen(x_next, sharpness, is_final_step=True)

        # Soft clamp: EDM benefits from it (Karras recommendation), FM does not.
        # FM latents rely on fine-grained extremes for texture chaos (fur, hair, water).
        # Stock KSampler never clamps between steps — we match that for FM.
        if is_edm:
            x_next = _soft_clamp(x_next, bound=5.0, softness=0.25)

        if preview_denoised is not None and state.prev_denoised is not None:
            with torch.no_grad():
                diff = (preview_denoised - state.prev_denoised).abs().mean()
                base = state.prev_denoised.abs().mean().clamp_min(1e-8)
                dsigma = max(abs(float(sigma_curr) - state.prev_sigma), 1e-8)
                state.curvature = float((diff / base / dsigma).item())
        else:
            state.curvature = 0.0

        state.prev_denoised = preview_denoised.detach().clone() if preview_denoised is not None else None
        state.prev_sigma = float(sigma_curr)

        x = x_next

        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma_curr,
                'sigma_next': sigma_next,
                'denoised': preview_denoised,
            })

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
                "auto_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-select scheduler & flow_shift for FM few-step. Disable for full manual control."
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_schedulers"

    def get_sigmas(self, model, steps, scheduler_type, flow_shift, warmup_steps, auto_optimize):
        ms = model.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        device = ms.sigma_min.device
        sigmas = get_ddrk_sigmas(
            scheduler_type, steps, sigma_min, sigma_max,
            device=device, flow_shift=flow_shift, warmup_steps=warmup_steps,
            auto_optimize=auto_optimize
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
                "s_churn": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "EDM churn (Karras Alg 2). 0 = off. Try 5-15 for SDXL. EDM models only; silently ignored for Flow Matching."
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
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_samplers"

    def get_sampler(self, integrator, sde_strength, sharpness, saber_fusion,
                    saber_mode, use_ema_saber, ema_decay, dyn_thresh_percentile,
                    cfg_rescale, momentum_beta, sde_seed, s_churn, s_noise,
                    content_aware, auto_optimize):
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
            "s_churn": s_churn,
            "s_noise": s_noise,
            "content_aware": content_aware,
            "auto_optimize": auto_optimize,
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
                "saber_fusion": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stabilization blur. 0 = disabled. Disable for text/graphics."
                }),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 5}),
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
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "momentum_beta": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.05}),
                "sde_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "s_churn": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "EDM churn (Karras Alg 2). 0 = off. Try 5-15 for SDXL. EDM models only; silently ignored for Flow Matching."
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EDM churn noise multiplier. 1.0 = standard."
                }),
                "content_aware": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Content-aware SABER — edge-gated fusion. Disable for pixel-art/flat styles."
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
               ema_decay=0.7, dyn_thresh_percentile=0.995, cfg_rescale=0.0,
               momentum_beta=0.25, sde_seed=-1, s_churn=0.0, s_noise=1.0,
               content_aware=True):
        # Keep the full latent dict to preserve metadata (noise_mask, etc.)
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        noise_mask = latent.get("noise_mask", None)

        # Fix channel mismatch (matches stock KSampler / SamplerCustom path)
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        # Smart Defaults override — architecture only, NOT steps/cfg
        # Steps and CFG are checkpoint properties (training, LoRA, distillation).
        # They cannot be inferred from topology. Use your model card or manual tuning.
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

        # Match ComfyUI KSampler denoise logic: recalc total steps BEFORE sigmas
        steps_denoised = steps
        if denoise < 1.0:
            steps = max(1, int(steps / denoise))

        ms = model.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        device = ms.sigma_min.device  # consistent with SchedulerNode

        sigmas = get_ddrk_sigmas(
            scheduler_type, steps, sigma_min, sigma_max,
            device=device, flow_shift=flow_shift, warmup_steps=warmup_steps,
            auto_optimize=auto_optimize
        )

        noise = comfy.sample.prepare_noise(latent_samples, seed, None)

        if denoise < 1.0:
            sigmas = sigmas[-(steps_denoised + 1):]
            # Do NOT manually scale latent_samples here — KSampler.sample()
            # internally calls model_sampling.noise_scaling(sigmas[0], noise, latent_image)
            # which already mixes noise and latent correctly.

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
            "s_churn": s_churn,
            "s_noise": s_noise,
            "content_aware": content_aware,
            "auto_optimize": auto_optimize,
        }
        sampler_obj = comfy.samplers.KSAMPLER(sample_ddrk_omega, extra_options=extra)

        # Live preview — graceful fallback for older ComfyUI builds
        try:
            import comfy.latent_preview as lp
            callback = lp.prepare_callback(model, steps)
        except Exception:
            callback = None

        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler_obj, sigmas, positive, negative,
            latent_image=latent_samples, noise_mask=noise_mask,
            callback=callback, disable_pbar=False, seed=seed
        )

        out = latent.copy()
        out["samples"] = samples
        return (out,)


# =========================================================
# REGISTRATION
# =========================================================

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
