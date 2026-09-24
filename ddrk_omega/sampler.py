"""
DDRK Omega Sampler v1.11.0
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

    # HC2 time variable. False: lambda = -log(sigma), the variance-exploding
    # form every HC2 release before 1.9.0 used on every model. True: lambda =
    # log((1 - sigma) / sigma), the half-log-SNR of a Flow Matching model
    # x = (1 - sigma) x0 + sigma eps, with the (1 - sigma_next) factor on the
    # high-order terms that the exact data-prediction integral carries.
    hc2_flow: bool = False

    # Denoiser output at the START of the last step, whatever integrator took
    # it. Every integrator evaluates D(x, sigma) first; since 1.10.0 that
    # value also feeds HC2's history, so an HC2 step that follows a Heun or
    # RK4 step in auto mode extrapolates from the previous step instead of
    # from whichever step last happened to be HC2.
    last_D: Optional[torch.Tensor] = None

    # Zero-cost corrector (hc2_free_corrector): what the previous HC2 step
    # needs to be corrected once the denoiser at its end point is known.
    hc2_pc: Optional[Dict[str, Any]] = None
    hc2_free_corr: float = -1.0

    # HC3: mean extrapolation trust of the last step (telemetry only).
    hc3_theta: float = -1.0

    # False skips the diagnostics that need a GPU->CPU sync (limiter
    # fraction, activity) when nothing reads them.
    hc2_stats: bool = True


# Clamp for the Flow Matching log-SNR at sigma = 1, where it is -inf. Same
# offset ComfyUI uses for its own RF multistep samplers.
_FLOW_SIGMA_CAP = 1.0 - 1e-4


def _hc2_lambda(sigma: float, flow: bool) -> float:
    """HC2 time variable: -log(sigma) on EDM, half-log-SNR on Flow Matching."""
    s = max(float(sigma), 1e-7)
    if flow:
        s = min(s, _FLOW_SIGMA_CAP)
        return math.log((1.0 - s) / s)
    return -math.log(s)


def _phi2(h: float) -> float:
    """h - 1 + e^-h: weight of the first derivative of D over a step of h."""
    if abs(h) < 1e-3:
        return h * h / 2.0 - h ** 3 / 6.0 + h ** 4 / 24.0
    return (h - 1.0) + math.exp(-h)


def _phi3(h: float) -> float:
    """h^2/2 - h + 1 - e^-h: weight of the second derivative of D over h."""
    if abs(h) < 1e-3:
        return h ** 3 / 6.0 - h ** 4 / 24.0 + h ** 5 / 120.0
    return (h * h / 2.0 - h + 1.0) - math.exp(-h)


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


    # Each introspection step below is best-effort by design, but a silent
    # fallback can pick the wrong family halfway through a sweep and you would
    # compare an FM profile against an EDM profile without ever knowing. The
    # fallbacks stay; they now announce themselves and are recorded in the
    # profile so the telemetry shows which path was taken.
    detect_warnings = []

    is_edm = True
    try:
        ms = model.get_model_object("model_sampling")
        is_edm = float(ms.sigma_max) > 5.0
    except Exception as e:
        detect_warnings.append(f"model_sampling unreadable ({type(e).__name__}: {e}); "
                               f"assuming EDM")


    latent_ch = 4
    if latent_samples is not None:
        try:
            latent_ch = int(latent_samples.shape[1])
        except Exception as e:
            detect_warnings.append(f"latent shape unreadable ({type(e).__name__}: {e}); "
                                   f"assuming {latent_ch} channels")


    family = "unknown"
    image_model = None
    guidance_embed = False
    model_class = ""
    try:

        inner_model = getattr(model, "model", None)
        cfg = getattr(inner_model, "model_config", {}) if inner_model is not None else {}
        # ComfyUI's supported_models class (SD15, SDXL, SD3, Flux,
        # FluxSchnell, QwenImage, ...): the most direct statement of what the
        # model is, and the only one for families whose unet_config carries
        # no image_model.
        if cfg is not None and not isinstance(cfg, dict):
            model_class = type(cfg).__name__
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
                    # An image_model this table does not know. Cosmos,
                    # PixArt, Hunyuan-DiT and CogVideoX are EDM-type, so
                    # the sigma range decides, not the mere presence of
                    # the key (up to 1.10.0 every one of them became "fm").
                    family = "edm" if is_edm else "fm"
            elif model_class.startswith("SD3"):
                # SD3 / SD3.5 have no image_model and no context_dim, so up
                # to 1.10.0 they fell through to the 16-channel heuristic
                # and were reported as Flux.
                family = "sd3"
            else:

                if adm == 2816:
                    family = "sdxl"
                elif ctx == 768:
                    family = "sd15"
                elif ctx == 1024:
                    family = "sd2"
                elif ctx in (2048, 4096):
                    family = "flux"
    except Exception as e:
        detect_warnings.append(f"model config introspection failed "
                               f"({type(e).__name__}: {e}); falling back to heuristics")


    detected_by = "model_config"
    if family == "unknown":
        detected_by = "heuristic"
        detect_warnings.append("family not identified from the model config; "
                               "guessing from sigma_max and latent channel count")
    if family == "unknown":
        if is_edm:
            family = "edm"
        elif latent_ch == 16:
            family = "flux"
        else:
            family = "fm"


    print(f"[DDRK Detect] model_class={model_class or '?'}, raw_image_model={image_model!r}, family={family}, "
          f"guidance_embed={guidance_embed}, is_edm={is_edm}, latent_ch={latent_ch}, "
          f"detected_by={detected_by}")
    for w in detect_warnings:
        print(f"[DDRK Detect] WARNING: {w}")


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


            # hc3: HC2 (the most accurate at equal model calls in the 1.9.0
            # image A/B on FM) plus trust damping and the zero-cost corrector,
            # same one call per step. auto spends 2-4 calls per step early.
            integrator="hc3",
            sde_strength=0.0,
            sharpness=0.10,
            saber_fusion=0.0,
            momentum_beta=0.0,
            hint=(f"{family.upper()}: Flow Matching. Steps/CFG vary wildly by checkpoint. "
                  f"{'Guidance embed detected — distilled variant, try CFG≈1.0, steps 4-8. ' if guidance_embed else ''}"
                  f"Check your model card. Integrator: hc3 (one model call per step, "
                  f"third order, self-damping at high CFG). "
                  f"Use the model's native resolution (~1 MP)."),
        )
    elif family == "fm":
        profile.update(
            scheduler_type="ddrk_auto",
            flow_shift=1.5,

            integrator="hc3",
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
    profile["model_class"] = model_class
    profile["guidance_embed"] = guidance_embed
    profile["detected_by"] = detected_by
    profile["detect_warnings"] = detect_warnings
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
    # Statistics are taken PER BATCH ITEM. Reducing over the whole tensor
    # (torch.std_mean(t) with no dim) pools every image in the batch into one
    # mean and one std, so image A's content shifts image B's mask. Measured
    # effect before this fix: up to 2.4e-2 max-abs drift in item 0 when only
    # item 1 changed, roughly four orders of magnitude above the 1e-6 GPU
    # noise floor. It also made batch=1 and batch=4 disagree for the same seed.
    # At batch size 1 this is a bit-exact no-op, so no earlier measurement
    # taken at batch=1 is invalidated.
    dims = tuple(range(1, t.dim()))
    std, mean = torch.std_mean(t, dim=dims, keepdim=True)
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

    # Restore the 5D shape BEFORE gating. _zscore_gate normalises per batch
    # item, and while the frames are folded into the batch dimension that
    # would silently become per FRAME - changing video behaviour and inviting
    # temporal flicker in the SDE mask. Gating on the restored shape keeps the
    # statistic pooled across frames, exactly as before, while still keeping
    # separate images in a batch apart.
    if is_5d:
        var = var.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
    flatness = 1.0 - _zscore_gate(var)
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
    state.last_D = denoised
    d = (x - denoised) / _safe_sigma(sigma)
    dt = sigma_next - sigma
    d = _ab2_extrapolate(d, state, dt=float(dt), beta=momentum_beta)
    return x + d * dt, denoised, d


# heun_step and rk4_step accept momentum_beta for call compatibility but no
# longer apply it (1.10.0). They still record their slope, so an Euler step
# that follows them in auto mode extrapolates from the right history.
def heun_step(x, sigma, sigma_next, model_fn, state: SamplerState, momentum_beta: float = None):
    denoised = model_fn(x, sigma)
    state.last_D = denoised
    d = (x - denoised) / _safe_sigma(sigma)
    dt = sigma_next - sigma
    x_next = x + d * dt

    if float(sigma_next) > 1e-7:
        denoised_2 = model_fn(x_next, sigma_next)
        d2 = (x_next - denoised_2) / _safe_sigma(sigma_next)
        d_avg = (d + d2) * 0.5
        _ab2_extrapolate(d_avg, state, dt=float(dt), beta=0.0)
        x_next = x + d_avg * dt
        return x_next, denoised_2, d_avg

    _ab2_extrapolate(d, state, dt=float(dt), beta=0.0)
    return x_next, denoised, d


def rk4_step(x, sigma, sigma_next, model_fn, state: SamplerState, momentum_beta: float = None):
    dt = sigma_next - sigma
    s = _safe_sigma(sigma)

    if float(sigma_next) <= 1e-7:


        denoised = model_fn(x, sigma)
        state.last_D = denoised
        d = (x - denoised) / s
        x_next = x + d * dt
        _ab2_extrapolate(d, state, dt=float(dt), beta=0.0)
        return x_next, denoised, d

    s_mid = _safe_sigma(sigma + dt * 0.5)
    s_next = _safe_sigma(sigma_next)

    denoised_1 = model_fn(x, sigma)
    state.last_D = denoised_1
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
    _ab2_extrapolate(d_final, state, dt=float(dt), beta=0.0)
    return x + d_final * dt, denoised_4, d_final


def _hc2_free_correction(pc: Dict[str, Any], denoised: torch.Tensor,
                         lam: float) -> Optional[torch.Tensor]:
    """Zero-cost corrector for the HC2 step that just ended (UniC-style).

    The predictor had to EXTRAPOLATE the denoiser across its step from past
    evaluations. Once the model has been called at the step's end point, the
    same step can be redone by INTERPOLATING between known values instead -
    linear through D_prev and D_now, quadratic when one more past value is
    available - without another model call. The previous HC2 step's
    correction is swapped for the interpolated one; the result is returned
    as a delta so anything applied to the latent in between (SABER, clamps)
    is kept. The next step uses the model output taken at the uncorrected
    point, as UniPC does (Zhao et al. 2023); that costs nothing in order.

    Measured on the analytic mixture bench (tests/bench_analytic.py) at equal
    model calls, EDM Karras: error 1.1-2x lower than HC2 at 8-12 calls,
    1.3-4x at 20-30, and the order rises from 2 to ~3. On Flow Matching with
    the model's schedule it is within a few percent, because there the final
    one-shot jump to sigma = 0 dominates the error and no multistep
    correction reaches it. Not validated on images.
    """
    h = lam - pc["lam"]
    if abs(lam - pc["lam_next"]) > 1e-6 or abs(h) < 1e-8:
        # The latent did not arrive here by that step (SDE re-noising,
        # churn, a restart jump, another integrator): nothing to correct.
        return None
    alpha = pc["alpha_next"]
    r_a = (denoised - pc["D"]) / h
    corr = r_a * (alpha * _phi2(h))
    if pc["D2"] is not None:
        h_b = pc["lam"] - pc["lam2"]
        if abs(h_b) > 1e-8:
            r_b = (pc["D"] - pc["D2"]) / h_b
            dd = (r_a - r_b) / (h + h_b)
            corr = corr + dd * (alpha * (2.0 * _phi3(h) - h * _phi2(h)))
    applied = pc["applied"]
    return corr - applied if applied is not None else corr


def _hc3_trust(r: torch.Tensor, state: SamplerState) -> Optional[torch.Tensor]:
    """HC3's extrapolation trust, per batch item, in [0, 1].

    HC2 extrapolates the denoiser across the next step with the slope r of
    its last two evaluations. That is second order when the trajectory is
    resolved and an overshoot when it is not - high CFG at few steps, where
    the guided denoiser swings between evaluations. On the analytic bench at
    CFG 6 and 5-10 model calls every multistep method, HC2 included, was
    1.2-4.5x less accurate than plain Euler for exactly this reason.

    The slope history says which case applies. When r agrees with the slope
    before it, the linear model of D is holding; when it has turned or
    jumped, it is not. With rho = |r - r_prev| / (|r| + |r_prev|) (norms per
    batch item), theta = 1 - rho scales the extrapolation: ~1 on a smooth
    trajectory, where rho is O(h) and the method keeps its order, and ~0 when
    consecutive slopes disagree, where the step falls back to DDIM - exact
    for a denoiser that is constant over the step. A causal, per-image
    version of the "lower order when unsure" rule that DPM-Solver and UniPC
    apply only by step index.
    """
    if (state.hc2_D_prev is None or state.hc2_D_prev2 is None
            or state.hc2_lambda_prev is None or state.hc2_lambda_prev2 is None):
        return None
    h_b = state.hc2_lambda_prev - state.hc2_lambda_prev2
    if abs(h_b) < 1e-8:
        return None
    r_prev = (state.hc2_D_prev - state.hc2_D_prev2) / h_b
    dims = tuple(range(1, r.dim()))
    n_r = torch.linalg.vector_norm(r, dim=dims, keepdim=True)
    n_p = torch.linalg.vector_norm(r_prev, dim=dims, keepdim=True)
    n_d = torch.linalg.vector_norm(r - r_prev, dim=dims, keepdim=True)
    theta = (1.0 - n_d / (n_r + n_p).clamp_min(1e-12)).clamp(0.0, 1.0)
    if state.hc2_stats:
        try:
            state.hc3_theta = float(theta.mean().item())
        except Exception:
            state.hc3_theta = -1.0
    return theta


def hc2_step(x, sigma, sigma_next, model_fn, state: SamplerState,
             momentum_beta: float = None, limiter_kappa: float = 1.0,
             max_order: int = 2, corrector_thresh: float = 0.0,
             free_corrector: bool = False, damping: bool = False):
    s = _safe_sigma(sigma)
    sigma_next_f = float(sigma_next)

    denoised = model_fn(x, sigma)
    state.last_D = denoised
    flow = bool(getattr(state, "hc2_flow", False))
    lam = _hc2_lambda(s, flow)

    pc, state.hc2_pc = state.hc2_pc, None
    state.hc2_free_corr = -1.0
    state.hc3_theta = -1.0
    # On the final step the output is the denoiser itself, so correcting x
    # there would change nothing.
    if free_corrector and pc is not None and sigma_next_f > 1e-7:
        delta = _hc2_free_correction(pc, denoised, lam)
        if delta is not None:
            x = x + delta
            if state.hc2_stats:
                try:
                    state.hc2_free_corr = float(delta.abs().mean().item())
                except Exception:
                    state.hc2_free_corr = -1.0

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

    h = _hc2_lambda(sigma_next_f, flow) - lam
    exp_neg_h = math.exp(-h)
    # Weight of the high-order terms. The exact data-prediction integral is
    #   x_t = (s_t/s_s) x_s + alpha_t (1 - e^-h) D + alpha_t (h - 1 + e^-h) D' + ...
    # with alpha = 1 on EDM and alpha = 1 - sigma on Flow Matching. Before
    # 1.9.0 HC2 used alpha = 1 and lambda = -log(sigma) on FM too, which
    # overweights the correction exactly where alpha is small - the
    # high-noise steps, where the live Krea 2 logs showed 36-43% of elements
    # hitting the limiter.
    alpha_next = (1.0 - sigma_next_f) if flow else 1.0

    # First order is the same in both parameterisations (it is DDIM, which on
    # FM is exact Euler): alpha_t (1 - e^-h) == 1 - sigma_t / sigma_s.
    # The EDM path keeps the exp(-h) spelling so it stays bit-exact with 1.8.0.
    first_order = ((1.0 - sigma_next_f / s) if flow
                   else (1.0 - exp_neg_h)) * (denoised - x)
    x_next = x + first_order
    order_used = 1
    applied = None
    bootstrapped = False
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
        bootstrapped = True

    elif has_1 and abs(h_prev) > 1e-8:
        r = (denoised - state.hc2_D_prev) / h_prev
        r_raw = r
        theta = None
        if damping:
            theta = _hc3_trust(r, state)
            if theta is not None:
                r = r * theta
        corr2 = r * (alpha_next * ((h - 1.0) + exp_neg_h))
        correction = corr2
        order_used = 2

        has_2 = (max_order >= 3 and state.hc2_D_prev2 is not None
                 and state.hc2_lambda_prev2 is not None)
        if has_2:
            h_prev2 = state.hc2_lambda_prev - state.hc2_lambda_prev2
            if abs(h_prev2) > 1e-8:


                d_prev = (state.hc2_D_prev - state.hc2_D_prev2) / h_prev2
                dd = (r_raw - d_prev) / (h_prev + h_prev2)
                if theta is not None:
                    # Curvature is a difference of slopes: trust it less.
                    dd = dd * (theta * theta)
                corr3 = dd * (alpha_next * ((h * h - 2.0 * h + 2.0)
                                            - 2.0 * exp_neg_h
                                            + h_prev * ((h - 1.0) + exp_neg_h)))


                try:
                    n3 = float(corr3.abs().mean().item())
                    n2 = float(corr2.abs().mean().item())
                    ratio = n3 / max(n2, 1e-12)
                except Exception:
                    ratio = float('inf')
                state.hc2_err_ratio = ratio
                if ratio < 1.0:


                    corr3 = torch.clamp(corr3, -corr2.abs(), corr2.abs())
                    correction = corr2 + corr3
                    order_used = 3

        bound = limiter_kappa * first_order.abs()
        limited = torch.clamp(correction, -bound, bound)
        if state.hc2_stats or corrector_thresh > 0.0:
            try:
                state.hc2_limited_frac = float(
                    (correction.abs() > bound).float().mean().item())


                fo_mean = float(first_order.abs().mean().item())
                x_scale = float(x.abs().mean().item())
                denom = max(fo_mean, 1e-3 * max(x_scale, 1e-8))
                state.hc2_activity = min(
                    float(correction.abs().mean().item()) / denom, 100.0)
            except Exception:
                state.hc2_limited_frac = -1.0
        x_next = x_next + limited
        applied = limited

        if corrector_thresh > 0.0 and state.hc2_activity > corrector_thresh:


            denoised_c = model_fn(x_next, sigma_next)
            r_c = (denoised_c - denoised) / h
            corr_c = r_c * (alpha_next * ((h - 1.0) + exp_neg_h))
            corr_c = torch.clamp(corr_c, -bound, bound)
            x_next = x + first_order + corr_c
            applied = corr_c
            state.hc2_corrector_fired = True

    if free_corrector and not bootstrapped:
        state.hc2_pc = {
            "lam": lam, "lam_next": lam + h, "alpha_next": alpha_next,
            "D": denoised, "applied": applied,
            "D2": state.hc2_D_prev if has_1 else None,
            "lam2": state.hc2_lambda_prev if has_1 else None,
        }

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
    """Adams-Bashforth 2 slope extrapolation (momentum_beta), Euler only.

    beta = 1 turns Euler into variable-step AB2, a genuine second-order
    method; smaller values interpolate between the two. Up to 1.9.0 the same
    extrapolation was also applied on top of Heun and RK4, where it adds an
    O(dt) term to a slope that is already accurate to O(dt^2) or O(dt^4),
    which makes both methods first order. Measured on the analytic mixture
    bench (tests/analytic.py, lambda-uniform schedule, 128-256 calls): RK4
    went from order 4.0 to 0.9 at beta = 0.25 and its error grew ~250x;
    Heun from order 2.0 to 1.1-1.2, error 3-5x. Heun and RK4 now call this
    with beta = 0 so that it only records their slope.
    """
    if momentum_beta is not None:
        beta = momentum_beta
    if beta is None:
        beta = 0.0

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
                            return "hc2"
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
                return "hc2"
        return cfg


def _ancestral_split(sigma: float, sigma_next: float, eta: float) -> Tuple[float, float]:
    if sigma <= 1e-9 or sigma_next <= 1e-9 or eta <= 0.0:
        return sigma_next, 0.0
    eta = min(eta, 1.0)
    var_diff = max(sigma ** 2 - sigma_next ** 2, 0.0)
    sigma_up = min(sigma_next, eta * math.sqrt((sigma_next ** 2) * var_diff) / sigma)
    sigma_down = math.sqrt(max(sigma_next ** 2 - sigma_up ** 2, 0.0))
    return sigma_down, sigma_up


def _ancestral_split_flow(sigma: float, sigma_next: float,
                          eta: float) -> Tuple[float, float, float]:
    """Ancestral step for Flow Matching: (sigma_down, alpha_ratio, renoise).

    _ancestral_split is the variance-exploding recipe: step to sigma_down,
    then add noise of std sigma_up. On FM, x = (1 - s) x0 + s eps, that
    over-noises and never rescales the signal part. The FM recipe (the one
    ComfyUI's euler_ancestral_RF uses) steps to sigma_down, scales by
    (1 - sigma_next) / (1 - sigma_down), and re-noises so the total noise
    std lands exactly on sigma_next.
    """
    if sigma <= 1e-9 or sigma_next <= 1e-9 or eta <= 0.0:
        return sigma_next, 1.0, 0.0
    eta = min(eta, 1.0)
    sigma_down = sigma_next * (1.0 + (sigma_next / sigma - 1.0) * eta)
    alpha_next, alpha_down = 1.0 - sigma_next, 1.0 - sigma_down
    ratio = alpha_next / max(alpha_down, 1e-12)
    renoise = math.sqrt(max(sigma_next ** 2 - (sigma_down * ratio) ** 2, 0.0))
    return sigma_down, ratio, renoise


def _renoise_flow(x: torch.Tensor, sigma: float, sigma_up_to: float,
                  noise: torch.Tensor) -> torch.Tensor:
    """Move an FM latent from sigma UP to sigma_up_to (a restart jump).

    x = (1 - s) x0 + s eps  ->  x' = (1 - s') x0 + s' eps'. Scaling by
    (1 - s') / (1 - s) keeps the signal where it belongs; the added noise
    makes the total noise std s'. The VE jump used before 1.9.0 added
    sqrt(s'^2 - s^2) and left the signal unscaled.
    """
    ratio = (1.0 - sigma_up_to) / max(1.0 - sigma, 1e-12)
    add = math.sqrt(max(sigma_up_to ** 2 - (sigma * ratio) ** 2, 0.0))
    return x * ratio + noise * add


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

    def __call__(self, x: torch.Tensor, sigma_up: float, mask: torch.Tensor) -> torch.Tensor:
        if sigma_up <= 1e-9:
            return torch.zeros_like(x)
        noise = torch.randn_like(x, generator=self.get_generator(x.device))
        return noise * sigma_up * mask


def _resolution_aware_shift(latent_h: int, latent_w: int,
                            base_shift: float = 0.5, max_shift: float = 1.15,
                            base_seq: int = 256, max_seq: int = 4096) -> float:
    seq_len = max(1, (latent_h // 2) * (latent_w // 2))
    span = max(max_seq - base_seq, 1)
    frac = (seq_len - base_seq) / span
    frac = min(max(frac, 0.0), 1.0)
    mu = base_shift + (max_shift - base_shift) * frac
    return float(math.exp(mu))


def _insert_restarts(sigmas: torch.Tensor, repeats: int, k_steps: int,
                     t_min_frac: float, t_max_frac: float):
    if repeats <= 0 or k_steps <= 0:
        return sigmas, None

    s_list = [float(v) for v in sigmas.tolist()]
    s_max = max(s_list)
    lo = t_min_frac * s_max
    hi = t_max_frac * s_max
    if not (0.0 < lo < hi <= s_max):
        return sigmas, None


    entry = None
    for idx in range(1, len(s_list)):
        if s_list[idx] > 1e-7 and s_list[idx] <= lo:
            entry = idx
            break
    if entry is None:
        smallest = min(v for v in s_list if v > 1e-7)
        print(f"[DDRK] restart: no schedule point falls in the window. "
              f"restart_t_min={t_min_frac:.3f} means sigma <= {lo:.4f}, but the "
              f"smallest non-terminal sigma is {smallest:.4f}. Raise "
              f"restart_t_min above {smallest / s_max:.3f} (and keep "
              f"restart_t_max above that). Restart disabled for this run.")
        return sigmas, None

    out = s_list[:entry + 1]
    is_jump = [False] * len(out)
    anchor = s_list[entry]

    for _ in range(repeats):
        out.append(hi)
        is_jump.append(True)
        for k in range(1, k_steps + 1):

            frac = k / k_steps
            out.append(hi * ((anchor / hi) ** frac))
            is_jump.append(False)

    for idx in range(entry + 1, len(s_list)):
        out.append(s_list[idx])
        is_jump.append(False)

    return (torch.tensor(out, dtype=sigmas.dtype, device=sigmas.device),
            is_jump)


def _flow_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    if abs(shift - 1.0) < 1e-4:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


_KNOWN_SCHEDULERS = frozenset({
    "ddrk_auto", "ddrk_model", "ddrk_model_beta",
    "ddrk_flow_linear", "ddrk_cosine", "ddrk_beta",
    "ddrk_fewstep", "ddrk_flow_cosmos",
    "ddrk_edm_karras", "ddrk_edm_simple", "ddrk_edm_poly",
    "ddrk_anima",  # legacy alias kept for saved workflows; maps to ddrk_flow_linear
})


def _resolve_scheduler_name(scheduler_type: str, is_edm: bool) -> str:
    """What ddrk_auto actually means for this model family.

    Up to 1.7.0 ddrk_auto on Flow Matching meant ddrk_cosine (or a shift-1.0
    linear schedule under auto_optimize). Live telemetry from Elysium
    (Krea 2 Turbo, 10 steps, Sept 2026) showed what that does: cosine puts
    its densest steps at sigma ~1, so the first two steps covered 2.7% of the
    sigma range for 3 of 11 model calls, while the resolution-derived shift
    left the final one-shot jump at 0.37-0.40 on native-resolution latents.
    The model's own schedule (ComfyUI "simple" on its shifted sigma table)
    ends that jump at ~0.26. So on FM ddrk_auto now defers to the model.
    """
    if scheduler_type == "ddrk_auto" and not is_edm:
        return "ddrk_model"
    return scheduler_type


def _model_sigmas(model_sampling, steps: int, device,
                  comfy_name: str = "simple") -> Optional[torch.Tensor]:
    """A schedule on the model's own sigma table, via
    comfy.samplers.calculate_sigmas(ms, comfy_name): "simple" for ddrk_model,
    "beta" for ddrk_model_beta.

    Returns None (and says why) when it cannot be built, so the caller can
    fall back loudly instead of silently substituting another schedule.
    """
    label = "ddrk_model" if comfy_name == "simple" else f"ddrk_model_{comfy_name}"
    if model_sampling is None:
        print(f"[DDRK] {label}: no model_sampling was passed to "
              "get_ddrk_sigmas, so the model's own shift is unknown.")
        return None
    calc = getattr(comfy.samplers, "calculate_sigmas", None)
    if calc is None:
        print(f"[DDRK] {label}: comfy.samplers.calculate_sigmas is missing "
              "in this ComfyUI build.")
        return None
    try:
        sigmas = calc(model_sampling, comfy_name, steps)
    except Exception as e:
        print(f"[DDRK] {label}: ComfyUI's '{comfy_name}' scheduler failed "
              f"({type(e).__name__}: {e}).")
        return None
    sigmas = torch.as_tensor(sigmas).to(device=device, dtype=torch.float32)
    # simple_scheduler indexes a finite sigma table; on a table shorter than
    # the step count it repeats entries. A repeated sigma is a zero-length
    # step that still costs a model call, so reject it instead of sampling it.
    ok = (len(sigmas) == steps + 1 and float(sigmas[-1]) == 0.0
          and bool(((sigmas[:-1] - sigmas[1:]) > 0).all()))
    if not ok:
        print(f"[DDRK] {label}: the model's '{comfy_name}' schedule for {steps} "
              f"steps is not strictly decreasing to zero "
              f"({[round(float(v), 4) for v in sigmas.tolist()]}).")
        return None
    return sigmas


def get_ddrk_sigmas(scheduler_type: str, steps: int, sigma_min: float,
                    sigma_max: float, device: torch.device,
                    flow_shift: float = 3.0, warmup_steps: int = 0,
                    beta_a: float = 2.0, beta_b: float = 1.0,
                    auto_optimize: bool = True,
                    model_sampling=None) -> torch.Tensor:
    is_edm = sigma_max > 5.0

    # An unrecognised name used to fall through to a default schedule without a
    # word. Same hazard as the integrator fallback above: a typo in a sweep
    # script silently changes the schedule while the log still says otherwise.
    if scheduler_type not in _KNOWN_SCHEDULERS:
        raise ValueError(
            f"Unknown scheduler_type {scheduler_type!r}. "
            f"Expected one of: {', '.join(sorted(_KNOWN_SCHEDULERS))}."
        )

    if scheduler_type == "ddrk_anima":


        print("[DDRK] 'ddrk_anima' is not a real schedule and has been removed "
              "from the UI; using ddrk_flow_linear, which is what it always did.")
        scheduler_type = "ddrk_flow_linear"

    scheduler_type = _resolve_scheduler_name(scheduler_type, is_edm)
    if scheduler_type == "ddrk_auto":
        # Only EDM reaches here; FM ddrk_auto was resolved to ddrk_model above.
        scheduler_type = "ddrk_edm_karras"

    if scheduler_type == "ddrk_model_beta":
        # ComfyUI's beta scheduler (alpha = beta = 0.6) on the model's own
        # sigma table: the model's shift is kept, but the steps crowd towards
        # both ends. On the analytic bench (tests/bench_analytic.py) the
        # final jump to sigma 0 - the largest single error on Flow Matching -
        # shrank 1.6-3.8x at equal steps (last sigma 0.114 vs 0.25 at 10
        # steps), and the error to the exact solution fell 2-4x. Popular for Flux in
        # the community; not yet A/B-tested on images in this project.
        model_sched = _model_sigmas(model_sampling, steps, device, "beta")
        if model_sched is not None:
            return model_sched
        print("[DDRK] ddrk_model_beta unavailable; using ddrk_model instead.")
        scheduler_type = "ddrk_model"

    if scheduler_type == "ddrk_model":
        model_sched = _model_sigmas(model_sampling, steps, device)
        if model_sched is not None:
            return model_sched
        # Loud fallback, never a silent one: the log must say which schedule
        # was really used, or an ablation records ddrk_model while measuring
        # something else.
        fallback = "ddrk_edm_karras" if is_edm else "ddrk_flow_linear"
        print(f"[DDRK] ddrk_model unavailable; using {fallback}"
              + ("" if is_edm else f" with flow_shift={flow_shift:.3f}")
              + " instead.")
        scheduler_type = fallback

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
                # HC2 order 2, not Euler (before 1.9.0 this said "euler").
                # Both cost one model call per step, and HC2 o2 was measured
                # more accurate at every step count from 2 (4x) to 20 (41x) -
                # the reason Lite never offers Euler. The <=6-step rule below
                # leaves hc2 alone, so this is what actually runs.
                integrator = "hc2"
            debug_lines.append(
                f"[DDRK Auto] FM few-step ({total_steps} steps): "
                f"SABER=0, SDE=0, momentum=0, sharp={sharpness:.2f}, "
                # The old text also claimed "scheduler=linear, shift<=1.0";
                # this function has never touched the schedule.
                f"integrator={integrator}")
        elif total_steps <= 20:
            saber_fusion = min(saber_fusion, 0.05)
            sde_strength = 0.0
            momentum_beta = min(momentum_beta, 0.10)
            sharpness = min(sharpness, 0.15)
            debug_lines.append(
                f"[DDRK Auto] FM mid-step ({total_steps} steps): "
                f"SABER<=0.05, SDE=0, momentum<=0.10, sharp<=0.15")

    if total_steps <= 6 and integrator not in ("hc2", "hc3"):


        # This rule fires whatever auto_optimize says. Elysium's "best" on
        # SDXL announced "RK4, four calls per step" while a 4-6 step turbo run
        # was really Euler, so the message has to say so plainly, and the
        # telemetry header records both the requested and the used integrator.
        if integrator != "euler":
            debug_lines.append(
                f"[DDRK] {total_steps} steps (<=6): integrator '{integrator}' "
                f"REPLACED by 'euler' (this happens even with auto_optimize "
                f"off). Higher-order integrators need more steps than this to "
                f"pay for their extra model calls. Use >=7 steps to keep your "
                f"choice.")
        integrator = "euler"


    if is_edm and total_steps <= 10 and integrator == "rk4":
        debug_lines.append(
            f"[DDRK] EDM at {total_steps} steps (<=10): integrator 'rk4' "
            f"REPLACED by 'heun' (even with auto_optimize off).")
        integrator = "heun"

    for line in debug_lines:
        print(line)

    return integrator, sde_strength, sharpness, saber_fusion, momentum_beta


def _integrator_step(chosen_integrator: str, x: torch.Tensor, sigma_curr, sigma_target,
                     model_fn, state: SamplerState, momentum_beta: float,
                     limiter_kappa: float = 1.0, hc2_max_order: int = 2,
                     hc2_corrector: float = 0.0, hc2_free_corrector: bool = False):
    if chosen_integrator == "hc2":
        return hc2_step(x, sigma_curr, sigma_target, model_fn, state,
                        momentum_beta=momentum_beta, limiter_kappa=limiter_kappa,
                        max_order=hc2_max_order, corrector_thresh=hc2_corrector,
                        free_corrector=hc2_free_corrector)
    if chosen_integrator == "hc3":
        # HC3 = HC2's predictor with trust damping (_hc3_trust) plus the
        # zero-cost corrector: third order at one model call per step.
        return hc2_step(x, sigma_curr, sigma_target, model_fn, state,
                        momentum_beta=momentum_beta, limiter_kappa=limiter_kappa,
                        max_order=hc2_max_order, corrector_thresh=hc2_corrector,
                        free_corrector=True, damping=True)
    if chosen_integrator == "rk4":
        out = rk4_step(x, sigma_curr, sigma_target, model_fn, state, momentum_beta=momentum_beta)
    elif chosen_integrator == "heun":
        out = heun_step(x, sigma_curr, sigma_target, model_fn, state, momentum_beta=momentum_beta)
    elif chosen_integrator == "euler":
        out = euler_step(x, sigma_curr, sigma_target, model_fn, state, momentum_beta=momentum_beta)
    else:
        # An unrecognised name used to fall through to Euler in silence. Through the
        # node dropdowns that was unreachable, but a sweep script driving the sampler
        # directly could typo "hc-2" and unknowingly measure Euler while recording
        # HC2 in its results. A silently substituted algorithm is worse than a
        # stopped run, so this raises.
        raise ValueError(
            f"Unknown integrator {chosen_integrator!r}. "
            f"Expected one of: auto, euler, heun, rk4, hc2, hc3."
        )
    # Feed the denoiser value at this step's start into HC2's history. Only
    # auto mode mixes integrators, so a pinned euler/heun/rk4 run never reads
    # it and is unaffected.
    if state.last_D is not None:
        lam = _hc2_lambda(_safe_sigma(sigma_curr), bool(state.hc2_flow))
        state.hc2_D_prev2, state.hc2_lambda_prev2 = state.hc2_D_prev, state.hc2_lambda_prev
        state.hc2_D_prev, state.hc2_lambda_prev = state.last_D, lam
    state.hc2_pc = None
    return out


def _conditioning_fingerprint(extra_args) -> str:
    try:
        import hashlib
        h = hashlib.sha256()
        found = False

        def absorb(obj, depth=0):
            nonlocal found
            if depth > 4:
                return
            if isinstance(obj, torch.Tensor):
                if obj.numel() > 0 and obj.is_floating_point():
                    with torch.no_grad():
                        t = obj.detach().float().flatten()
                        if t.numel() > 4096:
                            idx = torch.linspace(0, t.numel() - 1, 4096,
                                                 device=t.device).long()
                            t = t[idx]
                        h.update(str(tuple(obj.shape)).encode())
                        h.update(t.cpu().numpy().tobytes())
                        found = True
            elif isinstance(obj, dict):
                for k in sorted(obj.keys(), key=str):
                    absorb(obj[k], depth + 1)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    absorb(v, depth + 1)

        for key in ("cond", "uncond", "positive", "negative"):
            if isinstance(extra_args, dict) and key in extra_args:
                absorb(extra_args[key])
        if not found and isinstance(extra_args, dict):
            absorb(extra_args)
        return h.hexdigest()[:16] if found else "unavailable"
    except Exception as e:
        return f"error:{type(e).__name__}"


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


def _dbg_delta(before: torch.Tensor, after: torch.Tensor):
    # Returning None on failure made "the metric blew up" indistinguishable from
    # "the metric was never taken" in the telemetry these ablations are read from.
    # A failed measurement now carries its own error marker.
    try:
        with torch.no_grad():
            return float((after - before).abs().mean().item())
    except Exception as e:
        return {"error": str(e)}


def _sampling_family(model, sigmas: torch.Tensor) -> Tuple[float, bool, float, str]:
    """(sigma_max, is_edm, noise_scale, source) for the model being sampled.

    Up to 1.9.0 the family was guessed from the schedule alone
    (sigmas.max() > 5). A schedule that starts low - img2img or a hires fix
    at denoise below ~0.6, the second pass, a SplitSigmas tail - put an
    SDXL or SD1.5 run through the Flow Matching path: the FM noise formulas
    (which scale by 1 - sigma, negative above sigma = 1), FM sharpening, no
    EDM clamp. ComfyUI's own samplers ask the model instead
    (isinstance(model_sampling, CONST)), and so does this now. The schedule
    heuristic remains the fallback for direct calls without a ComfyUI model.
    """
    ms = None
    try:
        ms = model.inner_model.inner_model.model_sampling
    except AttributeError:
        pass
    if ms is not None:
        try:
            import comfy.model_sampling as cms
            is_edm = not isinstance(ms, cms.CONST)
            sigma_max = float(ms.sigma_max)
            noise_scale = float(getattr(ms, "noise_scale", 1.0)) if not is_edm else 1.0
            if math.isfinite(sigma_max) and sigma_max > 0.0:
                return sigma_max, is_edm, noise_scale, "model_sampling"
        except Exception as e:
            print(f"[DDRK] model_sampling unreadable ({type(e).__name__}: {e}); "
                  f"guessing the family from the schedule.")
    sigma_max = float(sigmas.max())
    return sigma_max, sigma_max > 5.0, 1.0, "schedule"


_ADAPT_MIN_GAP = 0.5  # no adapted step shorter than half its reference step


def _adapt_remaining_sigmas(sigmas_work: torch.Tensor, sigmas_ref: torch.Tensor,
                            i: int, adapt_scale: float, activity: float,
                            mean_activity: float, sigma_adapt: float) -> float:
    """HC2 step placement after step i. Edits sigmas_work[i+2 : -1] in place.

    Rewritten in 1.8.0 after live Krea 2 telemetry (six runs, 10 steps):

    * The sign was inverted. f was mean/activity, so a step with ABOVE-average
      correction activity made the NEXT step LONGER. Seed 227007225: activity
      0.286 on step 2 doubled step 3 (0.939 -> 0.829 instead of 0.888); step 3
      then hit activity 0.67 with 36-43% of elements clipped by the limiter -
      positive feedback. Now f = activity/mean: high activity shortens the
      next step, which is what "equalise the error" means.
    * The last non-zero sigma was scaled like the rest. The scale saturated at
      1.1 in all six runs, raising it 0.258 -> 0.284 and 0.367 -> 0.403, so the
      final one-shot jump to zero - the coarsest step of the run - grew by
      10%. It is now never raised above its reference value.
    * Clamping against the current sigma produced near-empty steps: 0.737 ->
      0.721 (delta 0.017) cost a full model call. Every adapted step is now
      kept at least _ADAPT_MIN_GAP of its reference step. The bound is relative
      to the reference step, not the schedule mean, because a shifted FM
      schedule legitimately starts with steps far below its mean (shift 3.16
      at 10 steps: first step 0.034 against a mean of 0.1).
    """
    last = len(sigmas_ref) - 2           # index of the last non-zero sigma
    start = i + 2
    if start > last or mean_activity <= 1e-12 or activity <= 0.0:
        return adapt_scale
    ref = [float(v) for v in sigmas_ref.tolist()]
    # Restart schedules jump back up; this controller only understands a
    # monotone tail, so it leaves such a schedule untouched.
    if any(ref[k - 1] <= ref[k] for k in range(start, last + 2)):
        return adapt_scale

    lo_s, hi_s = 1.0 - sigma_adapt, 1.0 + sigma_adapt
    f = (activity / mean_activity) ** (1.0 / 3.0)
    f = min(max(f, lo_s), hi_s)
    adapt_scale = min(max(adapt_scale * f, lo_s), hi_s)

    prev = float(sigmas_work[start - 1])
    for k in range(start, last + 1):
        target = ref[k] * adapt_scale
        if k == last:
            target = min(target, ref[k])
        # Room for this step (>= half its reference gap) ...
        hi = prev - _ADAPT_MIN_GAP * (ref[k - 1] - ref[k])
        # ... and for every step after it, down to zero.
        lo = _ADAPT_MIN_GAP * ref[k]
        value = max(min(target, hi), lo)
        sigmas_work[k] = value
        prev = value
    return adapt_scale


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
    # Off by default since 1.8.0. In 4 of 6 live Krea 2 runs the final
    # latent came out exactly symmetric (max = -min = 1.4350, 1.5301, 1.7353,
    # 1.3914): the element-wise clip fired on the last step, whose output is
    # the image, flattening the brightest 0.5% of latent values - the prime
    # suspect for dotted halos around light sources.
    dyn_thresh_percentile = kwargs.get("dyn_thresh_percentile", 1.0)


    latent_rescale = kwargs.get("latent_rescale", kwargs.get("cfg_rescale", 0.0))
    if latent_rescale > 0:


        print("[DDRK] latent_rescale is deprecated and measured harmful "
              "(~30% dynamic range loss plus visible artifacts on Flow "
              "Matching). Set it to 0. HC2's limiter_kappa covers the same "
              "need without the cost.")
    limiter_kappa = kwargs.get("limiter_kappa", 1.0)
    hc2_max_order = int(kwargs.get("hc2_max_order", 2))
    hc2_corrector = float(kwargs.get("hc2_corrector", 0.0))
    hc2_free_corrector = bool(kwargs.get("hc2_free_corrector", False))
    restart_repeats = int(kwargs.get("restart_repeats", 0))
    restart_steps = int(kwargs.get("restart_steps", 3))
    restart_t_min = float(kwargs.get("restart_t_min", 0.10))
    restart_t_max = float(kwargs.get("restart_t_max", 0.35))
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

    sigma_max, is_edm, noise_scale, family_source = _sampling_family(model, sigmas)
    # Karras Alg. 2 divides s_churn by the number of steps; restarts added
    # below do not count, as they do not in k-diffusion.
    churn_steps = total_steps

    integrator_requested = integrator
    integrator, sde_strength, sharpness, saber_fusion, momentum_beta = _resolve_effective_params(
        total_steps, is_edm, auto_optimize, integrator, sde_strength, sharpness,
        saber_fusion, momentum_beta,
    )

    rec = _DDRKDebugRecorder(tag=debug_tag) if debug_mode else None
    if rec is not None:
        rec.meta.update({
            "is_edm": is_edm,
            "family_source": family_source,
            "noise_scale": noise_scale,
            "total_steps": total_steps,
            "sigma_max": sigma_max,
            "sigma_start": float(sigmas[0]),
            "integrator_param": integrator,
            "integrator_requested": integrator_requested,
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
            "hc2_free_corrector": hc2_free_corrector,
            "sigma_adapt": sigma_adapt,
            "restart_repeats": restart_repeats,
            "restart_steps": restart_steps,
            "restart_t_min": restart_t_min,
            "restart_t_max": restart_t_max,
            "latent_rescale": latent_rescale,
            "saber_mode": saber_mode,
            "use_ema_saber": use_ema_saber, "ema_decay": ema_decay,
            "auto_optimize": auto_optimize,
            "latent_shape": list(x.shape),
            "latent_dtype": str(work_dtype),
            "work_device": str(work_device),
            "sigma_schedule": [round(float(s), 5) for s in sigmas.tolist()],
            "cond_fingerprint": _conditioning_fingerprint(extra_args),
        })


        for label, key in (("cfg", "debug_cfg"), ("seed", "debug_seed"),
                           ("denoise", "debug_denoise"), ("scheduler_type", "debug_scheduler_type"),
                           ("flow_shift", "debug_flow_shift"), ("steps_requested", "debug_steps_requested")):
            val = kwargs.get(key, None)
            if val is not None:
                rec.meta[label] = val

    state = SamplerState(total_steps=total_steps, is_edm=is_edm)
    # "ve" (default): lambda = -log(sigma), HC2 as it has always been.
    # "flow": the Flow Matching half-log-SNR with the (1 - sigma) weight -
    # the exact parameterisation for FM, but on images it measured a draw
    # (Anima, CFG 4, 25 steps, distance to an RK4-24 reference: 0.106 vs
    # 0.108 on one seed, 0.165 vs 0.155 on the other), so it stays opt-in.
    # On EDM both spellings are identical.
    hc2_space = str(kwargs.get("hc2_space", "ve"))
    if hc2_space not in ("ve", "flow"):
        raise ValueError(f"Unknown hc2_space {hc2_space!r}. "
                         f"Expected one of: ve, flow.")
    state.hc2_flow = hc2_space == "flow" and not is_edm
    state.hc2_stats = rec is not None or sigma_adapt > 0.0
    if rec is not None:
        rec.meta["hc2_space"] = ("flow" if state.hc2_flow else "ve")
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


    restart_flags = None
    if restart_repeats > 0:
        sigmas_rs, restart_flags = _insert_restarts(
            sigmas, restart_repeats, restart_steps, restart_t_min, restart_t_max)
        if restart_flags is not None:
            extra = restart_repeats * restart_steps
            if sde_strength > 0:
                print("[DDRK] restart is a deterministic-sampling technique; "
                      "with sde_strength > 0 you are paying for error "
                      "contraction twice.")
            print(f"[DDRK] restart: {restart_repeats} x {restart_steps} steps "
                  f"in sigma [{restart_t_min:.2f}, {restart_t_max:.2f}] of "
                  f"sigma_max -> +{extra} model calls")
            sigmas = sigmas_rs
            total_steps = len(sigmas) - 1
            router = AdaptivePhaseRouter(total_steps, sigma_max, is_edm)
            if sigma_adapt > 0.0:
                print("[DDRK] sigma_adapt stays inactive until the last "
                      "restart jump has passed: the step controller only "
                      "handles a monotone tail.")
            if rec is not None:


                rec.meta["total_steps"] = total_steps
                rec.meta["sigma_schedule"] = [round(float(v), 5)
                                              for v in sigmas.tolist()]
                rec.meta["restart_applied"] = True
                rec.meta["restart_extra_calls"] = restart_repeats * restart_steps
        else:
            print("[DDRK] restart: no valid entry point in the requested "
                  "sigma window; disabled for this run.")

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

        if restart_flags is not None and i + 1 < len(restart_flags)\
                and restart_flags[i + 1]:


            sc, sn = float(sigma_curr), float(sigma_next)
            var = max(sn * sn - sc * sc, 0.0)
            if var > 0.0:
                gen = sde.get_generator(x.device) if sde is not None else None
                noise = (torch.randn_like(x, generator=gen) if gen is not None
                         else torch.randn_like(x))
                if is_edm:
                    x = x + noise * (var ** 0.5)
                else:
                    x = _renoise_flow(x, sc, sn, noise * noise_scale)


            state.hc2_D_prev = None
            state.hc2_lambda_prev = None
            state.hc2_D_prev2 = None
            state.hc2_lambda_prev2 = None
            state.hc2_pc = None
            state.d_prev = None
            state.prev_denoised = None
            if rec is not None:
                rec.log_step(step=i, phase=router.get_phase(i),
                             restart_jump=True,
                             sigma_curr=round(sc, 5), sigma_next=round(sn, 5),
                             model_calls=0)
            progress_bar.update(0)
            continue

        if rec is not None:
            rec.start_step()
            step_rec: Dict[str, Any] = {"step": i, "phase": None}
            debug_counters["model_calls"] = 0
            debug_counters["dt_fired"] = 0
            sigma_curr_pre_churn = float(sigma_curr)

        if is_edm and s_churn > 0 and s_tmin <= float(sigma_curr) <= s_tmax:
            # Karras et al. 2022, Alg. 2: gamma = min(S_churn / N, sqrt(2) - 1)
            # with N the number of steps, as in k-diffusion. Up to 1.9.0 this
            # divided by sigma instead, so gamma sat at its sqrt(2) - 1 cap on
            # every step below sigma = S_churn / 0.414 (most of an SDXL run at
            # any useful setting) and the dial barely changed anything.
            gamma = min(s_churn / churn_steps, math.sqrt(2) - 1)
            sigma_hat = float(sigma_curr) * (1.0 + gamma)


            noise = torch.randn_like(x, generator=sde.get_generator(x.device)) * s_noise
            x_before_churn = x if rec is not None else None
            x = x + noise * math.sqrt(sigma_hat ** 2 - float(sigma_curr) ** 2)
            sigma_curr = torch.tensor(sigma_hat, device=work_device, dtype=torch.float32)


            state.hc2_D_prev = None
            state.hc2_lambda_prev = None
            state.hc2_D_prev2 = None
            state.hc2_lambda_prev2 = None
            state.hc2_pc = None
            state.d_prev = None
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
            flow_ratio = 1.0
            if sde_active:
                # sde_active already implies Flow Matching (not is_edm), so
                # this is always the FM recipe since 1.9.0 - see
                # _ancestral_split_flow for what the VE one got wrong.
                sigma_down, flow_ratio, sigma_up = _ancestral_split_flow(
                    float(sigma_curr), float(sigma_next), eta=sde_strength)
                sigma_down_t = torch.tensor(sigma_down, device=work_device, dtype=torch.float32)
            else:
                sigma_down_t = sigma_next

            x_next, preview_denoised, _ = _integrator_step(
                chosen_integrator, x, sigma_curr, sigma_down_t, model_fn, state, momentum_beta,
                limiter_kappa=limiter_kappa, hc2_max_order=hc2_max_order,
                hc2_corrector=hc2_corrector, hc2_free_corrector=hc2_free_corrector)

            if sde_active and sigma_up > 1e-9:
                edge_mask = log_mask(x_next)
                entropy = local_entropy_mask(x_next, window=5)
                flat_mask = ((1.0 - edge_mask) * entropy).clamp(0.0, 1.0)
                x_before_sde = x_next if rec is not None else None
                # The mask gates only the added noise. Where it is 0 the
                # latent is still rescaled, so a masked-out region lands at
                # std below sigma_next - the same trade-off the VE version
                # made, where masked regions were simply under-noised.
                x_next = x_next * flow_ratio + sde(x_next, sigma_up * noise_scale, flat_mask)
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
                hc2_corrector=hc2_corrector, hc2_free_corrector=hc2_free_corrector)
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
                hc2_corrector=hc2_corrector, hc2_free_corrector=hc2_free_corrector)
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

        if (sigma_adapt > 0.0 and chosen_integrator in ("hc2", "hc3")
                and state.hc2_activity > 0.0 and i + 2 < len(sigmas_work) - 1):
            activity_sum += state.hc2_activity
            activity_n += 1
            mean_act = activity_sum / activity_n
            adapt_scale = _adapt_remaining_sigmas(
                sigmas_work, sigmas_ref, i, adapt_scale,
                state.hc2_activity, mean_act, sigma_adapt)
            if rec is not None:
                step_rec["sigma_adapt_scale"] = round(float(adapt_scale), 5)

        if rec is not None:
            step_rec["curvature"] = state.curvature if integrator == "auto" and phase == 1 else None
            # HC2's fields persist between its steps; in auto mode only log
            # them on steps HC2 actually took.
            if chosen_integrator in ("hc2", "hc3"):
                if state.hc3_theta >= 0.0:
                    step_rec["hc3_trust"] = round(state.hc3_theta, 5)
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
                if state.hc2_free_corr >= 0.0:
                    step_rec["hc2_free_corr_delta"] = round(state.hc2_free_corr, 6)
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


    # Wide guard against real blowups, not a tone control. It used to be a
    # fixed +-7 (EDM) / +-20 (FM), which is right for a finished image but
    # clips a latent that is handed on still noisy - a SplitSigmas head, or
    # the first KSampler of a two-stage workflow - so the bound now grows
    # with the noise the output legitimately carries. Ending at sigma = 0 it
    # is exactly the old bound.
    sigma_end = max(float(sigmas_work[-1]), 0.0)
    if is_edm:
        final_bound = 7.0 + 5.0 * sigma_end
    else:
        final_bound = 20.0 + 5.0 * noise_scale * sigma_end
    x = torch.clamp(x, -final_bound, final_bound)

    # torch.clamp leaves NaN untouched, so a run that blew up used to finish
    # quietly and look like any other result. In debug_mode - the measurement
    # regime - that would silently poison an ablation average, so it raises.
    # In ordinary generation a partially usable image beats a lost run, so it
    # only warns loudly.
    if not torch.isfinite(x).all():
        n_nan = int(torch.isnan(x).sum().item())
        n_inf = int(torch.isinf(x).sum().item())
        msg = (f"non-finite values in the sampler output: "
               f"{n_nan} NaN, {n_inf} Inf out of {x.numel()} elements")
        if rec is not None:
            rec.meta["nonfinite_output"] = {"nan": n_nan, "inf": n_inf}
            rec.save()
            raise RuntimeError(
                f"[DDRK] {msg}. Raised because debug_mode is on and this result "
                f"would otherwise be recorded as a valid measurement."
            )
        print(f"[DDRK] WARNING: {msg}. The image may be partly corrupt. "
              f"Enable debug_mode to fail fast on this instead.")

    if rec is not None:
        rec.save()

    return x


class DDRKOmegaSchedulerNode:
    DESCRIPTION = ('DDRK sigma schedules as a SIGMAS output, for SamplerCustom / SamplerCustomAdvanced. ddrk_auto picks the right one for the model family.')

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Auto-detects FM vs EDM by sigma_max"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "scheduler_type": ([
                    "ddrk_auto",
                    "ddrk_model",
                    "ddrk_model_beta",
                    "ddrk_cosine",
                    "ddrk_beta",
                    "ddrk_flow_linear",
                    "ddrk_flow_cosmos",
                    "ddrk_fewstep",
                    "ddrk_edm_karras",
                    "ddrk_edm_poly",
                    "ddrk_edm_simple",
                ], {"default": "ddrk_auto",
                    "tooltip": "ddrk_auto: EDM -> ddrk_edm_karras; Flow Matching -> ddrk_model (since 1.8.0; was ddrk_cosine). ddrk_model = the model's own 'simple' schedule, so the shift comes from the model (and any ModelSampling* node in the graph); flow_shift is ignored. ddrk_model_beta = ComfyUI's 'beta' on the same table: same shift, steps denser at both ends, a much smaller final jump (analytic bench: 2-4x less error on FM; not yet image-tested)."}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Used by the ddrk_flow_*/cosine/beta/cosmos schedules only. ddrk_model and FM ddrk_auto take the shift from the model."}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 5}),
                "beta_a": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Shape parameter A, used only by ddrk_beta. Higher A front-loads larger steps at high sigma and leaves finer steps near sigma 0. Ignored by every other scheduler."}),
                "beta_b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Shape parameter B, used only by ddrk_beta. Raising B above ~2 shifts resolution toward high sigma and leaves a large final step, which is usually undesirable. A=2, B=1 gives Karras-like monotonically shrinking steps."}),
                "auto_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "No effect on the schedule since 1.8.0 (it used to force a shift-1.0 linear / cosine FM schedule under ddrk_auto). Kept so saved workflows load."
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/DDRK Omega/custom sampling"

    def get_sigmas(self, model, steps, scheduler_type, flow_shift, warmup_steps,
                   auto_optimize, beta_a=2.0, beta_b=1.0):
        ms = model.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        device = ms.sigma_min.device
        sigmas = get_ddrk_sigmas(
            scheduler_type, steps, sigma_min, sigma_max,
            device=device, flow_shift=flow_shift, warmup_steps=warmup_steps,
            beta_a=beta_a, beta_b=beta_b, auto_optimize=auto_optimize,
            model_sampling=ms,
        )
        return (sigmas,)


class DDRKOmegaSamplerNode:
    DESCRIPTION = ('The DDRK sampler as a SAMPLER output, for SamplerCustom / SamplerCustomAdvanced. For a one-node setup use DDRK Omega Auto or Lite instead.')

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "integrator": (["auto", "hc3", "hc2", "rk4", "heun", "euler"], {"default": "hc3",
                    "tooltip": "hc3 (recommended): HC2 plus trust damping and a zero-cost corrector - third order at one model call per step, and it falls back towards first order by itself where high CFG makes extrapolation unsafe. hc2: second order, one call per step (the 1.6-1.10 default for FM). euler: first order, one call. heun: 2 calls per step. rk4: 4 calls per step. auto: per-phase choice (heun/rk4 early, hc2 late; 1-4 calls per step). Compare integrators at equal MODEL CALLS, not equal steps."}),
                "sde_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Ancestral SDE noise, gated to flat regions. 0 = fully deterministic. FM only - ignored on EDM, which uses s_churn. MEASURED (Krea2 Turbo, 8 steps, 3 seeds): 0.00, 0.08 and 0.15 were indistinguishable by dynamic range (within 0.2%); 0.30 widened it 3.6%. The operator saw no quality change at any setting, only a different composition - which is what stochastic sampling does, it moves to a different sample rather than a better one. Treat this as a variation dial, not a quality dial."
                }),
                "sharpness": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.5, "step": 0.01,
                    "tooltip": "Final-step perceptual sharpen. FM only - disabled on EDM by design, so it does nothing there. MEASURED (Krea2 Turbo, 8 steps, 3 seeds): 0.10 widened dynamic range 2.6%, 0.12 widened it 3.2%, both consistent across seeds, and the operator saw a clear sharpness increase. The effect had NOT saturated at the point where an inherited cap used to cut it off, so values above 0.12 are now reachable and genuinely untested - halos and edge ringing are the failure mode to watch for."
                }),
                "saber_fusion": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Spatial stabilization (a gated blur). 0 disables the module entirely. MEASURED (Krea2 Turbo, 8 steps, 3 seeds): this consistently NARROWS dynamic range - 0.10 by 0.4%, 0.15 by 0.8% - and the operator reported smoother shadows and a silkier look, but graininess appearing by 0.15. Every other measurement in this project has associated a narrower range with a softer, less detailed result, so treat this as a stylistic smoothing control with a real cost, not as a quality improvement. Turn it off for text, graphics and pixel art."
                }),
                "saber_mode": (["auto", "image", "video"], {
                    "default": "auto",
                    "tooltip": "Auto detects video by 5D latent with >1 frame."
                }),
                "use_ema_saber": ("BOOLEAN", {"default": True}),
                "ema_decay": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01}),
                "dyn_thresh_percentile": ("FLOAT", {
                    "default": 1.0, "min": 0.9, "max": 1.0, "step": 0.001,
                    "tooltip": "Percentile latent limiter. 1.0 = disabled (default since 1.8.0). Clips individual latent values above the percentile. MEASURED (Krea 2 Turbo, live runs): at 0.995 it fired on the final FM step - whose output is the image - in 4 of 6 runs, leaving the final latent exactly symmetric (max = -min), i.e. the brightest 0.5% flattened. Suspect for dotted halos around lights. On EDM it fires below 30% of sigma_max, i.e. on most steps."
                }),
                "latent_rescale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "DEPRECATED - leave at 0. Attenuates latent values beyond ~2 std from the per-image mean. Ablation measured ~30% dynamic range loss and visible artifacts: the transfer curve has a kink at the 2-std threshold, applied per element. HC2's limiter_kappa handles overshoot without this cost. Also note this was never classical CFG-rescale; ComfyUI combines cond and uncond before the sampler runs, so that method cannot be implemented here."
                }),
                "restart_repeats": ("INT", {
                    "default": 0, "min": 0, "max": 4,
                    "tooltip": "Restart Sampling (Xu et al. 2023). 0 = off. Deterministic solvers accumulate discretisation error; jumping back up in noise a few times contracts it the way an SDE does, without paying on every step. Each repeat costs restart_steps extra model calls. Set sde_strength to 0 when using this. MEASURED: fires correctly and does what it should on EDM (Illustrious, 20 steps: +0.8% dynamic range for 15% more calls, so not a clear win by that metric). On Flow Matching the first attempt never fired at all - see restart_t_min, which must be set relative to the schedule's actual sigma range."
                }),
                "restart_steps": ("INT", {
                    "default": 3, "min": 1, "max": 8,
                    "tooltip": "Steps taken to descend back after each restart jump. Total extra model calls = restart_repeats x restart_steps."
                }),
                "restart_t_min": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 0.9, "step": 0.01,
                    "tooltip": "Lower edge of the restart window, as a FRACTION OF SIGMA_MAX - and that is the catch, because the two model families live on different sigma scales. On EDM sigma_max is ~14.6, so 0.10 means sigma 1.46 and the schedule passes through it. On Flow Matching sigma_max is 1.0, so 0.10 means sigma 0.10 - while an 8-step schedule's smallest non-terminal sigma is around 0.33, and restart silently never fired. For FM at few steps use 0.40 or higher. If no schedule point falls in the window the console now says so and prints the value that would work."
                }),
                "restart_t_max": ("FLOAT", {
                    "default": 0.35, "min": 0.02, "max": 0.95, "step": 0.01,
                    "tooltip": "Upper edge: how far back up the jump goes, as a fraction of sigma_max. Must exceed restart_t_min. Larger jumps contract more error but discard more of the structure already resolved."
                }),
                "hc2_corrector": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "HC2 selective corrector. 0 = off. On steps where the high-order correction exceeds this fraction of the first-order step, HC2 spends a SECOND model call to redo the step with an interpolated slope instead of an extrapolated one. MEASURED (Krea2 Turbo, 8 steps, CFG 1, 3 seeds): correction activity peaks around 0.14, so thresholds of 0.30 and 0.50 NEVER FIRE - those settings are identical to 0. At 0.10 it fired on 3 of 8 steps for 12 calls instead of 9; dynamic range fell 4.2% while the operator judged detail and lighting improved, so the metric and the eye disagree here. Useful range is roughly 0.05-0.15. EXPERIMENTAL."
                }),
                "sigma_adapt": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "HC2 adaptive step placement. 0 = off. Moves the intermediate sigma values by up to this fraction to equalise estimated error across steps; step COUNT, start and terminal zero are unchanged. MEASURED (Krea2 Turbo, 8 steps, CFG 1, 3 seeds): 0.10 gave the largest single gain of any option tested, +3.3% dynamic range, reaching +11% on one seed. 0.30 and 0.50 went slightly negative, matching the operator's report that higher values start to hurt on some seeds. Start at 0.10. Note the controller saturates at its own bound when activity falls monotonically, which is most of a run - so above ~0.20 this behaves less like adaptation and more like a fixed schedule stretch."
                }),
                "hc2_max_order": ("INT", {
                    "default": 2, "min": 1, "max": 3,
                    "tooltip": "HC2 maximum order. 2 = second order, one model call per step (default). 3 = allow third order: uses two past denoiser evaluations, spends ONE extra model call on the first step to bootstrap (a multistep method's first step is otherwise first-order and caps the whole run's accuracy), and falls back to second order on any step where the expansion stops converging. Ignored by every other integrator."
                }),
                "limiter_kappa": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "HC2 slope limiter. Caps the 2nd-order correction at this multiple of the 1st-order step, per element. MEASURED (Anima, CFG 5, 3 seeds): the control responds monotonically - kappa 3.0 clips 2.1% of elements on average, 1.0 clips 6.1%, 0.5 clips 11.8%. By dynamic range the default 1.0 came out best, 0.5 worst (-1.6%). Note the original premise was only partly borne out: going from CFG 1 to CFG 5 raised correction activity by ~57%, not by the order of magnitude that would make aggressive limiting necessary. Leave at 1.0 unless you see overshoot. Ignored by every other integrator."
                }),
                "momentum_beta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.8, "step": 0.05,
                    "tooltip": "Adams-Bashforth 2 slope extrapolation for EULER steps only (1.0 = full AB2, a second-order method at one call per step). 0 = plain Euler. Ignored by heun, rk4 and hc2: on top of a higher-order method it makes it first order (measured: RK4 error ~250x, Heun 3-5x at 128-256 calls), so since 1.10.0 it is no longer applied there."
                }),
                "sde_seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": "SDE noise seed. -1 derives the seed from ComfyUI's globally seeded Torch RNG, so the workflow seed remains reproducible."
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
                    "tooltip": "Flow Matching only: caps or disables SABER/SDE/momentum/sharpness by step budget, and at <=10 steps turns integrator 'auto' into 'hc2'. Does NOT set steps/cfg or the schedule."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Write a per-step diagnostics log (JSON+CSV+summary) to your ComfyUI output folder. Off by default; adds overhead only when on."
                }),
                "debug_tag": ("STRING", {
                    "default": "",
                    "tooltip": "Optional label included in the debug log filename, e.g. 'test1'."
                }),
            },
            "optional": {
                "hc2_space": (["ve", "flow"], {
                    "default": "ve",
                    "tooltip": "HC2 time variable on Flow Matching. ve (default): lambda = -log(sigma), as HC2 always was. flow: the exact FM parameterisation, half-log-SNR log((1-sigma)/sigma) with a (1-sigma) weight on the correction. MEASURED (Anima, CFG 4, 25 steps, 2 seeds, distance to an RK4 reference): a draw - better on one seed, worse on the other - so it is opt-in. No effect on EDM, where both are the same."
                }),
                "hc2_free_corrector": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "HC2 zero-cost corrector (UniPC-style). Off by default. After each HC2 step the model is called at the step's end point anyway, for the next step; with this on, that output is also used to redo the finished step by interpolation instead of extrapolation, then the next step proceeds from the corrected latent. No extra model calls. MEASURED on the analytic bench only (tests/analytic.py, equal calls): EDM Karras 20-32 calls, error 3-3.6x lower than plain HC2; Flow Matching with the model's schedule, a few percent, because the final jump to sigma 0 dominates there. NOT validated on images - A/B it before relying on it. Ignored by other integrators and on steps that follow SDE noise, churn or a restart jump."
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/DDRK Omega/custom sampling"

    def get_sampler(self, integrator, sde_strength, sharpness, saber_fusion,
                    saber_mode, use_ema_saber, ema_decay, dyn_thresh_percentile,
                    latent_rescale, limiter_kappa, restart_repeats, restart_steps, restart_t_min, restart_t_max, hc2_corrector, sigma_adapt, hc2_max_order, momentum_beta, sde_seed, s_churn, s_tmin, s_tmax,
                    s_noise, content_aware, auto_optimize, debug_mode=False, debug_tag="",
                    hc2_space="ve", hc2_free_corrector=False):
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
            "hc2_free_corrector": hc2_free_corrector,
            "sigma_adapt": sigma_adapt,
            "restart_repeats": restart_repeats,
            "restart_steps": restart_steps,
            "restart_t_min": restart_t_min,
            "restart_t_max": restart_t_max,
            "latent_rescale": latent_rescale,
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
            "hc2_space": hc2_space,
        }
        sampler = comfy.samplers.KSAMPLER(sample_ddrk_omega, extra_options=extra)
        return (sampler,)


class DDRKOmegaSmartConfigNode:
    DESCRIPTION = ('Shows what DDRK detects about a model (family, guidance embedding) and the settings it would recommend. Does not sample.')

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
    CATEGORY = "sampling/DDRK Omega/utils"

    def detect(self, model, latent_image):
        profile = _detect_model_profile(model, latent_image.get("samples"))
        print(f"[DDRK SmartConfig] Detected family: {profile['family'].upper()}")
        print(f"[DDRK SmartConfig] Hint: {profile['hint']}")
        return (profile["family"], profile["hint"], profile["guidance_embed"], profile["scheduler_type"],
                profile["flow_shift"], profile["integrator"],
                profile["sde_strength"], profile["sharpness"],
                profile["saber_fusion"], profile["momentum_beta"])


class DDRKOmegaUnifiedKSamplerNode:
    DESCRIPTION = ('The full DDRK sampler with every control: schedules, integrators (HC3/HC2/RK4/Heun/Euler), SDE, churn, restarts, enhancers, second pass and telemetry. For everyday use start with DDRK Omega Auto.')

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
                    "ddrk_auto", "ddrk_model", "ddrk_model_beta", "ddrk_cosine", "ddrk_beta",
                    "ddrk_flow_linear", "ddrk_flow_cosmos", "ddrk_fewstep",
                    "ddrk_edm_karras", "ddrk_edm_poly", "ddrk_edm_simple",
                ], {"default": "ddrk_auto",
                    "tooltip": "ddrk_auto: EDM -> ddrk_edm_karras; Flow Matching -> ddrk_model (since 1.8.0; was ddrk_cosine). ddrk_model = the model's own 'simple' schedule, so the shift comes from the model (and any ModelSampling* node in the graph); flow_shift and auto_flow_shift are ignored. ddrk_model_beta = ComfyUI's 'beta' on the same table: same shift, steps denser at both ends, a much smaller final jump (analytic bench: 2-4x less error on FM; not yet image-tested)."}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Used by the ddrk_flow_*/cosine/beta/cosmos schedules only. ddrk_model and FM ddrk_auto take the shift from the model."}),
                "auto_flow_shift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Derive flow_shift from the latent's token count instead of using the widget value, the way Flux/SD3 intend (shift should grow with image area). At 512x768 the derived value is ~2.05, at 1024x1024 ~3.16 - so a single fixed number cannot be right at both. FM only; ignored on EDM, and ignored by ddrk_model / FM ddrk_auto, which use the model's own shift. WARNING: if ModelSamplingFlux is already in your graph it applies the same shift at the model level and this would double it. Use one or the other. UNVALIDATED - no ablation yet."
                }),
                "integrator": (["auto", "hc3", "hc2", "rk4", "heun", "euler"], {"default": "hc3",
                    "tooltip": "hc3 (recommended): HC2 plus trust damping and a zero-cost corrector - third order at one model call per step, and it falls back towards first order by itself where high CFG makes extrapolation unsafe. hc2: second order, one call per step (the 1.6-1.10 default for FM). euler: first order, one call. heun: 2 calls per step. rk4: 4 calls per step. auto: per-phase choice (heun/rk4 early, hc2 late; 1-4 calls per step). Compare integrators at equal MODEL CALLS, not equal steps."}),
                "sde_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Ancestral SDE noise, gated to flat regions. 0 = fully deterministic. FM only - ignored on EDM, which uses s_churn. MEASURED (Krea2 Turbo, 8 steps, 3 seeds): 0.00, 0.08 and 0.15 were indistinguishable by dynamic range (within 0.2%); 0.30 widened it 3.6%. The operator saw no quality change at any setting, only a different composition - which is what stochastic sampling does, it moves to a different sample rather than a better one. Treat this as a variation dial, not a quality dial."}),
                "sharpness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5, "step": 0.01,
                    "tooltip": "Final-step perceptual sharpen. FM only - disabled on EDM by design, so it does nothing there. MEASURED (Krea2 Turbo, 8 steps, 3 seeds): 0.10 widened dynamic range 2.6%, 0.12 widened it 3.2%, both consistent across seeds, and the operator saw a clear sharpness increase. The effect had NOT saturated at the point where an inherited cap used to cut it off, so values above 0.12 are now reachable and genuinely untested - halos and edge ringing are the failure mode to watch for."}),
                "saber_fusion": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Spatial stabilization (a gated blur). 0 disables the module entirely. MEASURED (Krea2 Turbo, 8 steps, 3 seeds): this consistently NARROWS dynamic range - 0.10 by 0.4%, 0.15 by 0.8% - and the operator reported smoother shadows and a silkier look, but graininess appearing by 0.15. Every other measurement in this project has associated a narrower range with a softer, less detailed result, so treat this as a stylistic smoothing control with a real cost, not as a quality improvement. Turn it off for text, graphics and pixel art."
                }),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 5}),
                "beta_a": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Shape parameter A, used only by ddrk_beta. Higher A front-loads larger steps at high sigma and leaves finer steps near sigma 0. Ignored by every other scheduler."}),
                "beta_b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Shape parameter B, used only by ddrk_beta. Raising B above ~2 shifts resolution toward high sigma and leaves a large final step, which is usually undesirable. A=2, B=1 gives Karras-like monotonically shrinking steps."}),
                "auto_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Flow Matching only: caps or disables SABER/SDE/momentum/sharpness by step budget, and at <=10 steps turns integrator 'auto' into 'hc2'. Does NOT set steps/cfg or the schedule."
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
                "dyn_thresh_percentile": ("FLOAT", {"default": 1.0, "min": 0.9, "max": 1.0, "step": 0.001,
                    "tooltip": "Percentile latent limiter. 1.0 = off (default since 1.8.0): at 0.995 it clipped the brightest 0.5% of the final FM latent in 4 of 6 live Krea 2 runs."}),
                "latent_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "DEPRECATED - leave at 0. Measured ~30% dynamic range loss and visible artifacts. Use HC2 limiter_kappa instead."}),
                "restart_repeats": ("INT", {
                    "default": 0, "min": 0, "max": 4,
                    "tooltip": "Restart Sampling (Xu et al. 2023). 0 = off. Deterministic solvers accumulate discretisation error; jumping back up in noise a few times contracts it the way an SDE does, without paying on every step. Each repeat costs restart_steps extra model calls. Set sde_strength to 0 when using this. MEASURED: fires correctly and does what it should on EDM (Illustrious, 20 steps: +0.8% dynamic range for 15% more calls, so not a clear win by that metric). On Flow Matching the first attempt never fired at all - see restart_t_min, which must be set relative to the schedule's actual sigma range."
                }),
                "restart_steps": ("INT", {
                    "default": 3, "min": 1, "max": 8,
                    "tooltip": "Steps taken to descend back after each restart jump. Total extra model calls = restart_repeats x restart_steps."
                }),
                "restart_t_min": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 0.9, "step": 0.01,
                    "tooltip": "Lower edge of the restart window, as a FRACTION OF SIGMA_MAX - and that is the catch, because the two model families live on different sigma scales. On EDM sigma_max is ~14.6, so 0.10 means sigma 1.46 and the schedule passes through it. On Flow Matching sigma_max is 1.0, so 0.10 means sigma 0.10 - while an 8-step schedule's smallest non-terminal sigma is around 0.33, and restart silently never fired. For FM at few steps use 0.40 or higher. If no schedule point falls in the window the console now says so and prints the value that would work."
                }),
                "restart_t_max": ("FLOAT", {
                    "default": 0.35, "min": 0.02, "max": 0.95, "step": 0.01,
                    "tooltip": "Upper edge: how far back up the jump goes, as a fraction of sigma_max. Must exceed restart_t_min. Larger jumps contract more error but discard more of the structure already resolved."
                }),
                "hc2_corrector": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "HC2 selective corrector. 0 = off. On steps where the high-order correction exceeds this fraction of the first-order step, HC2 spends a SECOND model call to redo the step with an interpolated slope instead of an extrapolated one. MEASURED (Krea2 Turbo, 8 steps, CFG 1, 3 seeds): correction activity peaks around 0.14, so thresholds of 0.30 and 0.50 NEVER FIRE - those settings are identical to 0. At 0.10 it fired on 3 of 8 steps for 12 calls instead of 9; dynamic range fell 4.2% while the operator judged detail and lighting improved, so the metric and the eye disagree here. Useful range is roughly 0.05-0.15. EXPERIMENTAL."
                }),
                "sigma_adapt": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "HC2 adaptive step placement. 0 = off. Moves the intermediate sigma values by up to this fraction to equalise estimated error across steps; step COUNT, start and terminal zero are unchanged. MEASURED (Krea2 Turbo, 8 steps, CFG 1, 3 seeds): 0.10 gave the largest single gain of any option tested, +3.3% dynamic range, reaching +11% on one seed. 0.30 and 0.50 went slightly negative, matching the operator's report that higher values start to hurt on some seeds. Start at 0.10. Note the controller saturates at its own bound when activity falls monotonically, which is most of a run - so above ~0.20 this behaves less like adaptation and more like a fixed schedule stretch."
                }),
                "hc2_max_order": ("INT", {
                    "default": 2, "min": 1, "max": 3,
                    "tooltip": "HC2 maximum order. 2 = second order, one model call per step (default). 3 = allow third order: uses two past denoiser evaluations, spends ONE extra model call on the first step to bootstrap (a multistep method's first step is otherwise first-order and caps the whole run's accuracy), and falls back to second order on any step where the expansion stops converging. Ignored by every other integrator."
                }),
                "limiter_kappa": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "HC2 slope limiter. Caps the 2nd-order correction at this multiple of the 1st-order step, per element. MEASURED (Anima, CFG 5, 3 seeds): the control responds monotonically - kappa 3.0 clips 2.1% of elements on average, 1.0 clips 6.1%, 0.5 clips 11.8%. By dynamic range the default 1.0 came out best, 0.5 worst (-1.6%). Note the original premise was only partly borne out: going from CFG 1 to CFG 5 raised correction activity by ~57%, not by the order of magnitude that would make aggressive limiting necessary. Leave at 1.0 unless you see overshoot. Ignored by every other integrator."
                }),
                "momentum_beta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.8, "step": 0.05,
                    "tooltip": "Adams-Bashforth 2 slope extrapolation for EULER steps only. 0 = plain Euler. Ignored by heun, rk4 and hc2 since 1.10.0: on a higher-order method it makes it first order (measured: RK4 error ~250x, Heun 3-5x)."}),
                "sde_seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": "SDE noise seed. -1 derives the seed from ComfyUI's globally seeded Torch RNG, so the workflow seed remains reproducible."
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
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Write a per-step diagnostics log (JSON+CSV+summary) to your ComfyUI output folder. Off by default; adds overhead only when on."
                }),
                "debug_tag": ("STRING", {
                    "default": "",
                    "tooltip": "Optional label included in the debug log filename, e.g. 'test1'."
                }),
                "hc2_space": (["ve", "flow"], {
                    "default": "ve",
                    "tooltip": "HC2 time variable on Flow Matching. ve (default): lambda = -log(sigma), as HC2 always was. flow: the exact FM parameterisation, half-log-SNR log((1-sigma)/sigma) with a (1-sigma) weight on the correction. MEASURED (Anima, CFG 4, 25 steps, 2 seeds, distance to an RK4 reference): a draw - better on one seed, worse on the other - so it is opt-in. No effect on EDM, where both are the same."
                }),
                "refine_scale": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Second pass (hires fix in latent space). 1.0 = off. Above 1.0 the finished latent is upscaled by this factor and re-sampled from refine_denoise with the same settings: more pixels for eyes, hands and texture than the model gets at its base size. Costs refine_steps extra calls at the larger size."
                }),
                "refine_denoise": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": "How much of the upscaled latent the second pass rewrites. 0.3-0.4 adds detail and keeps the composition; above ~0.5 it starts to redraw."
                }),
                "refine_steps": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": "Model calls spent on the second pass (steps actually run at refine_denoise)."
                }),
                "hc2_free_corrector": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "HC2 zero-cost corrector (UniPC-style). Off by default. After each HC2 step the model is called at the step's end point anyway, for the next step; with this on, that output is also used to redo the finished step by interpolation instead of extrapolation, then the next step proceeds from the corrected latent. No extra model calls. MEASURED on the analytic bench only (tests/analytic.py, equal calls): EDM Karras 20-32 calls, error 3-3.6x lower than plain HC2; Flow Matching with the model's schedule, a few percent, because the final jump to sigma 0 dominates there. NOT validated on images - A/B it before relying on it. Ignored by other integrators and on steps that follow SDE noise, churn or a restart jump."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/DDRK Omega"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               denoise, scheduler_type, flow_shift, auto_flow_shift, integrator,
               sde_strength,
               sharpness, warmup_steps, auto_optimize, smart_defaults,
               saber_fusion=0.0, saber_mode="auto", use_ema_saber=True,
               ema_decay=0.7, dyn_thresh_percentile=1.0, latent_rescale=0.0, limiter_kappa=1.0, restart_repeats=0, restart_steps=3, restart_t_min=0.10, restart_t_max=0.35, hc2_corrector=0.0, sigma_adapt=0.0, hc2_max_order=2,
               momentum_beta=0.0, sde_seed=-1, s_churn=0.0, s_tmin=0.0,
               s_tmax=999999.0, s_noise=1.0, content_aware=True,
               beta_a=2.0, beta_b=1.0,
               debug_mode=False, debug_tag="", hc2_space="ve",
               refine_scale=1.0, refine_denoise=0.35, refine_steps=5,
               hc2_free_corrector=False):
        # Snapshot BEFORE smart_defaults rewrites anything: the second pass
        # re-enters with exactly what the user asked for.
        call_args = {k: v for k, v in locals().items() if k != "self"}

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


        resolved_scheduler = _resolve_scheduler_name(scheduler_type, sigma_max > 5.0)
        model_shift = False
        if resolved_scheduler in ("ddrk_model", "ddrk_model_beta") and not (sigma_max > 5.0):
            # The model's own schedule already carries its shift; deriving a
            # second one from the latent size would be ignored anyway, and
            # saying nothing would let a log claim a shift that was not used.
            if auto_flow_shift:
                print(f"[DDRK] {scheduler_type}: FM schedule comes from the "
                      f"model's own sampling (shift included); "
                      f"auto_flow_shift and flow_shift={flow_shift:.2f} ignored.")
            model_shift = True
        elif auto_flow_shift and not (sigma_max > 5.0):
            try:
                lat = latent_image["samples"]
                lh, lw = int(lat.shape[-2]), int(lat.shape[-1])
                derived = _resolution_aware_shift(lh, lw)
                print(f"[DDRK] auto_flow_shift: latent {lw}x{lh} -> "
                      f"{(lh // 2) * (lw // 2)} tokens -> shift {derived:.3f} "
                      f"(widget value {flow_shift:.2f} ignored)")
                flow_shift = derived
            except Exception as e:
                print(f"[DDRK] auto_flow_shift failed ({e}); "
                      f"keeping flow_shift={flow_shift}")


        sigmas = get_ddrk_sigmas(
            scheduler_type, steps, sigma_min, sigma_max,
            device=device, flow_shift=flow_shift, warmup_steps=warmup_steps,
            beta_a=beta_a, beta_b=beta_b, auto_optimize=auto_optimize,
            model_sampling=ms,
        )

        # batch_index (from latent batch nodes) picks which noise each item
        # gets, exactly as the stock KSampler does.
        noise = comfy.sample.prepare_noise(latent_samples, seed,
                                           latent.get("batch_index", None))

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
            "hc2_free_corrector": hc2_free_corrector,
            "sigma_adapt": sigma_adapt,
            "restart_repeats": restart_repeats,
            "restart_steps": restart_steps,
            "restart_t_min": restart_t_min,
            "restart_t_max": restart_t_max,
            "latent_rescale": latent_rescale,
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
            # The resolved name, not the widget: 1.7.0 logs said "ddrk_auto"
            # while the schedule actually run was cosine.
            "debug_scheduler_type": (scheduler_type if resolved_scheduler == scheduler_type
                                     else f"{scheduler_type}->{resolved_scheduler}"),
            "debug_flow_shift": "model" if model_shift else flow_shift,
            "debug_steps_requested": steps_denoised,
            "hc2_space": hc2_space,
        }
        sampler_obj = comfy.samplers.KSAMPLER(sample_ddrk_omega, extra_options=extra)

        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler_obj, sigmas, positive, negative,
            latent_image=latent_samples, noise_mask=noise_mask,
            callback=callback, disable_pbar=False, seed=seed
        )

        out = latent.copy()
        out["samples"] = samples
        if float(refine_scale) > 1.0:
            return self._refine(samples, call_args)
        return (out,)

    def _refine(self, samples, call_args):
        """Second pass: upscale the finished latent and re-sample part of it.

        The model draws eyes, hands and fine texture with the pixels it is
        given; at its base size a face may get a few dozen latent cells. The
        pass re-enters sample() with the same settings, refine_denoise as the
        denoise and refine_steps as the step count, so the schedule, the
        integrator and HC2's parameterisation all stay exactly what the user
        chose - only the canvas is larger.
        """
        scale = float(call_args["refine_scale"])
        up = _upscale_latent(samples, scale)
        print(f"[DDRK] refine: latent {list(samples.shape[-2:])} -> "
              f"{list(up.shape[-2:])}, denoise {call_args['refine_denoise']:.2f}, "
              f"{int(call_args['refine_steps'])} steps")
        args = dict(call_args)
        # Keep the rest of the latent dict: noise_mask (inpainting - the mask
        # is resized to the new canvas when sampling) and batch_index. Up to
        # 1.9.0 the second pass dropped them and redrew masked-out regions.
        args.update(latent_image={**call_args["latent_image"], "samples": up},
                    refine_scale=1.0,
                    denoise=float(call_args["refine_denoise"]),
                    steps=int(call_args["refine_steps"]))
        if args.get("debug_tag"):
            args["debug_tag"] = f"{args['debug_tag']}_refine"
        return self.sample(**args)


def _upscale_latent(samples: torch.Tensor, scale: float) -> torch.Tensor:
    """Resize a 4-D or 5-D latent by `scale`, keeping sizes even.

    Even, because every patchified model here (Flux, Krea 2, Qwen, Anima)
    folds 2x2 latent cells into one token. bislerp is ComfyUI's latent
    resampler; plain bicubic is the fallback when it is unavailable.
    """
    five = samples.dim() == 5
    x = samples
    if five:
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    h, w = int(x.shape[-2]), int(x.shape[-1])
    nh = max(2, int(round(h * scale / 2.0)) * 2)
    nw = max(2, int(round(w * scale / 2.0)) * 2)
    try:
        import comfy.utils
        y = comfy.utils.common_upscale(x, nw, nh, "bislerp", "disabled")
    except Exception as e:
        print(f"[DDRK] refine: bislerp unavailable ({type(e).__name__}); "
              f"using bicubic")
        y = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bicubic",
                                            align_corners=False)
    if five:
        y = y.reshape(b, t, c, nh, nw).permute(0, 2, 1, 3, 4)
    return y.contiguous()


# Order 3 needs history to pay for its bootstrap call. Measured on the analytic
# convergence problem, HC2 order 3 is WORSE than order 2 below this many steps
# (at 2 steps it loses even to Euler) and only pulls ahead from 5 upward.
_HC2_ORDER3_MIN_STEPS = 5


# Second pass used by Lite "best" on Flow Matching (see _lite_sampler_params).
_LITE_REFINE_SCALE = 1.33
_LITE_REFINE_DENOISE = 0.35


def _lite_refine(steps: int) -> dict:
    """Second-pass settings for Lite "best": ~40% of the base steps, >= 4."""
    return {"refine_scale": _LITE_REFINE_SCALE,
            "refine_denoise": _LITE_REFINE_DENOISE,
            "refine_steps": max(4, int(round(steps * 0.4)))}


def _lite_sampler_params(is_edm: bool, quality: str, character: str,
                         steps: int = 20) -> dict:
    """Map the two Lite dials onto validated sampler parameters.

    On Flow Matching every quality level costs the same one model call per step,
    apart from a single extra bootstrap call at "best". The dial therefore trades
    numerical aggressiveness, not speed - the speed control is the steps widget.
    Euler is deliberately absent: HC2 order 2 costs exactly the same one call per
    step and was measured more accurate at every step count tested, from 2 steps
    (4x) through 20 (41x), so there is no configuration in which Euler is the
    better trade.
    """
    if is_edm:
        quality_params = {
            "fast": {"integrator": "euler", "sigma_adapt": 0.0},
            "balanced": {"integrator": "heun", "sigma_adapt": 0.0},
            "best": {"integrator": "rk4", "sigma_adapt": 0.0},
        }
    else:
        # "best" since 1.9.0: HC2 order 2 plus the second pass, not HC2
        # order 3. Order 3 was accepted on 1 of 9 steps in live Krea 2 runs
        # and never showed an image gain; the second pass showed one on
        # every seed of the 1.9.0 A/B (RedCraft Krea 2 x3, Anima x2).
        quality_params = {
            "fast": {"integrator": "hc2", "hc2_max_order": 2, "sigma_adapt": 0.0},
            "balanced": {"integrator": "hc2", "hc2_max_order": 2, "sigma_adapt": 0.10},
            "best": {"integrator": "hc2", "hc2_max_order": 2,
                     "sigma_adapt": 0.10, **_lite_refine(steps)},
        }
    character_params = {
        "neutral": {"sharpness": 0.0, "saber_fusion": 0.0},
        "sharp": {"sharpness": 0.12, "saber_fusion": 0.0},
        "smooth": {"sharpness": 0.0, "saber_fusion": 0.10},
    }
    try:
        return quality_params[quality] | character_params[character]
    except KeyError as e:
        raise ValueError(f"Unknown DDRK Omega Lite setting: {e.args[0]!r}") from None


class DDRKOmegaLiteKSamplerNode(DDRKOmegaUnifiedKSamplerNode):
    DESCRIPTION = ('DDRK with two dials (quality, character) on top of your own steps and CFG. Settings mapped to validated full-node values.')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "latent_image": ("LATENT", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "quality": (["fast", "balanced", "best"], {"default": "balanced"}),
                "character": (["neutral", "sharp", "smooth"], {"default": "neutral"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample_lite"
    CATEGORY = "sampling/DDRK Omega"

    def sample_lite(self, model, positive, negative, latent_image, seed, steps,
                    cfg, quality, character):
        ms = model.get_model_object("model_sampling")
        params = _lite_sampler_params(float(ms.sigma_max) > 5.0,
                                      quality, character, steps=steps)
        return super().sample(
            model, positive, negative, latent_image, seed, steps, cfg,
            denoise=1.0,
            scheduler_type="ddrk_auto",
            flow_shift=3.0,
            auto_flow_shift=False,
            integrator=params["integrator"],
            sde_strength=0.0,
            sharpness=params["sharpness"],
            warmup_steps=0,
            auto_optimize=False,
            smart_defaults=False,
            saber_fusion=params["saber_fusion"],
            saber_mode="auto",
            use_ema_saber=True,
            ema_decay=0.7,
            # Off since 1.8.0: at 0.995 it clipped the final FM latent in 4 of
            # 6 live Krea 2 runs (see sample_ddrk_omega).
            dyn_thresh_percentile=1.0,
            latent_rescale=0.0,
            limiter_kappa=1.0,
            restart_repeats=0,
            restart_steps=3,
            restart_t_min=0.10,
            restart_t_max=0.35,
            hc2_corrector=0.0,
            sigma_adapt=params["sigma_adapt"],
            hc2_max_order=params.get("hc2_max_order", 2),
            momentum_beta=0.0,
            sde_seed=-1,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=999999.0,
            s_noise=1.0,
            content_aware=True,
            beta_a=2.0,
            beta_b=1.0,
            debug_mode=False,
            debug_tag="",
            refine_scale=params.get("refine_scale", 1.0),
            refine_denoise=params.get("refine_denoise", 0.35),
            refine_steps=params.get("refine_steps", 5),
        )


# --------------------------------------------------------------------------
# DDRK Omega Auto: one node, no sampler knowledge needed.
# --------------------------------------------------------------------------

# Starting values per model: (label, CFG, steps at "balanced"). Keyed by
# ComfyUI's supported_models class name, the most reliable statement of what
# the model is. They are the values the model makers and ComfyUI's own
# templates use, not DDRK measurements; finetunes and LoRAs move them, which
# is why the node prints what it chose and takes overrides.
_AUTO_PRESETS = {
    "SD15": ("SD 1.5", 7.0, 25),
    "SD20": ("SD 2.x", 7.0, 25),
    "SDXL": ("SDXL", 6.0, 25),
    "SSD1B": ("SDXL (SSD-1B)", 6.0, 25),
    "Segmind_Vega": ("SDXL (Vega)", 6.0, 25),
    "KOALA_700M": ("SDXL (KOALA)", 6.0, 25),
    "KOALA_1B": ("SDXL (KOALA)", 6.0, 25),
    "SDXLRefiner": ("SDXL refiner", 5.0, 20),
    "SD3": ("SD 3 / 3.5", 4.5, 28),
    "AuraFlow": ("AuraFlow", 3.5, 25),
    "PixArtAlpha": ("PixArt-alpha", 4.5, 20),
    "PixArtSigma": ("PixArt-sigma", 4.5, 20),
    "HunyuanDiT": ("Hunyuan-DiT", 6.0, 25),
    "HunyuanDiT1": ("Hunyuan-DiT", 6.0, 25),
    # Guidance-distilled: CFG 1, the strength comes from the FluxGuidance
    # value in the conditioning (3.5 when none is set).
    "Flux": ("Flux (guidance-distilled)", 1.0, 20),
    "Flux2": ("Flux 2 (guidance-distilled)", 1.0, 24),
    "Chroma": ("Chroma", 4.0, 26),
    "Lumina2": ("Lumina 2", 4.0, 30),
    "QwenImage": ("Qwen-Image", 2.5, 20),
    "QwenImage21": ("Qwen-Image", 2.5, 20),
    "HiDream": ("HiDream (dev settings)", 1.0, 28),
    "Anima": ("Anima", 4.0, 25),
    "WAN21_T2V": ("Wan (video)", 5.0, 25),
    "WAN22_T2V": ("Wan (video)", 5.0, 25),
    "WAN21_I2V": ("Wan (video)", 5.0, 25),
    "LTXV": ("LTX-Video", 3.0, 30),
    "HunyuanVideo": ("HunyuanVideo (embedded guidance)", 1.0, 25),
    "CosmosT2V": ("Cosmos", 7.0, 35),
    "CosmosT2IPredict2": ("Cosmos Predict2", 4.0, 35),
}

# Models that are few-step by construction: always sampled in turbo mode.
_AUTO_FEW_STEP = {
    "FluxSchnell": "Flux schnell",
    "ZImage": "Z-Image Turbo",
    "HunyuanVideo15_SR_Distilled": "HunyuanVideo 1.5 SR (distilled)",
}

_AUTO_TURBO_STEPS = {"fast": 4, "balanced": 6, "best": 8}
_AUTO_STEP_SCALE = {"fast": 0.6, "balanced": 1.0, "best": 1.5}


def _auto_settings(model, latent, quality: str, model_type: str,
                   steps: int, cfg: float) -> dict:
    """Everything the Auto node decides, as one dict (unit-testable)."""
    if quality not in _AUTO_STEP_SCALE:
        raise ValueError(f"Unknown quality {quality!r}.")
    if model_type not in ("auto", "standard", "turbo / lightning / few-step"):
        raise ValueError(f"Unknown model_type {model_type!r}.")
    ms = model.get_model_object("model_sampling")
    try:
        import comfy.model_sampling as cms
        is_edm = not isinstance(ms, cms.CONST)
    except Exception:
        is_edm = float(ms.sigma_max) > 5.0
    inner = getattr(model, "model", None)
    cfg_obj = getattr(inner, "model_config", None)
    cls = type(cfg_obj).__name__ if cfg_obj is not None else ""

    few_step = cls in _AUTO_FEW_STEP
    turbo = few_step or model_type == "turbo / lightning / few-step"
    if model_type == "standard":
        turbo = False
    if few_step:
        label = _AUTO_FEW_STEP[cls]
        base_cfg, base_steps = 1.0, 6
    elif cls in _AUTO_PRESETS:
        label, base_cfg, base_steps = _AUTO_PRESETS[cls]
    else:
        label = "EDM model" if is_edm else "Flow Matching model"
        base_cfg, base_steps = (6.0, 25) if is_edm else (3.5, 24)
    family = "EDM" if is_edm else "Flow Matching"

    if turbo:
        auto_steps, auto_cfg = _AUTO_TURBO_STEPS[quality], 1.0
    else:
        auto_steps = max(4, int(round(base_steps * _AUTO_STEP_SCALE[quality]))) \
            if (is_edm or quality != "best") else base_steps
        auto_cfg = base_cfg
    use_steps = int(steps) if steps and steps > 0 else auto_steps
    use_cfg = float(cfg) if cfg and cfg > 0 else auto_cfg

    video = latent is not None and getattr(latent, "dim", lambda: 4)() == 5
    refine = {}
    # "best" on Flow Matching = the second pass, the one quality step the
    # 1.9.0 image A/B confirmed (Krea 2 x3, Anima x2). Not on EDM (latent
    # upscaling there was never A/B-tested), not on video, not in turbo mode.
    if quality == "best" and not is_edm and not turbo and not video:
        refine = _lite_refine(use_steps)

    return {
        "label": label, "family": family, "model_class": cls or "?",
        "is_edm": is_edm, "turbo": turbo,
        "steps": use_steps, "cfg": use_cfg,
        "steps_auto": not (steps and steps > 0),
        "cfg_auto": not (cfg and cfg > 0),
        # Karras on EDM; in turbo mode the model's own schedule, whose 4
        # steps land on the timesteps (999/749/499/249) that Lightning/Turbo
        # style models are distilled at. Flow Matching: the model's own.
        "scheduler": ("ddrk_model" if (is_edm and turbo) else "ddrk_auto"),
        "integrator": "hc3",
        **refine,
    }


def _auto_summary(st: dict, denoise: float) -> str:
    def mark(auto):
        return " (auto)" if auto else ""
    sched = {"ddrk_auto": ("Karras" if st["is_edm"] else "model schedule"),
             "ddrk_model": "model schedule"}[st["scheduler"]]
    parts = [f"{st['label']} [{st['family']}]",
             f"{st['steps']} steps{mark(st['steps_auto'])}",
             f"CFG {st['cfg']:g}{mark(st['cfg_auto'])}",
             f"HC3 integrator, {sched}"]
    if st["turbo"]:
        parts.append("turbo/few-step mode")
    if denoise < 1.0:
        parts.append(f"denoise {denoise:g}")
    calls = st["steps"]
    if st.get("refine_scale", 1.0) > 1.0:
        parts.append(f"second pass {st['refine_scale']:g}x "
                     f"({st['refine_steps']} steps, denoise {st['refine_denoise']:g})")
        calls += st["refine_steps"]
    parts.append(f"~{calls} model calls")
    return "DDRK Omega Auto: " + " | ".join(parts)


class DDRKOmegaAutoNode(DDRKOmegaUnifiedKSamplerNode):
    DESCRIPTION = ("One-click DDRK sampling. Connect model, prompts and latent; the node "
                   "recognises the model (SD 1.5, SDXL, SD3, Flux, Qwen-Image, Chroma, "
                   "Wan and more) and chooses steps, CFG, schedule and integrator itself. "
                   "The 'settings' output says exactly what it chose. Set steps or CFG "
                   "above 0 to override them.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "latent_image": ("LATENT", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True}),
                "quality": (["fast", "balanced", "best"], {"default": "balanced",
                    "tooltip": "fast: ~0.6x the steps. balanced: the model's usual step count. "
                               "best: EDM (SD/SDXL) 1.5x the steps; Flow Matching (Flux, SD3, Qwen...) "
                               "adds a second pass at 1.33x resolution, ~1.7x the time and more VRAM."}),
                "model_type": (["auto", "standard", "turbo / lightning / few-step"], {"default": "auto",
                    "tooltip": "Pick 'turbo / lightning / few-step' for Turbo, Lightning, Hyper, LCM, "
                               "DMD or schnell-style models and LoRAs: 4-8 steps at CFG 1. 'auto' "
                               "recognises few-step base models (Flux schnell, Z-Image Turbo) but "
                               "cannot see a turbo LoRA or finetune."}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "0 = automatic for this model and quality. Anything above 0 is used as is."}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "0 = automatic for this model (e.g. SDXL 6, SD3 4.5, Flux 1). "
                               "Anything above 0 is used as is."}),
            },
            "optional": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "1.0 for text-to-image. Below 1.0 for img2img: how much of the input "
                               "latent is redrawn."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "settings")
    OUTPUT_TOOLTIPS = ("The sampled latent - connect it to VAE Decode.",
                       "What the node chose: model, steps, CFG, integrator, schedule.")
    FUNCTION = "sample_auto"
    CATEGORY = "sampling/DDRK Omega"

    def sample_auto(self, model, positive, negative, latent_image, seed, quality,
                    model_type, steps, cfg, denoise=1.0):
        st = _auto_settings(model, latent_image.get("samples"), quality,
                            model_type, steps, cfg)
        summary = _auto_summary(st, float(denoise))
        print(f"[DDRK] {summary}")
        out = super().sample(
            model, positive, negative, latent_image, seed, st["steps"], st["cfg"],
            denoise=float(denoise),
            scheduler_type=st["scheduler"],
            flow_shift=3.0,
            auto_flow_shift=False,
            integrator=st["integrator"],
            sde_strength=0.0,
            sharpness=0.0,
            warmup_steps=0,
            auto_optimize=False,
            smart_defaults=False,
            saber_fusion=0.0,
            dyn_thresh_percentile=1.0,
            latent_rescale=0.0,
            limiter_kappa=1.0,
            restart_repeats=0,
            hc2_corrector=0.0,
            sigma_adapt=0.0,
            hc2_max_order=2,
            momentum_beta=0.0,
            sde_seed=-1,
            s_churn=0.0,
            content_aware=True,
            refine_scale=st.get("refine_scale", 1.0),
            refine_denoise=st.get("refine_denoise", 0.35),
            refine_steps=st.get("refine_steps", 5),
        )
        return (out[0], summary)


class DDRKFluxConditioning:
    DESCRIPTION = ('Inspect and reshape text conditioning: padding attenuation, token gain and norm equalisation. Prints what it did.')


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "pad_attenuation": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Scales padding tokens toward zero. 0 = untouched (stock Flux), 1 = fully zeroed (what Chroma does). Only acts when the encoder supplied an attention_mask; without one the padding boundary is unknown and this is skipped. Try 0.5-1.0 on short prompts, where padding dominates the sequence."
                }),
                "active_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Multiplier on real (non-padding) tokens. 1.0 = off. Raising it strengthens prompt influence in a blunt way; large values distort the embedding geometry the model was trained on, so stay near 1."
                }),
                "norm_equalize": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends per-token vector norms toward the mean norm of the active region. 0 = off, 1 = every active token has equal norm. Attention weight scales with token magnitude, so a few high-norm tokens can crowd out the rest; this evens that out. Directions are preserved - only lengths change."
                }),
                "front_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales the first third of the ACTIVE region. 1.0 = off."
                }),
                "mid_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales the middle third of the ACTIVE region. 1.0 = off."
                }),
                "end_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales the last third of the ACTIVE region. 1.0 = off. Note these thirds are token positions, not meanings - the model assigns no fixed role to prompt position, so treat this as a coarse emphasis dial, not semantic targeting."
                }),
                "report": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print detected layout, token counts and before/after norms to the console. On by default because a conditioning transform you cannot inspect is indistinguishable from a placebo."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "sampling/DDRK Omega/utils"

    _LAYOUTS = {
        4096: "Flux.1 (T5-XXL, 4096)",
        7680: "FLUX.2 klein 4B (3 x 2560)",
        12288: "FLUX.2 klein 9B (3 x 4096)",
        2048: "SDXL (CLIP-G + CLIP-L)",
        768: "SD1.x (CLIP-L)",
    }

    def apply(self, conditioning, pad_attenuation, active_gain, norm_equalize,
              front_scale, mid_scale, end_scale, report):
        neutral = (pad_attenuation == 0.0 and active_gain == 1.0
                   and norm_equalize == 0.0 and front_scale == 1.0
                   and mid_scale == 1.0 and end_scale == 1.0)
        if neutral:
            if report:
                print("[DDRK Cond] all controls neutral - passthrough")
            return (conditioning,)

        out = []
        for idx, entry in enumerate(conditioning):
            emb, meta = entry[0], entry[1]
            meta = meta.copy()
            if not isinstance(emb, torch.Tensor) or emb.dim() < 2:
                out.append([emb, meta])
                continue

            t = emb.clone().float()
            width = int(t.shape[-1])
            seq = int(t.shape[-2])
            layout = self._LAYOUTS.get(width, f"unknown ({width}-wide)")

            mask = meta.get("attention_mask", None)
            if isinstance(mask, torch.Tensor) and mask.numel() >= seq:
                m = mask.reshape(-1)[:seq].to(t.device)
                active = int(m.sum().item())
            else:
                m = None
                active = seq

            active = max(1, min(active, seq))
            norm_before = float(t[..., :active, :].norm(dim=-1).mean().item())

            if m is not None and pad_attenuation > 0.0 and active < seq:
                keep = m.to(t.dtype).reshape(
                    *([1] * (t.dim() - 2)), seq, 1)
                t = t * (keep + (1.0 - keep) * (1.0 - pad_attenuation))

            if active_gain != 1.0:
                t[..., :active, :] *= active_gain

            if norm_equalize > 0.0:
                seg = t[..., :active, :]
                n = seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                target = n.mean(dim=-2, keepdim=True)

                t[..., :active, :] = seg * (
                    (1.0 - norm_equalize) + norm_equalize * (target / n))

            if front_scale != 1.0 or mid_scale != 1.0 or end_scale != 1.0:


                a = active // 3
                b = (2 * active) // 3
                if front_scale != 1.0:
                    t[..., :a, :] *= front_scale
                if mid_scale != 1.0:
                    t[..., a:b, :] *= mid_scale
                if end_scale != 1.0:
                    t[..., b:active, :] *= end_scale

            norm_after = float(t[..., :active, :].norm(dim=-1).mean().item())
            t = t.to(emb.dtype)

            if report:
                pad_note = (f"{seq - active} padding"
                            if m is not None else "no attention_mask - "
                                                  "padding boundary unknown, "
                                                  "pad_attenuation skipped")
                print(f"[DDRK Cond] #{idx} {layout} | seq {seq} | "
                      f"{active} active, {pad_note} | "
                      f"mean active norm {norm_before:.4f} -> {norm_after:.4f}")

            out.append([t, meta])
        return (out,)


NODE_CLASS_MAPPINGS = {
    "DDRKOmegaAutoNode": DDRKOmegaAutoNode,
    "DDRKFluxConditioning": DDRKFluxConditioning,
    "DDRKOmegaSchedulerNode": DDRKOmegaSchedulerNode,
    "DDRKOmegaSamplerNode": DDRKOmegaSamplerNode,
    "DDRKOmegaUnifiedKSamplerNode": DDRKOmegaUnifiedKSamplerNode,
    "DDRKOmegaLiteKSamplerNode": DDRKOmegaLiteKSamplerNode,
    "DDRKOmegaSmartConfigNode": DDRKOmegaSmartConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDRKOmegaAutoNode": "DDRK Omega Auto (one-click)",
    "DDRKFluxConditioning": "DDRK Flux Conditioning",
    "DDRKOmegaSchedulerNode": "DDRK Omega Scheduler",
    "DDRKOmegaSamplerNode": "DDRK Omega Sampler",
    "DDRKOmegaUnifiedKSamplerNode": "DDRK Omega Unified KSampler (advanced)",
    "DDRKOmegaLiteKSamplerNode": "DDRK Omega Lite",
    "DDRKOmegaSmartConfigNode": "DDRK Omega Smart Config",
}
