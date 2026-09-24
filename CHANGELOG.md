# Changelog

## 1.10.0 — 2026-09-24

An audit release. 1.9.0 did not load at all; beyond that, every change below
is either a bug fix with a regression test or an opt-in option, and the
default Flow Matching path (Lite, pinned HC2) is bit-identical to 1.9.0.
Numbers marked *analytic* come from the new CPU bench (`tests/analytic.py`):
latent elements drawn from a Gaussian mixture, for which the denoiser and the
exact ODE solution are known, so the error of a sampler can be measured
directly. That checks numerics. It says nothing about images, and nothing
below has been run on a GPU.

### Fixed

- **1.9.0 could not be imported.** Two stray lines after
  `NODE_DISPLAY_NAME_MAPPINGS` were an `IndentationError`, so ComfyUI skipped
  the whole pack: none of the six nodes appeared.
- **EDM img2img, hires fix, the second pass and SplitSigmas tails ran on the
  Flow Matching path.** The family was guessed from the schedule
  (`sigmas.max() > 5`), so an SDXL/SD1.5 run starting below sigma 5 (denoise
  below ~0.6) got the FM SDE formula (which scales by 1 - sigma, negative
  above sigma 1), FM sharpening and no EDM clamp. The sampler now asks the
  model, as ComfyUI's own samplers do (`isinstance(model_sampling, CONST)`),
  and falls back to the schedule only for direct calls. *Analytic*, EDM
  img2img with the Unified node's default settings: RMSE to the exact
  solution 0.27 -> 0.004 at denoise 0.5-0.7, 0.033 -> 0.004 at 0.3, and the
  1.9.0 result had ~8% less contrast (std 0.47 vs 0.51).
- **`s_churn` did not follow Karras Alg. 2.** gamma was
  `min(s_churn / sigma, sqrt(2) - 1)`; the algorithm (and k-diffusion) uses
  `s_churn / N`, the number of steps. The old form sat at its cap on almost
  every step, so values from ~5 up all behaved alike. Existing workflows that
  use churn will look different: the same number now means what it means in
  every other sampler.
- **`momentum_beta` made Heun and RK4 first order.** The AB2 extrapolation was
  applied on top of their slopes. It now applies to Euler steps only, where
  it is a genuine Adams-Bashforth 2 step (beta = 1 -> order 2); Heun and RK4
  ignore it, like HC2 always did. *Analytic*: at beta 0.25 RK4 went from
  order 4.0 to 0.9 (error ~250x at 128 calls) and Heun from 2.0 to 1.1-1.2.
  On the realistic schedules the effect is smaller and one-sided by family:
  EDM Karras, error 1.1-1.9x (Heun) and 1.2-6.2x (RK4) higher with momentum;
  Flow Matching with the model's schedule, 0-17% *lower*, because there the
  extrapolation partly offsets the large final jump to sigma 0. This matters
  for defaults: the Sampler and Unified nodes default to momentum 0.25 with
  `auto`, which on EDM means Heun/RK4 phases.
- **The final guard clamp clipped noisy hand-offs.** A fixed +-7 (EDM) /
  +-20 (FM) is right for a finished latent but clipped one handed on still
  noisy (a SplitSigmas head at sigma 5 has std ~5). The bound now grows with
  the output sigma; ending at 0 it is exactly the old bound.
- **The second pass dropped `noise_mask` and `batch_index`** and so redrew
  masked-out (inpainting) regions. Both are now carried into it.
- **`batch_index` was ignored by the Unified/Lite noise**, unlike the stock
  KSampler (`prepare_noise(..., batch_index)`).
- **FM noise ignored the model's `noise_scale`** (HiDream-O1 declares 8). SDE
  and restart noise are now scaled by it, as ComfyUI's RF samplers do.
- The bundled `workflow/ddrk_test_workflow.json` stored 11 widget values for a
  node that now has 29, so ComfyUI shifted every value onto the wrong input.
  Regenerated for the current nodes, with a test that keeps it in sync.
- `pyproject.toml` pointed the registry icon at `ddrk_logo.png`, which was
  deleted; it is `logo.png`.
- Telemetry logged HC2's statistics on non-HC2 steps in auto mode.

### Changed

- **`auto` mode: HC2 steps extrapolate from the previous step.** Every
  integrator now records the denoiser output at its step's start into HC2's
  history; before, an HC2 step after a Heun/RK4 step used whatever HC2 step
  came last, possibly many steps back. *Analytic*: EDM Karras, error 1.6-2.6x
  lower at 39-49 calls and up to 5% higher at 13-29; Flow Matching, equal or
  up to 14% lower.
- **Smart Config / `smart_defaults` on Flow Matching now pick `hc2`** instead
  of `auto` - the integrator the 1.9.0 image A/B found most accurate at equal
  calls, and the one Lite uses. The hint text no longer recommends Heun/RK4 on
  FM.
- HC2 skips its GPU->CPU statistics (limiter fraction, activity) when neither
  telemetry nor `sigma_adapt` reads them.

### Added

- **`hc2_free_corrector`** (Unified and Sampler nodes, off by default): a
  zero-cost corrector in the style of UniPC's UniC (Zhao et al. 2023). After
  each HC2 step the model is called at the step's end point anyway; that
  output is also used to redo the finished step by interpolation instead of
  extrapolation. No extra model calls. *Analytic*: EDM Karras, error 1.1-2x
  lower than HC2 at 8-12 calls and 1.3-4x at 20-30; the asymptotic order
  rises from 2 to ~3. On Flow Matching with the model's schedule it is within
  a few percent (2% worse to 5% better), because there the final one-shot
  jump to sigma 0 dominates the error and no multistep correction reaches
  it. Not validated on images: opt-in until an A/B.
- `hc2_space` on the Sampler node (it was Unified-only).
- A test suite: `tests/`, 148 CPU tests against a real ComfyUI checkout -
  schedules, convergence orders, equal-call comparisons, determinism, batch
  isolation, 4D/5D, call counts, telemetry, every fix above, and the Lite =
  Unified `torch.equal` check across all quality/character combinations on
  both families. Each 1.10.0 fix has a test that fails on 1.9.0. Plus
  `tests/bench_analytic.py`, which prints the accuracy tables, and a GitHub
  workflow that runs the suite.

### Measured, and negative (analytic)

- **`sigma_adapt = 0.10` was 0-22% *less* accurate than no adaptation** on
  both families and both mixtures. Lite `balanced`/`best` keep it (the 1.8.0
  controller has still not been compared on images), but this is one more
  reason to A/B it.
- **HC2 order 3 only reaches order 3 with the limiter effectively off**
  (kappa 100: 3.07); at the default kappa 1 it converges at order 2, because
  the per-element limiter clips the correction wherever the first-order step
  passes through zero.
- **`hc2_space = flow` was not more accurate** than `ve` on the analytic FM
  problem with the model's schedule (from 9% worse to 2% better), matching the
  1.9.0 "draw" on images.
- A final-step Richardson extrapolation (in sigma^2, no extra calls) was tried
  for the FM final jump: 1.7-2.5x better at 6-8 steps, worse at 20-32 steps
  and on EDM. Not shipped.
- HC2 order 2 and ComfyUI's `dpmpp_2m` are within a few percent of each other
  everywhere measured, as expected: both are second-order exponential
  multistep methods.

## 1.9.0 — 2026-09-23

The first release validated on the GPU image path, not only by CPU tests.
Test bench: one process loads the model once and runs every variant with the
same seed, prompt and resolution; decoded images are compared by eye and by
RMSE to a high-accuracy reference (RK4, 24 steps = 96 model calls) of the same
seed. Models: RedCraft Krea 2 (Flow Matching, distilled, 12 steps, CFG 1,
1024x1024, seeds 505050/5050/50) and MolKeunMix Anima (Flow Matching,
25 steps, CFG 4, 832x1216, seeds 505050/5050). Comparison sheets are in
`docs/ab-1.9.0/`.

### Measured (these back the 1.8.0 changes as well)

- **Canvas size is the largest single factor.** Krea 2 at 512x512 vs
  1024x1024, same seed and settings: hands, faces and fabric only resolve at
  1 MP. (Relevant to wrappers that read "native size" from CivitAI metadata,
  which reported 512x512 for a model trained at 1-3 MP.)
- **The pre-1.8.0 FM defaults were the "washed-out Anima" bug.** Cosine
  schedule + `auto_flow_shift` + limiter 0.995 + old `sigma_adapt`: RMSE to
  the reference 0.286 / 0.281 against 0.108 / 0.155 for 1.8.0 defaults - an
  overexposed, low-contrast image, visible at a glance.
- **HC2 beats Euler and Heun at equal model calls** (Anima, 25 calls):
  RMSE 0.108 / 0.155 (HC2) vs 0.123 / 0.210 (Euler) vs 0.123 / 0.230
  (Heun, 13 steps). On distilled Krea 2 at 12 steps all three were visually
  close - the integrator matters where the trajectory is hard (CFG > 1).

### Added

- **Second pass (`refine_scale`, `refine_denoise`, `refine_steps`)** on the
  Unified node: the finished latent is upscaled (bislerp, even sizes, 4-D and
  5-D) and re-sampled from `refine_denoise` with the same settings. Off by
  default (`refine_scale = 1.0`). Measured at 1.33x / 0.35: visibly more skin,
  hair and fabric detail on every A/B seed (Krea 2 x3, Anima x2) without
  changing the composition; +50-60 s per image on an 8 GB RTX 2070 (Krea 2
  12 steps: 71 s -> 125 s), peak VRAM 7.1 GB at 1360x1360.
- **`hc2_space`** (`ve` default, `flow`): HC2 in the exact Flow Matching
  parameterisation - lambda = log((1-sigma)/sigma) with a (1-sigma) weight on
  the correction. On images it measured a draw (0.106 vs 0.108, 0.165 vs
  0.155), on the analytic FM Gaussian problem it is more accurate from 10
  steps up and less accurate at 6-8, so it ships opt-in and `ve` stays the
  default. No effect on EDM.

### Changed

- **Lite `best` on Flow Matching = HC2 order 2 + second pass** (1.33x,
  denoise 0.35, ~40% of the steps, at least 4), replacing HC2 order 3, which
  was accepted on 1 of 9 steps in live runs and never showed an image gain.
- **`auto_optimize` on FM at <= 10 steps turns `auto` into `hc2`**, not
  `euler`: same cost per step, more accurate at every step count measured. Its
  console line no longer claims "scheduler=linear, shift<=1.0" - it never
  touched the schedule.

### Fixed

- **SDE on Flow Matching used the variance-exploding ancestral recipe**
  (step to sigma_down, add sigma_up noise), which over-noises FM latents and
  never rescales the signal. It now uses the FM recipe (the one ComfyUI's
  `euler_ancestral_RF` uses): scale by (1-sigma_next)/(1-sigma_down) and
  re-noise so the total noise std is exactly sigma_next.
- **Restart jumps on Flow Matching** added sqrt(s'^2 - s^2) of noise with no
  signal rescale, so the latent landed on the wrong marginal. They now scale
  by (1-s')/(1-s) and add exactly the missing noise. EDM restarts unchanged.

## 1.8.0 — 2026-09-23

These changes come from six live Elysium runs on Krea 2 Turbo (Animosity,
10 steps, CFG 1, HC2) with `debug_mode` on: 15-22 September 2026, 512x512 and
768x1344, JSON files `ddrk_debug_seed*.json`. No GPU generation was run for
this release. Everything below is checked by CPU tests only (`test_g_release_180.py`),
and the visual effect still needs an A/B on the GPU.

### Changed (sampler output changes)

- **New schedule `ddrk_model`.** It builds sigmas with
  `comfy.samplers.calculate_sigmas(model_sampling, "simple", steps)`, so the
  shift comes from the model itself, plus any `ModelSampling*` node in the
  graph. When no model sampling is available (direct calls), it falls back
  loudly to `ddrk_flow_linear` on FM or `ddrk_edm_karras` on EDM, and never
  silently.
- **`ddrk_auto` on Flow Matching now resolves to `ddrk_model`**, not
  `ddrk_cosine`. The telemetry showed two problems with cosine plus a shift:
  - the first two steps covered 2.7% of the sigma range but cost 3 of 11 model
    calls;
  - on 1 MP latents, the final one-shot jump to zero came from 0.367-0.403. The
    model's own schedule ends it at ~0.26.
  `auto_optimize` no longer changes the schedule (it still caps enhancers).
  `flow_shift` and `auto_flow_shift` are ignored by `ddrk_model`, and the
  Unified node says so in the console. The telemetry header now records the
  resolved schedule (`ddrk_auto->ddrk_model`) and `flow_shift: "model"`.
- **`sigma_adapt` controller rewritten** (`_adapt_remaining_sigmas`):
  - *Sign.* A step with above-average HC2 activity used to lengthen the next
    step (seed 227007225: step 3 doubled, then 36-43% of the elements hit the
    limiter). Now it shortens it.
  - *Last sigma.* The scale saturated at 1.1 in all six runs and raised the
    last non-zero sigma by 10% (0.258->0.284, 0.367->0.403). It is now never
    raised above its reference value.
  - *Minimum step.* Clamping produced a 0.737->0.721 step (delta 0.017) that
    cost a full model call. No adapted step is now shorter than half its
    reference step.
  - The +3.3% dynamic-range result for `sigma_adapt=0.10` was measured on the
    old controller. The Lite preset keeps 0.10, but this needs a new image A/B.
- **`dyn_thresh_percentile` defaults to 1.0 (off)** in the Sampler node, the
  Unified node, Lite and direct `sample_ddrk_omega` calls. At 0.995 it clipped
  the final FM latent in 4 of 6 runs: the final max was exactly -min (±1.4350,
  ±1.5301, ±1.7353, ±1.3914). That flattens the brightest 0.5% of values. It is
  the prime suspect for dotted halos around lights.

### Fixed (honesty)

- The integrator replacement rules (`<=6` steps -> euler; EDM `<=10` steps:
  rk4 -> heun) print "REPLACED ... even with auto_optimize off". Telemetry now
  records `integrator_requested` next to `integrator_param`.

## 1.7.0 — 2026-08-20

### Fixed

- Removed the unused private `AdaptiveSDE._get_gen` alias; all internal generator access already used `get_generator`.
- Corrected both sampler-node `sde_seed=-1` tooltips. The seed is derived from ComfyUI's globally seeded Torch RNG and is reproducible from the workflow seed; it is not inherently non-deterministic.
- Documented UI overrides, dead code, error-handling gaps, duplicated logic, and swallowed exceptions in `AUDIT.md`.

### Added

- Added **DDRK Omega Lite**, a thin wrapper over the existing Unified node and `sample_ddrk_omega` implementation.
- Added auditable Flow Matching/EDM mappings for Lite's `quality` and `character` controls.
- Added tests for Lite registration, schema, mappings, invalid dial values, and bit-exact matching against the full node for every FM/EDM quality/character combination.
- Extended `b1_equiv.py` to four batch=1 cases covering 4D and 5D latents, varied seeds, and the full enhancer stack, with strict `torch.equal` assertions against `sampler.py.bak-1.6.0`.
- Documented the measured integrator convergence orders, equal-call accuracy results, HC2's EDM negative result, SABER's smoothing cost, and SDE's batch-shape limitation.

### Changed

- **Lite `quality=fast` on Flow Matching now uses HC2 order 2 instead of Euler.** HC2 order 2 costs the same one model call per step and was measured more accurate at every step count from 2 upward, so Euler gave up accuracy for no saving. The FM dial now reads: fast = HC2 order 2, balanced = adds the measured `sigma_adapt=0.10`, best = HC2 order 3.
- **Lite `quality=best` falls back to HC2 order 2 below 5 steps.** Order 3 pays a bootstrap model call and is measurably worse than order 2 on short schedules. `_lite_sampler_params` now takes `steps`.
- The EDM quality row is unchanged and remains uncompared at equal compute.

### Ablation integrity

These five guards change behaviour only on malformed input or on a run that has
already produced non-finite values. Every valid configuration is bit-exact
against 1.6.0, verified across 48 scheduler/shape/integrator combinations.

- An unknown `integrator` name now raises `ValueError` instead of silently
  running Euler. Unreachable through the node dropdowns, reachable through a
  typo in a sweep script - where it would record HC2 while measuring Euler.
- An unknown `scheduler_type` now raises `ValueError` instead of silently
  falling back to a default schedule. Same hazard, same reasoning.
- Non-finite output is no longer silent. `torch.clamp` does not remove NaN, so a
  blown-up run used to finish looking like any other result. With `debug_mode`
  on - the measurement regime - it now raises. Without it, generation continues
  with a loud warning so an ordinary run is not lost.
- `_dbg_delta` returns an error marker instead of `None`, so "the metric failed"
  is no longer indistinguishable from "the metric was never taken".
- Model-family detection keeps its best-effort fallbacks but now announces them
  and records `detected_by` and `detect_warnings` in the profile, so a
  misdetection partway through a sweep is visible afterwards.
