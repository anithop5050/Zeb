# ZEB

## Purpose

This document is the blueprint and scan report for the watermarking codebase in this repository.

It has two goals:

1. Explain how the system is structured and how the major paths work.
2. Record what was verified, what was fixed, and what still needs attention.

This is a thorough repo scan, not a guarantee that no future issue exists.

## Repo Map

### Core runtime areas

- `training/`
  - `models.py`: pixel watermark encoder/decoder architectures.
  - `semantic_watermark.py`: semantic or feature-level watermark path.
  - `adversarial_poison.py`: FGSM/PGD-style poison layer used during training.
  - `attacks.py`: differentiable and stochastic attack simulation layer.
  - `utils_loss_metrics.py`: PSNR, BER, SSIM, composite training loss.
  - `train.py`: general training script.
  - `train_production.py`: production-oriented fine-tuning script focused on low alpha.

- `inference/`
  - `inference.py`: main CLI for embed/decode and batch processing.
  - `app.py`: Tkinter GUI for embedding, decoding, previewing, and owner registry workflow.

- `core/`
  - `seed_registry.py`: SQLite-backed ownership registry and audit log.
  - `reliability.py`: save-path validation, alpha clamping, quantization-aware helpers, post-save verification.
  - `registry_cli.py`: seed registry CLI plus batch processing bridge.
  - `seed_registry.db`: SQLite database used by the registry.

- `analysis/`
  - `attack_simulator.py`: GUI for testing attack robustness interactively.

- `tools/`
  - `test_robustness.py`: scripted robustness runner.
  - `extract_subset.py`: dataset extraction helper.

- Tests
  - `tests/unit/test_reliability.py`: coverage for reliability helpers.
  - `tests/regression/test_alpha_range_fix.py`: alpha-range alignment regression checks.
  - `tests/integration/`: currently minimal.

## System Blueprint

### 1. Training path

Training is built as a multi-defense stack:

1. Input image enters `RobustWatermarkEncoder`.
2. A 64-bit watermark is embedded through a U-Net-like residual encoder.
3. `SemanticWatermarkEncoder` optionally adds a low-amplitude geometric/edge-aware perturbation.
4. `AdversarialPoisoner` perturbs images to train decoder resilience.
5. `AttackSimulationLayer` applies synthetic corruption matching deployment threats.
6. `WatermarkDecoder` predicts embedded bits.
7. Composite losses balance reconstruction quality, bit accuracy, semantic consistency, and perturbation budget.

There are two training strategies in the repo:

- `training/train.py`
  - Broad-purpose trainer.
  - Now memory-safe by default with resize to `128x128`, `BATCH_SIZE=4`, gradient accumulation, AMP, and allocator hinting.

- `training/train_production.py`
  - Fine-tuning path for low-alpha deployment.
  - Uses `IMAGE_SIZE=128` and alpha focus around `0.020-0.030`.
  - Better aligned with current inference constraints than `train.py`.

### 2. Inference path

`inference/inference.py` is the main operational path.

Embed flow:

1. Load image at full resolution.
2. Generate deterministic watermark from seed.
3. Clamp alpha via `core/reliability.py` when available.
4. Resize working copy to model resolution (`128x128`).
5. Embed watermark with encoder.
6. Optionally apply semantic layer.
7. Upscale only the watermark delta back to full resolution.
8. Optionally apply perceptual or HVS masking.
9. Save as lossless output and optionally post-verify extraction.

Decode flow:

1. Load image at full resolution.
2. Resize to model resolution.
3. Decode with pixel decoder or semantic decoder.
4. Compare against expected seed when available.

### 3. Ownership and audit path

`core/seed_registry.py` stores:

- seed-to-owner mappings
- metadata
- audit events
- image counts

This is used by the GUI and CLI to tie embedded watermarks to owners.

### 4. Reliability layer

`core/reliability.py` is a cross-cutting protection layer:

- path normalization to lossless output
- alpha floor and ceiling
- quantization simulation
- post-save verification
- redundant bit coding helpers
- BER utilities and logging

## What Was Verified

### Static verification

- All Python files compile successfully with `py_compile`.
- The edited notebooks still parse as valid JSON.

### Test verification

- `pytest -q` result: `32 passed, 1 xfailed`

Meaning:

- Current committed tests are passing.
- One regression test intentionally marks a known unresolved alpha-range mismatch as expected failure.

## Fixes Applied During This Scan

### Earlier memory fix already in place

- `training/train.py`
  - resized training images before batching
  - enabled gradient accumulation
  - enabled AMP
  - reduced default batch size to a safer value
  - added CUDA allocator hint

- Colab helpers and notebooks were updated to match the real training path.

### Additional repo fixes from this full scan

- `inference/inference.py`
  - batch directory embedding now follows the same alpha-clamping and adaptive-alpha behavior as single-image embedding.
  - this removes single-vs-batch behavior drift.

- `core/registry_cli.py`
  - fixed wrong owner email field lookup.
  - fixed broken `src/` path assumption for batch processing.
  - aligned default batch alpha from `0.10` to `0.035`.

- `tools/test_robustness.py`
  - added fallback to repo `test/` images when the old sample-image directory does not exist.

- `training/train_production.py`
  - added import fallback so it works both from the repo layout and from the Colab copied-file layout.

## Key Findings

### 1. Main architectural risk: alpha-range drift still exists

This is the biggest remaining design issue.

- `training/train.py` still trains across `0.3 -> 0.1`.
- inference and reliability clamp operational alpha to `0.020 -> 0.055`.
- `training/train_production.py` is much closer to deployment needs.
- the regression suite explicitly documents this as a known mismatch and keeps it `xfail`.

Implication:

- the repo has two different truths:
  - a legacy broad trainer
  - a deployment-aligned fine-tuner

Recommendation:

- choose one canonical training path.
- if low-alpha production is the target, promote `train_production.py` concepts into the canonical trainer or clearly mark `train.py` as legacy.

### 2. Batch and single-image inference had drift

This was real and was fixed.

Before fix:

- single-image embed used reliability alpha clamping and adaptive/HVS-aware path
- batch embed did not fully mirror that behavior

After fix:

- batch embed now uses the same safer alpha handling path

### 3. Tooling had layout assumptions that did not match this repo

This was also real and was fixed.

Examples:

- `registry_cli.py` assumed a non-existent `src/` directory
- `test_robustness.py` assumed a non-existent sample image directory

### 4. Some surfaces are still placeholders or lightly implemented

Notable examples:

- `training/semantic_watermark.py` explicitly describes itself as a lightweight placeholder.
- `core/seed_registry.py::verify_record_integrity()` always returns `True`; integrity verification is not fully implemented.
- `batch_checker` integration in `inference/inference.py` is optional and may not exist.
- several root docs or scripts are empty placeholders:
  - `AUTOTUNING_FIXES.md`
  - `FINE_TUNE_INSTRUCTIONS.md`
  - `QUICK_RESUME.md`
  - `inspect_system.py`
  - `plan.md`
  - `test_autotuning_fixes.py`
  - `test_end_to_end.py`

### 5. External dependency risk in seed registration

`core/seed_registry.py` attempts IP-based geolocation during registration.

Implication:

- registration can slow down or degrade in offline or restricted-network environments.
- location data quality is environment-dependent.

Recommendation:

- make geolocation explicitly optional or lazy.

### 6. Test coverage is strongest around helpers, weaker around end-to-end behavior

Covered reasonably:

- reliability primitives
- alpha clamping rules

Less covered:

- full training loop behavior
- checkpoint loading across all variants
- GUI workflows
- batch registry flow
- embed/decode round-trip with real checkpoint in automated tests

## Operational Understanding

### If you want to train

Use:

- `training/train.py` for general experimentation
- `training/train_production.py` for deployment-oriented low-alpha fine-tuning

Important:

- if deployment alpha really is `0.020-0.055`, `train_production.py` is closer to target behavior than the legacy trainer.

### If you want to embed or decode

Use:

- `inference/inference.py` for CLI
- `inference/app.py` for GUI

Most important runtime constraints:

- model resolution is `128x128`
- outputs are meant to preserve original resolution by upscaling the learned delta
- lossless save formats are preferred

### If you want ownership tracking

Use:

- `core/seed_registry.py`
- `core/registry_cli.py`
- GUI owner registration in `inference/app.py`

## Recommended Next Steps

### High priority

1. Decide the canonical alpha policy for the repo.
2. Either:
   - align `training/train.py` to deployment alpha behavior, or
   - clearly mark it as legacy and standardize on `train_production.py`.
3. Add at least one real end-to-end inference round-trip test using the checkpoint.

### Medium priority

1. Implement real integrity checking in `SeedRegistry.verify_record_integrity()`.
2. Make IP-based geolocation opt-in.
3. Consolidate duplicated training logic between notebook helpers, `train.py`, and `train_production.py`.

### Low priority

1. Replace placeholder docs and empty scripts with real content or remove them.
2. Add GUI smoke tests where practical.

## Current Confidence Summary

### Strong confidence

- helper modules compile
- current test suite passes
- batch CLI and robustness tool path bugs identified in this scan are fixed
- batch embedding behavior is now closer to single-image embedding behavior

### Moderate confidence

- primary CLI and training code paths are understandable and internally coherent
- the memory/OOM path is materially safer than before

### Limited confidence

- no full GPU training run was executed in this environment
- GUI flows were inspected but not manually driven end-to-end
- no full checkpoint embed/decode regression test was added in this pass

## Bottom Line

The codebase is viable and much more internally consistent after this scan, but it still has one meaningful unresolved architecture decision:

- the repo has not fully unified training alpha assumptions with deployment alpha assumptions.

If that is addressed next, the rest of the repo becomes much easier to reason about and maintain.
