# HierAdaptMR — Cross-Center Cardiac MRI Reconstruction with Hierarchical Feature Adapters

PyTorch implementation developed for the **CMRxRecon2025** multi-center cardiac MRI reconstruction challenge. This repository extends the **PromptMR** prompt-learning unrolled reconstruction network (Zhao et al., arXiv:2309.13839) and the **fastMRI** data/tooling ecosystem into a hierarchical, multi-center, multi-contrast, pathology-aware reconstruction framework.

---

## 1. The Problem

Cardiac magnetic resonance (CMR) imaging is acquired under severe throughput constraints, motivating k-space undersampling; reconstruction from accelerated acquisitions is an ill-posed inverse problem whose solution depends on coil geometry, calibration signal (ACS) layout, trajectory design, contrast protocol, and scanner vendor. Deep-learning reconstructions trained at a single site routinely **fail to generalize across centers** because these acquisition conditions shift between scanners (UIH vs. Siemens vs. Philips vs. GE), field strengths, and pulse sequences (Cine, LGE, mapping, perfusion, flow, black-blood, …).

The CMRxRecon2025 challenge formalizes this setting: multi-coil, multi-center, multi-contrast 3D cardiac k-space volumes must be reconstructed from undersampled masks at acceleration factors in {8, 16, 24}, evaluated against fully-sampled references with SSIM. Three properties make the problem hard:

1. **Domain heterogeneity.** Six-plus centers with different vendors/scanners produce systematically different image statistics, noise textures, and artifact patterns from the same nominal sequence.
2. **Trajectory diversity.** Sampling is not restricted to Cartesian lines — radial and variable-density-Gaussian k-t trajectories appear, each requiring a different auto-calibration region for sensitivity estimation.
3. **Pathology confounding.** Hypertrophic cardiomyopathy (HCM) cases exhibit wall-morphometry outliers; a model that "corrects" such frames based on a healthy-population prior can cause clinically harmful distortions, so pathology adaptation must be *gated*, not unconditional.

A single monolithic network either underfits minority domains or requires retraining per site. The goal of this repository is therefore: **keep one shared reconstruction backbone and learn minimal, safe, metadata-conditioned corrections** that recover cross-center performance without destabilizing the base model.

---

## 2. What the Original Code Already Solved

Two upstream bodies of work form the foundation, and their original contributions are preserved essentially intact:

### 2.1 PromptMR / PromptMR-plus backbone (`mri_network/promptmrplusV2.py`)

- **Variational unrolled optimization.** `PromptMR` stacks `num_cascades` `PromptMRBlock`s; each cascade alternates (i) a denoising prior applied to the current image estimate and (ii) a data-consistency step that enforces agreement with the measured (masked) k-space. This converts a learned network into an approximate solver of the reconstruction inverse problem rather than a black-box regressor.
- **Prompt-based parameter efficiency.** Each UNet level injects small learnable prompt tokens (`prompt_dim`, `len_prompt`, `prompt_length`) instead of scaling the whole denoiser, giving strong reconstruction quality with modest parameter counts.
- **Channel-attention blocks (CAB/CALayer)** for feature recalibration inside the encoder/decoder stages.
- **Learning-based sensitivity-map estimation (SME).** Coil-combination weights are produced by a dedicated PromptUnet branch operating on ACS-extracted low-frequency k-space, avoiding fixed sum-of-squares combination errors.
- **Adjacent-slice (k-t) input stacking.** Multi-slice context is concatenated as extra real channels (`num_adj_slices`), exploiting through-plane redundancy of cardiac volumes.

### 2.2 fastMRI-derived data and training infrastructure

- **`MaskFunc` abstraction (`data_loading/subsample.py`).** Deterministic, seed-controlled undersampling with guaranteed fully-sampled central lines (ACS), supporting Cartesian equispaced/random/Gaussian families.
- **HDF5 dataset machinery (`data_loading/mri_data.py`).** Multi-coil complex k-space stored as interleaved-real tensors, volume/slice indexing, mask application, RSS target images.
- **Standard training loop.** Adam optimizer, StepLR decay, SSIM loss (`mri_utils.losses.SSIMLoss`), validation-SSIM checkpointing, TensorBoard logging.
- **Distributed-sampler utilities** and Lightning-style `DataModule` configuration plumbing.

In short, the originals solved **single-domain, high-quality unrolled reconstruction**: given one center, one contrast family, and a Cartesian mask policy, reconstruct accurately and efficiently. They did **not** address domain shift, trajectory-dependent calibration, pathology safety, or curriculum control over heterogeneous subpopulations.

---

## 3. New Contributions in Detail

### 3.1 Hierarchical feature-adapter module (new; `mri_network/multi_center_adapter.py`)

`MultiCenterAdaptivePromptMR` wraps the backbone **without modifying its weights** and applies residual, image-domain corrections conditioned on acquisition metadata.

**(a) Three parallel adapter banks** (`nn.ModuleDict`):
- *Center adapters* — one `FeatureAdapter` per participating center (`Center001`–`Center003`, `Center005`–`Center007`), annotated with vendor/scanner identity (e.g., `UIH_30T_umr780`, `Siemens_30T_Prisma`).
- *Contrast adapters* — nine adapters covering the full CMRxRecon taxonomy: Cine, LGE, Mapping, T1w, T2w, Perfusion, T1rho, Flow2d, BlackBlood.
- *Pathology adapters* — one `HCMPathologyAdapter` (§3.2).

**(b) Bounded residual topology.** Each `FeatureAdapter` is a bottleneck Conv–BN–ReLU ×2 → Conv → Tanh stack (1→32→32→1 channels, 3×3 kernels). The correction is
`x ← x + clamp(α, −0.1, 0.1) · f(x)` with α initialized to 0.01, guaranteeing that at initialization the adapter perturbs the backbone output by at most 1 % of the bounded residual magnitude. Batch normalization and the tanh gate were added specifically to stabilize joint training with the backbone relative to a naive residual block.

**(c) Canonicalization for heterogeneous FOVs/intensities.** Adaptations run in a normalized space: per-sample instance normalization (`norm`/`unnorm`, std clamped at 1e-8) followed by padding H/W to multiples of 8 (`pad`/`unpad`), mirroring `NormPromptUnet`. Adapters see resolution-compatible, scale-invariant inputs while the outer loop preserves original image statistics and geometry.

**(d) Metadata-driven routing.** `extract_metadata_from_filenames` parses `Center\d+` via regular expression and matches the contrast token directly from the HDF5 filename (format `Center001_UIH_30T_umr780_Cine_P001_cine_lax_3ch.h5`). Per batch element: center adapter first, then contrast adapter (sequential composition of residuals), then the pathology adapter if gated. Unrecognized metadata silently bypasses the corresponding bank — i.e., graceful degradation to the backbone.

**(e) Failure containment.** NaN/Inf checks before and after normalization, adaptation, unpadding, and unnormalization; any detected anomaly reverts that sample's contribution to pre-adaptation features. A final shape check bilinearly re-interpolates drifted outputs. These guards were introduced after observing divergence modes when several center adapters were trained jointly with the backbone.

### 3.2 HCM pathology adapter with Mahalanobis-distance gating (new)

A distributionally-safe extension specialized to hypertrophic cardiomyopathy:

- `HCMPathologyAdapter` mirrors the `FeatureAdapter` topology but exposes the intermediate 32-channel ReLU feature tensor (`return_features=True`) for statistical analysis.
- `HCMMahalanobisGating` collects spatially-average-pooled adapter features during training, restricted to **end-diastolic frames** (temporal index recovered per sample as `slice_num // num_slc`; only `temporal_idx == 0` contributes; collection toggled by `enable_hcm_collection`, enabled in `train_hcm_adapter.py`). At each epoch end, `fit()` estimates mean μ and covariance Σ with ε·I ridge regularization (ε = 1e-6), computes the precision matrix via `torch.linalg.inv`, and sets an acceptance threshold from the χ² distribution, df = 32 at the 95th percentile (`scipy.stats.chi2.ppf`).
- **Inference-time gating.** With squared Mahalanobis distance `d² = (φ − μ)ᵀ Σ⁻¹ (φ − μ)`:
  `output = m ⊙ HCM(x) + (1 − m) ⊙ x`, `m = 1[d² < τ]`.
  The pathology adapter fires only where its internal features lie inside the calibrated in-distribution manifold; out-of-distribution frames fall back to the center/contrast-adapted reconstruction, preventing unsupported pathological "corrections."
- Gating statistics are serialized inside checkpoints (`hcm_gating` field) and separately as `hcm_gating_stats.pt`, restored with `strict=False` loading for inter-stage transfer.

### 3.3 Backbone modifications (`mri_network/promptmrplusV2.py`)

While the cascade structure is inherited, three changes adapt it to multi-center data:

**(a) Mask-type-aware sensitivity estimation.** The original SME applies a fixed low-frequency retention policy. Here `SensitivityModel.forward` first calls `_identify_mask_type(mask)`, which inspects the central ±10 k-space lines: fully sampled center lines classify the trajectory as Cartesian (`ktUniform`/`ktGaussian`), otherwise radial (`ktRadial`). Cartesian masks follow the original `get_pad_and_num_low_freqs` → `batched_mask_center` path; radial masks use `_apply_radial_center_mask`, restricting the calibration window to ±10 samples in both phase directions. This makes ACS selection consistent with non-Cartesian k-t trajectories present in CMRxRecon2025.

**(b) Historical-feature aggregation across cascades.** `PromptUnet.forward` accepts/returns an optional `history_feat` list. When `n_history > 0`, each decoder level concatenates a rolling buffer of features from previous cascade iterations (tile-initialized at the first cascade), propagating iterative context through the unrolled network — absent from the original formulation.

**(c) Adaptive input and widened capacity.** `adaptive_input=True` with ring buffer `n_buffer=4` adapts adjacent-slice stacking (`num_adj_slices=5`, i.e., 2×5 = 10 real input planes). Trained configurations widen feature dimensions relative to paper defaults: `feature_dim=[72, 96, 120]`, `prompt_dim=[24, 48, 72]`, SME `sens_feature_dim=[36, 48, 60]`, `len_prompt=[5,5,5]`, prompt sizes `[64, 32, 16]`, CAB counts `[2,3,3]/[2,2,3]/[1,1,1]/3` (see `train_adapter.py::cli_main`).

### 3.4 Data-loading and sampling updates (`data_loading/`)

**(a) Challenge-consistent composite mask pool.** Fixed-low-frequency primitives (`FixedLowRandomMaskFunc`, `FixedLowEquiSpacedMaskFunc`, `FixedLowGaussianMaskFunc` with `alpha=0.28`, `sigma_factor=5.0`; `FixedLowRadialMaskFunc` with angular spacing 180°/step and corner cropping) are composed into **`CmrxRecon25MaskFunc`**, whose pool reproduces the protocol:

```python
mask_dict = {'uniform': [8,16,24], 'kt_radial': [8,16,24], 'kt_gaussian': [8,16,24]}
```

A mask family is drawn uniformly per call; k-t variants receive the slice offset, adjacent-slice window (symmetric ±2), and volume dimensions so trajectories stay consistent along the slice direction. `CmrxRecon25TestValMaskFunc` subclasses this for deterministic validation masks; retained central lines controlled by `--num_low_frequencies` (default 20).

**(b) Metadata-carrying datasets/transforms.** `CmrxReconSliceDataset` enumerates (volume, slice) pairs and gathers adjacent slices via `_get_ti_adj_idx_list` (edge-clipped); `CmrxReconInferenceSliceDataset` adds cached volume streaming (`_load_next_volume`). `CmrxReconDataTransform` emits a `PromptMRSample` NamedTuple carrying k-space, mask, RSS target, **filename, and slice number** — precisely the fields consumed by adapter routing (§3.1d) and ED-frame gating (§3.2).

**(c) Volume-consistent distributed sampling.** `VolumeSampler` / `InferVolumeDistributedSampler` keep all slices of one volume on the same device/rank (required because k-t adjacency and the history buffer are volume-local), with epoch-dependent shuffling. `data_module.py` resolves dataset/mask/transform classes dynamically from string paths in YAML configs.

**(d) Hierarchical stratified patient-level subsetting (new).** `uniform_stratified_sampling` replaces random frame subsampling: every file is parsed into **Center → Vendor/Scanner → Modality → Patient**, and patients (not slices) are sampled proportionally within each leaf, keeping whole subject studies together (no leakage, no over-representation of large centers). Two deviations encode empirical difficulty: `Center002` and `Center006/Siemens_30T_Prisma` always retain **all** patients. The subset is rebuilt every epoch with seed `args.seed + epoch·1000` (`create_stable_train_dataloader`) — stochastic regularization with internally reproducible epochs. Default ratio `--subset_ratio 0.3`.

**(e) Performance-aware curriculum (`finetun.py`, new).** On top of stratification: (1) `WeightedRandomSampler` over center-performance tiers — low performers (e.g., Center002) weight 3.0, medium 1.5, high 0.8; (2) differentiated learning rates in `setup_optimizer_with_performance_based_lr` — AdamW groups assign backbone `lr×0.01`, per-center adapters tier multipliers (0.01 high / 0.05 medium / 0.1 low), contrast adapters `lr×0.02`, with reduced weight decay (×0.1) on adapter groups.

### 3.5 Loss-function family (`mri_network/MultiScaleSSIMLoss.py`, new)

Beyond the plain windowed SSIM baseline:

- **`AdaptiveMultiScaleSSIMLoss`** — multi-scale SSIM over scales {1, 2, 4, 8} with *learnable* per-scale weights combined via softmax, average-pool downsampling, adaptive window `min(7, ⌊min(H,W)/4⌋·2+1)`, and percentile-based robust normalization (1st–99th quantile rescaling to [0, 1]) so outlier intensities do not dominate gradients.
- **`EnhancedSSIMOptimizedLoss`** — composes SSIM, MS-SSIM, a frequency-domain term (structural comparison of FFT magnitude spectra) and a Sobel-style edge-preserving term, with (i) epoch-annealed SSIM emphasis (`1 + 0.5·min(1, epoch/15)`), (ii) per-contrast weight profiles reflecting clinical priorities (e.g., LGE: ssim 1.3 / freq 0.4 / edge 0.3; Cine emphasizes temporal coherence), (iii) per-center loss boosts from validation diagnostics (e.g., Center002 ×1.3, Center006 ×0.9), and sigmoid-mapped z-score normalization. In the shipped scripts the default active objective remains `SSIMLoss`; the enhanced losses are retained as commented alternatives for ablation.

### 3.6 Training-engineering additions

Relative to the original loop: mixed precision (`GradScaler`) + gradient accumulation (`accumulation_steps=8`, partial batch flushed) + gradient clipping (`clip_grad_norm_`, max_norm 3) — necessary given batch size 1 per GPU for 3D multi-coil volumes; per-sample `normalize_kspace` and numerically stabilized root-sum-of-squares (`stable_rss`, eps 1e-8); early stopping (patience 6 epochs without SSIM improvement); dual-mode checkpoint loading (`--resume` restores best metrics for same-stage continuation; omitting it performs transfer initialization from a backbone checkpoint with metrics reset to 0.0/inf); checkpoints serialize backbone + all adapter banks + HCM gating stats; SLURM integration via `*.sbatch` (single-process and 4-GPU `torchrun`).

---

## 4. Repository Layout

```
.
├── train_adapter.py            # Stage 1: joint backbone + center/contrast/HCM adapter training
├── train_hcm_adapter.py        # Stage 2: frozen-backbone HCM adapter training w/ Mahalanobis collection
├── finetun.py                  # Stage 3: curriculum fine-tuning (tiered sampling + tiered LRs)
├── read_mat.py                 # .mat inspection utility (structure dump for raw CMRxRecon files)
├── UHeM_server.log             # Example cluster training log
├── train_no_distribution.sbatch# SLURM: single-process training submission
├── train_with_distribution.sbatch # SLURM: torchrun --nproc_per_node=4 distributed submission
│
├── mri_network/
│   ├── promptmrplusV2.py       # Backbone: PromptMR/PromptMR-plus V2 (PromptUnet, CAB, NormPromptUnet,
│   │                           #   PromptMRBlock cascade, SensitivityModel w/ mask-type awareness)
│   ├── multi_center_adapter.py # FeatureAdapter banks, MultiCenterAdaptivePromptMR wrapper,
│   │                           #   HCMPathologyAdapter, HCMMahalanobisGating, filename-metadata parsing
│   └── MultiScaleSSIMLoss.py   # AdaptiveMultiScaleSSIMLoss, EnhancedSSIMOptimizedLoss
│
├── data_loading/
│   ├── subsample.py            # MaskFunc primitives + CmrxRecon25MaskFunc / TestVal composite pools
│   ├── mri_data.py             # CmrxReconSliceDataset, inference dataset, adjacent-slice gathering
│   ├── transforms.py           # CmrxReconDataTransform -> PromptMRSample (carries filename/slice num)
│   ├── volume_sampler.py       # VolumeSampler / InferVolumeDistributedSampler (volume-local batches)
│   └── data_module.py          # Lightning DataModule with dynamic class resolution from YAML
│
└── data_preprocessing/
    ├── data_preprocessing.py   # Step 1: CMRxRecon .mat (MultiCoil FullSample) -> HDF5
    ├── split_data.py           # Step 2: center-based train/val JSON splits (uses creat_json.create_center_based_split)
    ├── analys_kspace.py        # K-space geometry/statistics analysis
    ├── print_kspace_shape.py / original_kspace_shape.log  # Shape audit tooling
    └── checking_val_result_shape.py                       # Validation-output geometry checker
```

Note: `mri_utils/` (losses such as `SSIMLoss`, math helpers) is imported as an external companion package from the upstream PromptMR codebase and is expected on `PYTHONPATH`.

---

## 5. Setup

**Requirements**
- Python ≥ 3.9, CUDA-enabled PyTorch (≥ 1.13 recommended; AMP APIs used), `numpy`, `h5py`, `scipy` (χ² thresholds), `tqdm`, `tensorboard`, `pytorch-lightning` (for `DataModule`), `fftc` (complex FFT/RSS utilities used by preprocessing).
- Hardware: ≥ 1 GPU with ≥ 24 GB memory (training uses batch size 1 per GPU for 3D multi-coil volumes); the distributed recipe targets 4×A100.

**Installation**

```bash
conda create -n cmr2025 python=3.10 -y && conda activate cmr2025
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy h5py scipy tqdm tensorboard pytorch-lightning fftc
git clone <this-repo> && cd HierAdaptMR
```

**Data preparation**

1. Download the CMRxRecon2025 challenge data (`ChallengeData/MultiCoil/TrainingSet/FullSample/**/*.mat`).
2. Convert MATLAB volumes to HDF5:
   ```bash
   python data_preprocessing/data_preprocessing.py \
       --input_matlab_folder /path/to/ChallengeData/MultiCoil \
       --output_h5_folder    /path/to/preprocess
   ```
3. Create center-based train/val splits (JSON):
   ```bash
   python data_preprocessing/split_data.py --output_h5_folder /path/to/preprocess
   ```
   Resulting filenames must preserve the metadata encoding `CenterNNN_<Vendor>_<Field>_<Scanner>_<Contrast>_P###_...h5`, since adapter routing and stratification parse them.
4. Point the training scripts at the processed root (defaults are hardcoded absolute paths; override with `--data_path`, `--experiments_output`, `--pretrained`).

**Pretrained backbone.** Joint training expects a backbone-only checkpoint (PromptMR V2 pretraining) passed via `--pretrained`; without `--resume` it is loaded as transfer initialization.

---

## 6. Training

Three staged entry points share a common harness but differ in trainability scope:

| Script | Backbone | Center/Contrast adapters | HCM adapter | Optimizer |
|---|---|---|---|---|
| `train_adapter.py` | trainable at `lr` (2e-4) | trainable at `lr×0.1` | trainable at `lr×0.1` | AdamW, StepLR(step 3, γ 0.9), wd 1e-4 |
| `train_hcm_adapter.py` | **frozen** (`requires_grad=False`) | frozen | **only** trainable group, `lr` | AdamW, StepLR |
| `finetun.py` | `lr×0.01` (base lr 1.5e-4) | tiered multipliers (§3.4e) | — | AdamW param groups, eps 1e-8 |

**Stage 1 — joint adapter training**

```bash
python train_adapter.py \
    --data_path /path/to/preprocess1 \
    --pretrained /path/to/backbone/checkpoint_epoch_0.pth.tar \
    --experiments_output ./output/stage1 \
    --max_epochs 20 --lr 2e-4 --subset_ratio 0.3 --batch_size 1 --seed 42
```
Each epoch rebuilds the stratified subset (seed `seed + epoch·1000`), trains with AMP + accumulation (8) + clipping (3), validates with SSIM, logs to TensorBoard, saves `checkpoint.pth.tar` (+ `best_ssim` copy), and stops early after 6 epochs without SSIM improvement.

**Stage 2 — HCM adapter with gating calibration**

```bash
python train_hcm_adapter.py \
    --data_path /path/to/preprocess1 \
    --pretrained ./output/stage1/best_ssim_checkpoint.pth.tar \
    --experiments_output ./output/stage2
```
Backbone and center/contrast banks are frozen; only `HCMPathologyAdapter` receives gradients. ED-frame features are collected throughout the epoch and `HCMMahalanobisGating.fit()` refits μ, Σ, and the χ²(df=32, p=0.95) threshold at each epoch end; statistics persist in the checkpoint and as `hcm_gating_stats.pt`.

**Stage 3 — performance-aware fine-tuning**

```bash
python finetun.py \
    --data_path /path/to/preprocess1 \
    --pretrained ./output/stage2/checkpoint.pth.tar \
    --experiments_output ./output/stage3 --lr 1.5e-4
```
Enables weighted-sampling tiers (3.0 / 1.5 / 0.8) and differentiated per-group learning rates, biasing optimization toward underperforming centers while protecting the backbone.

**Resume vs. transfer semantics.** `--resume` restores optimizer-independent state (best_SSIM / best_val_loss) for continuing the *same* stage; omitting it treats `--pretrained` as a foreign initialization and resets tracking metrics (0.0 / inf).

**Cluster deployment**

```bash
sbatch train_no_distribution.sbatch      # single-process
sbatch train_with_distribution.sbatch    # torchrun --nproc_per_node=4 (distributed; volume-local samplers)
```

---

## Citation & Acknowledgements

Built upon **PromptMR** (Zhao et al., *PromptMR: Prompt Learning for Undersampled Cardiac Magnetic Resonance Imaging*, arXiv:2309.13839) and the **fastMRI** tooling (Zbontar et al., 2018), using data of the **CMRxRecon / CMRxRecon2025** challenge. All adapter, gating, curriculum, and loss innovations above are original to this repository.
