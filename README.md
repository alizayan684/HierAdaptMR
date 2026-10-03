# HierAdaptMR — Cross-Center Cardiac MRI Reconstruction with Hierarchical Feature Adapters

A PyTorch implementation accompanying **[HierAdaptMR: Cross-Center Cardiac MRI Reconstruction with Hierarchical Feature Adapters](https://arxiv.org/abs/2508.13026)** (Xu & Oksuz, *StatXL 2025*), developed for the **CMRxRecon2025** multi-center cardiac MRI (CMR) reconstruction challenge.

---

## 1. Problem Statement

Deep-learning-based accelerated magnetic resonance imaging (MRI) reconstruction models are typically trained and evaluated on data acquired at a *single* imaging center, using *one* scanner vendor, field strength, pulse sequence, and contrast protocol. When such a model is deployed across multiple centers, its reconstruction quality degrades markedly because the image statistics shift with every link of the acquisition chain:

- **Center / site effects** — different scanners, coils, and reconstruction pipelines produce systematically different k-space and image-domain characteristics.
- **Contrast / modality effects** — Cine, LGE, T1/T2-weighted, mapping, perfusion, flow, black-blood and T1ρ sequences differ in tissue contrast, resolution, and temporal structure.
- **Pathology effects** — diseased anatomy (e.g., hypertrophic cardiomyopathy, HCM, with asymmetric septal hypertrophy) is under-represented in healthy-subject training data, so generic reconstructions may distort precisely the structures clinicians need for diagnosis.

The repository addresses the question: **how can a single shared reconstruction network be adapted to an entire center → vendor → modality hierarchy, plus pathology-specific corrections, without retraining or duplicating the backbone?** The answer implemented here is a *hierarchical feature-adapter* scheme built on top of a frozen prompt-learning-based unrolled reconstruction network, together with a distribution-aware gating mechanism that decides *when* a pathology adapter should be trusted at inference time.

---

## 2. What the Original Code Already Solved

The starting point of this work is the **PromptMR-plus** architecture (`mri_network/promptmrplusV2.py`, 783 lines), a prompt-learning-based unrolled model for multi-coil MR reconstruction (cf. arXiv:2309.13839), combined with the standard fastMRI-style data infrastructure. The original implementation provides:

### 2.1 The PromptMR-plus backbone
- **`PromptUnet` / `NormPromptUnet`** — a U-Net denoiser augmented with *learnable prompts*: compact token tensors (`PromptBlock`) injected into the encoder/decoder paths, allowing one network to specialize per-cascade behavior with negligible parameter overhead. Channel-attention blocks (`CAB`/`CALayer`) refine features along the channel dimension.
- **`SensitivityModel`** — a dedicated PromptUnet variant that estimates coil sensitivity maps from the auto-calibration region of k-space (with root-sum-of-squares normalization, mask-type identification, and radial-center masking handling).
- **`PromptMR`** — the full unrolled cascade (`num_cascades=12`) interleaving data-consistency steps (`PromptMRBlock`, with coil combine via `sens_reduce` and expansion via `sens_expand`) and prompt-conditioned priors; it supports **multi-slice k–t processing** (`num_adj_slices=5`) and an **adaptive-input mode** with slice buffers (`n_buffer=4`) and historical feature aggregation (`n_history=11`) for through-plane/temporal context.
- Per-sample normalization, padding to multiples of 8, and unpadding/unnormalization utilities (`NormPromptUnet.norm/unnorm/pad/unpad`).

### 2.2 Data pipeline for CMRxRecon2025
- **`data_preprocessing/data_preprocessing.py`** — converts raw MATLAB `.mat` multi-coil k-space volumes from the challenge into complex-valued HDF5 (`.h5`) tensors (real/imag stacked on the last channel dimension); `split_data.py` produces train/val splits. Helper scripts (`analys_kspace.py`, `print_kspace_shape.py`, `read_mat.py`) inspect k-space shapes and contents of the heterogeneous acquisitions.
- **`data_loading/subsample.py`** — `CmrxRecon25MaskFunc` implements the challenge variable-density random under-sampling masks (fixed central low-frequency block + variable high-frequency lines), seeded deterministically per volume so training and validation masks are reproducible.
- **`data_loading/mri_data.py`** — `CmrxReconSliceDataset` yields `(masked_kspace, mask, target, fname, slice_num, num_slc, max_value)` samples with adjacent-slice grouping (`_get_ti_adj_idx_list`); `CmrxReconInferenceSliceDataset` adds volume-level streaming with look-ahead prefetching for inference; a Calgary–Campinas dataset is also supported.
- **`data_loading/transforms.py`**, **`data_module.py`**, **`volume_sampler.py`** — k-space transforms, Lightning-style data-module wiring, and volume-wise sampling for distributed runs.

### 2.3 Baseline training machinery
The original single-center training loop established the recipe reused throughout: AdamW optimization with `StepLR` decay, mixed-precision (`GradScaler` + `autocast`), gradient accumulation (8 steps) with gradient-norm clipping (max_norm=3), SSIM-based loss (`fastmri`/`mri_utils.losses.SSIMLoss` with per-sample `data_range`), best-model checkpointing by validation SSIM, early stopping, and TensorBoard logging of scalars and example reconstructions.

**Limitation motivating this work:** this baseline is a *monolithic* model. One set of weights must implicitly average over all centers, vendors, contrasts, and pathologies present in the multi-center challenge data — leading to suboptimal per-site quality, catastrophic forgetting risk when fine-tuning on new sites, and no mechanism whatsoever for pathology-aware reconstruction.

---

## 3. New Contributions in Detail

All new contributions live in **`mri_network/multi_center_adapter.py`** (the adaptive wrapper) and the three training drivers (**`train_adapter.py`**, **`train_hcm_adapter.py`**, **`finetun.py`**). They are described below as implemented in the code.

### 3.1 Hierarchical residual feature adapters (`FeatureAdapter`)

Instead of retraining the backbone, lightweight conditional correction modules are attached *after* the reconstruction output. Each `FeatureAdapter` is a small residual convolutional block:

```
Conv2d(1→32, 3×3) → BN → ReLU → Conv2d(32→32, 3×3) → BN → ReLU → Conv2d(32→1, 3×3) → Tanh
```

and applies the gated residual update

$$x' = x + \operatorname{clamp}(\alpha,\,-0.1,\,0.1)\cdot \text{Tanh}\big(f_\theta(x)\big),\qquad \alpha_{\text{init}} = 0.01 .$$

Design choices, each aimed at *stability of adaptation*:
- **BatchNorm inside the adapter** stabilizes optimization on tiny per-center datasets.
- **`Tanh` output** bounds the residual shape of the correction.
- **Clamped learnable scalar `α ∈ [-0.1, 0.1]`, initialized at 0.01** guarantees the adapter starts as a near-identity map and can only ever *nudge* — never override — the backbone prediction. This is the key defense against domain-adaptation-induced collapse of the base reconstruction.

### 3.2 Two-level metadata routing: center and contrast adapters

`MultiCenterAdaptivePromptMR` maintains two registries of adapters instantiated as `nn.ModuleDict`s:

- **Six center adapters** — `Center001` (UIH 3T umr780), `Center002` (Siemens 3T CIMA), `Center003` (UIH 3T umr880), `Center005` (mixed scanners), `Center006` (Siemens 3T Prisma), `Center007` (Siemens mixed).
- **Nine contrast adapters** — `Cine`, `LGE`, `Mapping`, `T1w`, `T2w`, `Perfusion`, `T1rho`, `Flow2d`, `BlackBlood`.

Routing is fully automatic and requires no side-channel metadata files: `extract_metadata_from_filenames()` parses the challenge filename convention (e.g. `Center001_UIH_30T_umr780_Cine_P001_cine_lax_3ch.h5`) with a regex for `Center\d+` and a keyword scan for the contrast type. At forward time the adaptations are applied **sequentially and compositionally** per batch element:

$$x \xrightarrow{\;\text{center adapter}\;} x \xrightarrow{\;\text{contrast adapter}\;} x \xrightarrow{\;\text{pathology adapter (gated)}\;} \hat{x}$$

so the effective specialization space is the Cartesian product of ~6 centers × 9 contrasts while only 6 + 9 + 1 small adapter modules exist. Unknown center/contrast strings fall through to the identity (no adapter applied), giving graceful degradation on unseen sites.

### 3.3 UNet-style normalized adaptation envelope

Because adapters operate on reconstructed magnitude images whose intensity scale varies wildly across sequences, every adaptation pass mirrors the `NormPromptUnet` contract of the backbone:

1. **Per-sample normalization** — instance-standardize each `[B, H, W]` slice (std clamped ≥ 1e-8);
2. **Padding** — pad H/W up to the next multiple of 8 (matching the backbone's downsampling depth);
3. **Adaptation** — apply the routed adapter chain;
4. **Unpadding** — crop back with exact recorded pad offsets (with a robust centered-crop fallback);
5. **De-normalization** — restore the original mean/std.

This makes the adapter functions **scale-equivariant**: they learn geometric/intensity *corrections*, not absolute brightness offsets, which is what allows a single contrast adapter to transfer across patients with different dynamic ranges.

### 3.4 NaN/Inf-safe adaptation with guaranteed fallback

Every stage of the pipeline is instrumented with numerical guards: NaN/Inf checks on the input features, after normalization, after each individual adapter application, and on the final adapted output; try/except handlers around norm/pad/adapt/unpad/unnorm; a shape-consistency check with bilinear-interpolation recovery. On any failure the module **returns the unadapted backbone reconstruction** rather than corrupting it. In practice this turned out to be essential when training cascaded adapters on mixed-quality multi-center data with occasional degenerate slices.

### 3.5 Pathology adapter for HCM (`HCMPathologyAdapter`)

A third, disease-level tier targets **hypertrophic cardiomyopathy (HCM)**. Unlike the opaque `nn.Sequential` of `FeatureAdapter`, `HCMPathologyAdapter` is written as explicit layers (`conv1/bn1/relu1/conv2/bn2/relu2/conv3/tanh`) precisely so that its **intermediate 32-channel feature map** (output of the second ReLU) can be exposed via `forward(x, return_features=True)`. It uses the same clamped-α residual formulation. During training the HCM adapter is always applied; at inference its application is *conditional* — see §3.6.

### 3.6 Mahalanobis-distribution gating (`HCMMahalanobisGating`) — the core novelty

Naively applying a pathology adapter to *all* inputs would damage healthy cases (a false-positive correction). The repository implements a **statistical gate that activates the HCM adapter only on slices whose adapter-space features lie within the learned "HCM-like" distribution**:

- **Training-time collection.** With `enable_hcm_collection = True` (set by `train_hcm_adapter.py`), the pooled intermediate features of the HCM adapter — spatially averaged via `adaptive_avg_pool2d(features, 1)` to a 32-dimensional vector — are buffered per sample. Collection is restricted to the **end-diastole (ED) frame**: the per-sample temporal index is derived in `forward()` as `slice_num // num_slc`, and only frames with `temporal_idx == 0` contribute, avoiding duplicate/near-duplicate statistics from the same cine series.
- **Distribution fitting.** `fit()` computes the class-mean μ and the sample covariance Σ (with ε = 1e-6 ridge regularization) over the collected vectors and stores the **precision matrix** Σ⁻¹. The decision threshold is set from the χ² distribution: `threshold = chi2.ppf(0.95, df=32)` (scipy), i.e., the 95th percentile of the squared-Mahalanobis distance under the fitted Gaussian hypothesis.
- **Inference gating.** For a test slice, the squared Mahalanobis distance

$$D^2(\mathbf{z}) = (\mathbf{z}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{z}-\boldsymbol{\mu})$$

is computed from the pooled features; the output is a hard convex blend

$$\hat{x} = \mathbb{1}[D^2 < \tau]\cdot x_{\text{HCM-adapted}} \;+\; \mathbb{1}[D^2 \ge \tau]\cdot x_{\text{center/contrast-only}} .$$

If the gate has not been fitted yet, the HCM adapter passes through unchanged (safe default). The fitted statistics (μ, Σ⁻¹, τ) are serialized both inside epoch checkpoints (`'hcm_gating'` key) and separately as `hcm_gating_stats.pt`, and are restored automatically by all training scripts — so the gate survives save/resume cycles and can ship with the final model.

This gives the system a principled, calibration-free anomaly-detection component: the pathology expert is consulted only when the *model's own internal representation* says the slice looks pathological, implementing a mixture-of-experts routing rule that is estimated from feature-space geometry rather than from external HCM labels.

### 3.7 Hierarchical stratified subset sampling

Training on the full multi-center corpus is expensive and imbalanced (some centers contribute far more patients). `uniform_stratified_sampling()` builds the full **Center → Vendor → Modality → Patient** tree by parsing filenames, then draws subsets *uniformly across the hierarchy* so that each stratum is represented proportionally to strata count rather than to raw slice count — large centers cannot dominate the gradient. Patients in data-scarce strata (`Center002`, `Center006_Siemens_30T_Prisma`) are retained in full. Default configuration: `--use_subset True --subset_ratio 0.3`. A fresh dataloader is rebuilt each epoch (`create_stable_train_dataloader`) with epoch-dependent seeding for exposure diversity while keeping mask seeds deterministic.

### 3.8 Three-stage training curriculum (three driver scripts)

The repository operationalizes adaptation as a staged curriculum over which parameters are trainable:

| Script | Trainable parameters | Optimizer groups | Purpose |
|---|---|---|---|
| `train_adapter.py` | backbone + all adapters | base: `lr`; center/contrast/pathology adapters: `lr × 0.1` | Joint warm-up of adapters with gentle backbone updating (default `lr = 2e-4`) |
| `train_hcm_adapter.py` | **only `pathology_adapters`** (everything else frozen with `requires_grad=False`) | single group at `lr` | Learn the HCM expert + collect Mahalanobis statistics on top of converged center/contrast adapters |
| `finetun.py` | backbone + all adapters | **performance-based per-center learning rates** (base: `lr × 0.01`; high-performing centers `×0.01`, medium `×0.05`, low-performing `×0.1`, contrast adapters `×0.02`; reduced weight decay on adapters) | Final joint fine-tuning that lets struggling centers move faster while protecting already-good ones (default `lr = 1.5e-4`) |

Additional engineering refinements across all three scripts:
- **Transfer vs. resume semantics** (`--resume` flag): loading a *backbone* checkpoint resets the best-metric thresholds (fresh stage), whereas resuming the *same stage* restores `best_SSIM`/`best_val_loss` so earlier best checkpoints are not overwritten by worse epochs.
- **Best-SSIM checkpointing** (`best_ssim_model.pth.tar`) keyed on validation SSIM (= 1 − SSIMLoss with per-sample `data_range`), early stopping with patience 6, AMP + gradient accumulation (8) + clipping (norm ≤ 3).
- **Dataset statistics logging** (`log_dataset_statistics`) reporting per-center/per-contrast slice counts for reproducibility audits.
- Robust root-sum-of-squares and k-space normalization helpers (`stable_rss`, `normalize_kspace`) guarding against zero-variance coils.

### 3.9 Auxiliary loss research (`mri_network/MultiScaleSSIMLoss.py`)

Exploratory losses directly targeting the challenge metric are provided: `AdaptiveMultiScaleSSIMLoss` (learnable softmax-weighted multi-scale SSIM at pooling scales 1/2/4/8 with percentile-based robust normalization and adaptive Gaussian window sizing) and `EnhancedSSIMOptimizedLoss` (frequency-domain-augmented SSIM accepting contrast/center conditioning). These were evaluated during development; the released training loops use the stable plain SSIM loss by default, with the enhanced variants kept available behind commented call sites.

### 3.10 Summary of the delta over the original implementation

1. **+** Metadata-driven hierarchical adapter bank (6 center + 9 contrast modules) on a frozen reusable backbone.
2. **+** New pathology tier: HCM adapter with exposed intermediate features.
3. **+** Mahalanobis-distance gating with χ²-calibrated threshold, ED-frame-restricted feature collection, and checkpoint persistence — label-free routing of the pathology expert.
4. **+** Normalized/padded adaptation envelope making adapters scale-equivariant and shape-safe.
5. **+** Multi-layer NaN/Inf defenses with identity fallback at every stage.
6. **+** Hierarchical stratified subset sampling balancing the center/vendor/modality/patient tree.
7. **+** Three-stage curriculum (joint adapter training → frozen-backbone HCM training → performance-weighted fine-tuning) with differentiated parameter-group learning rates.
8. **+** Transfer-vs-resume checkpoint semantics, gating-stat serialization, SLURM/torchrun launch recipes.

---

## 4. Repository Layout

```
HierAdaptMR-main/
├── train_adapter.py                  # Stage 1: joint backbone + center/contrast/pathology adapter training
├── train_hcm_adapter.py              # Stage 2: frozen backbone, HCM-pathology-only training + Mahalanobis feature collection
├── finetun.py                        # Stage 3: performance-based per-center-LR joint fine-tuning
├── train_with_distribution.sbatch    # SLURM job: torchrun, 4×A100 distributed training
├── train_no_distribution.sbatch      # SLURM job: single-process training
├── read_mat.py                       # .mat/.h5 structural inspection utility
├── UHeM_server.log                   # Cluster environment setup notes (conda/pip commands used)
├── mri_network/
│   ├── promptmrplusV2.py             # [original] PromptMR-plus unrolled backbone (PromptUnet, SME, cascades)
│   ├── multi_center_adapter.py       # [NEW] FeatureAdapter, HCMPathologyAdapter, HCMMahalanobisGating,
│   │                                 #        MultiCenterAdaptivePromptMR wrapper
│   └── MultiScaleSSIMLoss.py         # [NEW] Adaptive multi-scale / frequency-enhanced SSIM losses (exploratory)
├── data_loading/
│   ├── mri_data.py                   # CmrxReconSliceDataset / inference dataset / Calgary–Campinas
│   ├── data_module.py                # Lightning-style data module wiring
│   ├── transforms.py                 # CmrxReconDataTransform (k-space preprocessing)
│   ├── subsample.py                  # CmrxRecon25MaskFunc under-sampling masks
│   └── volume_sampler.py             # Volume-wise sampling for distributed training
└── data_preprocessing/
    ├── data_preprocessing.py         # Raw .mat multi-coil k-space → .h5 conversion
    ├── split_data.py                 # Center-based train/val split generation
    ├── analys_kspace.py              # k-space analysis utilities
    ├── print_kspace_shape.py         # Acquisition shape auditing
    └── checking_val_result_shape.py  # Output-shape verification for challenge submissions
```

---

## 5. Setup

### Requirements
- Python 3.11, CUDA-capable NVIDIA GPU (development used A100 nodes; scripts run single-device, with a 4-GPU `torchrun` recipe provided).
- Packages: `torch`, `numpy`, `h5py`, `tqdm`, `tensorboard`, `fastmri`, `pyyaml`, `scipy`, `pytorch_msssim`, `fftc` (used by preprocessing).

```bash
conda create -n ruruCMR python=3.11
conda activate ruruCMR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install numpy h5py tqdm tensorboard fastmri pyyaml scipy pytorch_msssim
```

> Note: the training scripts import `from mri_utils.losses import SSIMLoss`; provide the `mri_utils` package on `PYTHONPATH` (it ships with the CMRxRecon challenge tooling) or substitute the local `mri_network.MultiScaleSSIMLoss` implementations.

### Data preparation

1. Download the **CMRxRecon2025** multi-coil dataset (`ChallengeData/MultiCoil/TrainingSet/FullSample/**/*.mat`).
2. Convert to HDF5:
   ```bash
   python data_preprocessing/data_preprocessing.py \
     --input_matlab_folder /path/to/ChallengeData/MultiCoil \
     --output_h5_folder /path/to/preprocess
   ```
3. Generate train/val splits:
   ```bash
   python data_preprocessing/split_data.py --output_h5_folder /path/to/preprocess
   ```

Filenames must preserve the challenge hierarchy, e.g. `Center001_UIH_30T_umr780_Cine_P001_cine_lax_3ch.h5`, which is parsed as `center / vendor / field-strength / scanner / modality / patient / sequence` and drives adapter routing and stratified sampling.

---

## 6. Training

The intended workflow is a **three-stage curriculum**; each stage consumes the previous stage's checkpoint via `--pretrained`.

### Stage 1 — Hierarchical adapter training (`train_adapter.py`)

Trains center + contrast (+ pathology) adapters jointly with light backbone updating (adapters at `0.1×` the base LR):

```bash
python train_adapter.py \
  --data_path /path/to/preprocessed \
  --experiments_output /path/to/out_stage1 \
  --pretrained /path/to/backbone_checkpoint.pth.tar \
  --batch_size 1 --max_epochs 20 --lr 2e-4
```

### Stage 2 — HCM pathology adapter (`train_hcm_adapter.py`)

Freezes everything except the HCM adapter, enables Mahalanobis feature collection (ED frames only), refits the gate at the end of every epoch, and saves `hcm_gating_stats.pt` at the end:

```bash
python train_hcm_adapter.py \
  --data_path /path/to/preprocessed \
  --experiments_output /path/to/out_stage2 \
  --pretrained /path/to/out_stage1/best_ssim_model.pth.tar \
  --batch_size 1 --max_epochs 20 --lr 2e-4
```

### Stage 3 — Performance-weighted fine-tuning (`finetun.py`)

Unfreezes the whole model with per-center learning rates inversely tied to observed per-center performance (default `lr = 1.5e-4`):

```bash
python finetun.py \
  --data_path /path/to/preprocessed \
  --experiments_output /path/to/out_stage3 \
  --pretrained /path/to/out_stage2/best_ssim_model.pth.tar \
  --resume   # only when continuing the SAME stage; omit for cross-stage transfer
```

### Key arguments (identical interface across all three scripts)

| Argument | Default | Description |
|---|---|---|
| `--data_path` | — | Preprocessed dataset root (contains split JSONs) |
| `--experiments_output` | — | Checkpoints + TensorBoard log directory |
| `--pretrained` | — | Input checkpoint (backbone or previous stage) |
| `--use_checkpoint` | `True` | Load pretrained weights (`strict=False`, tolerant of adapter keys) |
| `--resume` | off | Restore best-metric thresholds for same-stage continuation |
| `--use_subset` / `--subset_ratio` | `True` / `0.3` | Hierarchical stratified subset of training data |
| `--batch_size` | `1` | Per-step batch size (multi-slice k–t volumes) |
| `--lr` | `2e-4` (`1.5e-4` in `finetun.py`) | Base AdamW learning rate |
| `--lr_step_size` / `--lr_gamma` | `3` / `0.9` | StepLR schedule |
| `--weight_decay` | `1e-4` | AdamW regularization |
| `--max_epochs` | `20` | Training length (early stopping, patience = 6, on val SSIM) |
| `--num_low_frequencies` | `[20]` | Fully-sampled central k-space lines |
| `--num_adj_slices` | `5` | Adjacent slices for k–t processing |
| `--task_type` | `regular_task1` | Challenge task variant |
| `--gpus` / `--device` / `--num_workers` / `--seed` | `1` / `cuda` / `1` / `42` | Runtime configuration |

### Multi-GPU (SLURM)

```bash
sbatch train_with_distribution.sbatch   # torchrun, 4 GPUs, 10-day walltime
sbatch train_no_distribution.sbatch     # single-process equivalent
```

Edit partition/account/paths/conda environment at the top of each script for your cluster.

### Monitoring

```bash
tensorboard --logdir /path/to/experiments_output
```

Scalars (train loss, val loss, val SSIM, α values) and example reconstruction image grids are logged per epoch; per-epoch checkpoints (`checkpoint_epoch_N.pth.tar`), the best model (`best_ssim_model.pth.tar`), and the gating statistics (`hcm_gating_stats.pt`) are written to the same directory.

---

## Citation

```bibtex
@inproceedings{xu2025hieradaptmr,
  title={HierAdaptMR: Cross-Center Cardiac MRI Reconstruction with Hierarchical Feature Adapters},
  author={Xu, Ruru and Oksuz, Ilkay},
  booktitle={International Workshop on Statistical Atlases and Computational Models of the Heart (STACOM)},
  pages={299--310},
  year={2025},
  publisher={Springer}
}
```

## Acknowledgements

Built on the **PromptMR-plus** reconstruction backbone (prompt-learning unrolled MR reconstruction) and **fastMRI** tooling; trained on the **CMRxRecon2025** multi-center cardiac MRI challenge dataset. Cluster setup notes are preserved in `UHeM_server.log`.
