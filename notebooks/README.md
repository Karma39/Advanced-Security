[← Back to root](../README.md)

# `notebooks/` — Experiment Notebooks

These notebooks implement the full experimental workflow, from building data caches through the final ML vs. DL comparison. **Run them in order.** Each notebook saves its results to `results/` as a JSON file, so later notebooks can load results from earlier ones without re-running training.

---

## Execution Order

| # | Notebook | What it does | Outputs |
|---|----------|-------------|---------|
| 0 | `00_setup.ipynb` | Build all HDF5 caches from raw data | `processed_data/*.h5` (4 files) |
| 1 | `01_ml_baseline.ipynb` | RnF+ANOVA baseline, three protocol variants | `results/nb01_ml_baseline_trial_*.json` |
| 2 | `02_dl_baseline.ipynb` | DL baseline, 5 seeds, group-aware protocol | `results/nb02_dl_baseline_trial_*.json` |
| 3 | `03_comparison.ipynb` | ML vs. DL side-by-side | Figures (displayed inline) |
| 4 | `04_multiclass_classification.ipynb` | 12-class device identification | `results/nb04_multiclass_all_devices.json` |

---

## Notebook Descriptions

### `00_setup.ipynb` — Cache Builder
**Run this first, once.** Calls `dl_classification/run_pipeline.py` and `raw_dgt/run_pipeline_dgt.py` to process the raw `.bin` IQ recordings for all 12 devices and write four HDF5 files to `processed_data/`:

| Cache | Samples | Feature type |
|-------|---------|-------------|
| `fv_original.h5` | 134 | 505-dim feature vectors, 1 per transient |
| `fv_windowed.h5` | 1340 | 505-dim feature vectors, 10 windows/transient |
| `dgt_original.h5` | 134 | 150×150 DGT matrices, 1 per transient |
| `dgt_windowed.h5` | 1340 | 150×150 DGT matrices, 10 windows/transient |

All subsequent notebooks load from these caches — never from raw `.bin` files directly.

---

### `01_ml_baseline.ipynb` — RnF+ANOVA Baseline

Replicates the original `RF_Fingerprint.py` pipeline (RnF+ANOVA) and progressively aligns the evaluation protocol to match the DL pipeline. Three protocol differences are identified and applied one at a time to quantify their individual impact on reported ADR.

**Variant A — Fold-aware feature selection:** ANOVA top-k is fitted on training data only within each fold, rather than on the full dataset before splitting.

**Variant B — Full-fold averaging:** ADR is averaged across all CV folds rather than reporting only the best fold.

**Variant C — Fixed k:** The number of features k is fixed a priori (k=50) rather than swept over the test set.

| Protocol | ADR |
|----------|-----|
| Original (paper protocol) | **0.973** |
| Variant A | 0.786 |
| Variant A+B | 0.554 |
| Variant A+B+C | **0.458** |

Results are saved per trial and loaded by `03_comparison.ipynb`.

---

### `02_dl_baseline.ipynb` — Deep Learning Baseline
Trains three deep learning models under a group-aware three-way split (70% train / 15% val / 15% test) where all windows from the same transient stay in the same partition. Each model is trained over 5 random seeds to quantify uncertainty.

**Models evaluated:**
- **MLP-FV** — 3-layer MLP on 505-dim feature vectors (same features as the SVM).
- **GRU-DGT** — 2-layer unidirectional GRU on 150×150 DGT matrices.
- **BiGRU+SpecAug** — 2-layer bidirectional GRU on DGT matrices, with SpecAugment regularization during training.

**Results (5 seeds, held-out test set):**

| Model | Auth TVR | Rogue TVR | ADR |
|-------|----------|-----------|-----|
| MLP-FV | 0.624 ± 0.050 | 0.680 ± 0.015 | **0.660 ± 0.019** |
| GRU-DGT | 0.648 ± 0.131 | 0.567 ± 0.059 | **0.604 ± 0.055** |
| BiGRU+SpecAug | 0.641 ± 0.069 | 0.616 ± 0.051 | **0.627 ± 0.025** |

All three are statistically indistinguishable (Wilcoxon p > 0.0625 for all pairs; n=5 is the structural minimum for p < 0.05).

---

### `03_comparison.ipynb` — ML vs. DL Comparison
Loads the saved results from notebooks 01 and 02 and displays them side by side. Examines whether the DL models offer any advantage over the RnF+ANOVA baseline under the aligned protocol (ADR = 0.458), and discusses whether the gap is architectural or a consequence of the small dataset size (134 transients total).

---

### `04_multiclass_classification.ipynb` — 12-Class Device Identification
Extends the experiment to **multi-class classification**: instead of one binary classifier per device, a single model identifies which of the 12 devices transmitted a given transient. This is a harder but more scalable formulation.

Trains MLP-FV and BiGRU on all 12 devices and compares per-class accuracy. The goal is to test whether the low ADR in the binary experiments is due to limited data (the fundamental constraint) or to the binary classification formulation.

---

## Modules used by these notebooks

- [`binary_pla/`](../binary_pla/README.md) — training loop, models, data loader (used by notebooks 02, 04)
- [`dl_classification/`](../dl_classification/README.md) — FV cache builder (used by notebook 00)
- [`raw_dgt/`](../raw_dgt/README.md) — DGT cache builder (used by notebook 00)
- [`src/`](../src/README.md) — original preprocessing utilities (used indirectly via dl_classification)
