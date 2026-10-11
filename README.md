# Bonn EEG Dataset Acquisition & Reproducibility Audit

This repository contains the acquisition, verification, and forensic reproducibility audit scripts for the official **University of Bonn EEG Dataset**.

This audit serves as Task 1 for reproducing the methodology of:
> **Lashkari et al. (2026)**, *"Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots"*, *Biomedical Signal Processing and Control*, 118, 109675. [doi:10.1016/j.bspc.2026.109675](https://doi.org/10.1016/j.bspc.2026.109675).

The authoritative dataset source reference is:
> **Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001)**, *"Indications of nonlinear deterministic and finite dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state"*, *Physical Review E*, 64(6), 061907. [doi:10.1103/PhysRevE.64.061907](https://doi.org/10.1103/PhysRevE.64.061907).

---

## 1. Directory Structure

```text
phase_sync_eeg/
├── data/
│   └── bonn_eeg/
│       ├── archives/           # Official downloaded ZIP archives (z.zip, o.zip, n.zip, f.zip, s.zip)
│       ├── extracted/          # Extracted subsets A, B, C, D, E (100 signal TXT files each)
│       └── manifests/          # Raw extraction JSON manifest
├── audit/
│   ├── reports/
│   │   ├── download_report.md        # Download logs, HTTP status, and SHA-256 archive hashes
│   │   ├── dataset_audit_report.md   # Comprehensive dataset audit & discrepancy analysis
│   │   └── figures/                  # Visualization plots (sample counts, distributions, ADC bounds)
│   ├── tables/
│   │   ├── file_manifest.csv               # File-by-file manifest with hashes & parse statuses
│   │   ├── recording_signal_statistics.csv # Sample counts, extrema, std, IQR, zeros, ADC saturation
│   │   ├── subset_summary.csv              # Summary per subset (A–E)
│   │   ├── duplicate_report.csv            # Pairwise duplicate check report
│   │   └── paper_vs_dataset.csv            # Itemized comparison matrix (Paper vs Docs vs Data)
│   └── logs/
│       ├── download.log        # Streaming download & extraction log
│       └── audit.log           # Dataset audit execution log
├── scripts/
│   ├── download_bonn_dataset.py  # Safe streaming downloader & archive extractor
│   └── audit_bonn_dataset.py     # Standalone dataset verification and audit engine
├── .gitignore                  # Excludes raw data archives & signals from version control
└── README.md                   # Independent execution instructions & documentation
```

---

## 2. Environment Prerequisites

- **Python:** 3.10+ (tested on Python 3.13.9)
- **Standard Dependencies:** `numpy`, `scipy`, `matplotlib`, `requests`

```bash
# Verify environment dependencies
python3 -c "import numpy, scipy, matplotlib, requests; print('All prerequisites available')"
```

---

## 3. Independent Execution Commands

### Step A: Download & Safely Extract Archives
Downloads the five official subsets from University Hospital Bonn via streaming HTTP chunks (with exponential backoff and SHA-256 integrity checks), and extracts them safely:
```bash
python3 scripts/download_bonn_dataset.py
```
*Note: If valid archives already exist in `data/bonn_eeg/archives/`, the script verifies their integrity and reuses them without redundant network traffic.*

### Step B: Run the Forensic Dataset Audit
Reruns the dataset audit on the extracted files without re-downloading, recalculating all statistical metrics, generating tables, and generating verification plots:
```bash
python3 scripts/audit_bonn_dataset.py
```

---

## 4. Key Verified Findings

| Metric / Check | Value / Finding |
|---|---|
| **Subsets Audited** | 5 (`A`, `B`, `C`, `D`, `E`) |
| **Total Signal Files** | Exactly 500 (100 recordings per subset) |
| **Observed Samples per Recording** | **4,097 integer samples** across all 500 files (Discrepancy: Paper states 4,096) |
| **Sampling Frequency** | 173.61 Hz (Official documentation; unstated in raw integer files) |
| **Duration per Recording** | 4,097 / 173.61 = **23.5989 seconds** (~23.60 s) |
| **Parse Errors / NaNs / Infs** | 0 (100% clean integer ASCII format) |
| **Duplicate Signals** | 0 (All 500 signal arrays and SHA-256 digests are unique) |
| **ADC Saturation (+2047 μV)** | 12-bit signed ADC dynamic range `[-2048, +2047]`. Observed in `F009.txt` (Set D, 52 samples), `S059.txt` (Set E, 2 samples), `S065.txt` (Set E, 18 samples). |
| **Archive Artifacts** | All official archives contain Mac OS X metadata (`__MACOSX/._*`), safely isolated. |
| **File Extension Case** | Subset C uses `.TXT` (uppercase), while Subsets A, B, D, E use `.txt` (lowercase). |

---

## 5. Preprocessed Feature Dataset (`bonn_features_96.csv`)

To eliminate the need to repeat the resource-intensive preprocessing pipeline whenever a cloud compute (Kaggle/Colab) session expires, the preprocessed 96-feature dataset is included directly in this repository:

- **Location:** [`audit/batch_preprocessing/bonn_features_96.csv`](audit/batch_preprocessing/bonn_features_96.csv)
- **File Size:** ~909 KB (931,091 bytes CRLF)
- **Shape:** 500 samples $\times$ 103 columns (7 metadata columns + 96 GLCM feature columns)
- **SHA-256 Checksum:** `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1`

### Feature Extraction Pipeline
1. **Z-Score Normalization:** Population standard deviation (`ddof=0`).
2. **AMI Time Delay ($\tau$):** 32 equi-width bins, lags 1–50, fallback $\tau=10$.
3. **FNN Embedding Dimension ($m$):** Kennel Criterion 1 ($R_{\text{tol}}=15.0$, Theiler window $W=\tau$, threshold $< 1\%$).
4. **Generalized Recurrence Plot (GRP):** Continuous Euclidean distance matrix.
5. **Quantization ($K=16$):** Scaled via $\text{round}(RP / \max(RP) \times 15)$, yielding integer gray levels $\{0, \dots, 15\}$.
6. **GLCM Computation:** 8 symmetric, normalized Gray-Level Co-occurrence Matrices (angles $0^\circ$ and $45^\circ$, offsets $d \in \{1, 2, 3, 4\}$).
7. **Haralick Features:** 12 features per GLCM using 1-based indexing, producing exactly $8 \times 12 = 96$ features.

### Schema & Class Mapping
- **Columns 1–7 (Metadata):** `recording_id`, `filename`, `subset`, `class_label`, `case1_label`, `case2_label`, `case3_label`.
- **Columns 8–103 (Features):** `glcm_a00_d1_f01_energy` ... `glcm_a45_d4_f12_autocorrelation`.
- **Class Splits:**
  - `Case1_3Class`: healthy (200), interictal (200), ictal (100)
  - `Case2_Binary`: non_ictal (400), ictal (100)
  - `Case3_Pilot`: interictal (100), ictal (100) (Subsets D & E only)

---

## 6. Cloud / Kaggle Resumable Workflow

### Verification in Kaggle
```bash
shasum -a 256 audit/batch_preprocessing/bonn_features_96.csv
# Verified: af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1
```

### Automatic Resumable Training with Remote Checkpoint Restore
```bash
python scripts/classification/run_kaggle_k5.py \
    --input audit/batch_preprocessing/bonn_features_96.csv \
    --output /kaggle/working/experiments/fixed_k5 \
    --restore-dataset \
    --resume \
    --remote-restore
```
- Restores preprocessed dataset if not locally available.
- Restores outer-fold checkpoints from `anshu-ai-arch/eeg`.
- Automatically backs up each completed outer fold via GitHub API (`fold_callback`).
