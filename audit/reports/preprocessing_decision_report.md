# Bonn EEG Preprocessing Decision & Specification Report
**Project:** Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots (2026 Reproduction)  
**Document ID:** `PREPROCESSING-DECISION-REPORT-01`  
**Date:** 2026-10-10  
**Status:** Awaiting User Confirmation on Unresolved Scientific Decisions  

---

## 1. Current Repository & Environment State

- **Repository Root:** `/Users/anshutiwari123/phase_sync_eeg`
- **Git Tracking Status:**
  - Raw dataset archives (`data/bonn_eeg/archives/`) and extracted recordings (`data/bonn_eeg/extracted/`) are strictly excluded via `.gitignore` and kept immutable.
  - Manifests (`data/bonn_eeg/manifests/`), audit scripts (`scripts/`), audit reports/tables (`audit/`), and documentation (`README.md`) are tracked.
- **Dataset Availability & Integrity:**
  - Complete 500 recordings across subsets A, B, C, D, and E (100 files each).
  - All 500 files verified to contain exactly **4,097 integer samples**.
  - Sampling frequency documented as 173.61 Hz (metadata).
- **Existing Preprocessing Code:**
  - No preprocessing pipeline code currently exists in this repository. All code is currently audit/verification tooling (`download_bonn_dataset.py`, `audit_bonn_dataset.py`).
  - No conflicting preprocessing modules exist.
- **Environment & Dependencies:**
  - Python 3.13.9 (`/opt/anaconda3/bin/python3`).
  - Available packages: `numpy` (2.2.3), `scipy` (1.15.2), `matplotlib` (3.10.0), `pandas`, `requests`.

---

## 2. Fully Specified Requirements (Fixed Decisions)

The following parameters and constraints are fully settled and require no further deliberation:
1. **Dataset Scope:** Bonn EEG subsets A, B, C, D, E (500 recordings total).
2. **Signal Length:** All **4,097 samples** are preserved and processed. No truncation to 4,096.
3. **Immutability:** Raw ASCII files in `data/bonn_eeg/extracted/` remain read-only and unmodified.
4. **Sampling Frequency:** $F_s = 173.61$ Hz is treated strictly as acquisition metadata.
5. **Class Grouping & Benchmark Cases:**
   - Grouping: A+B (Healthy), C+D (Inter-ictal), E (Ictal).
   - Case 1: A+B vs C+D vs E (3-class, $N=500$).
   - Case 2: A+B+C+D vs E (binary, $N=500$).
   - Case 3: D vs E (binary, $N=200$).
   - Note on Paper Inconsistency: Table 1 labels C & D as "Pre-ictal", whereas Section 3.6.2 and the foundational paper (Andrzejak et al., 2001) define them as "inter-ictal". Standard inter-ictal interpretation is adopted.
6. **Filtering:** No additional digital software filtering (bandpass, highpass, or notch) is applied before embedding, in strict adherence to Section 3 of the paper.
7. **GLCM Configuration (Ground Truth: Section 3.5):**
   - Orientations: $\theta \in \{0^\circ, 45^\circ\}$ ($0, \pi/4$).
   - Offsets: $d \in \{1, 2, 3, 4\}$.
   - Total GLCMs: Exactly 8 matrices per recording (the conflicting 16-GLCM description in Section 3.3.2 is explicitly rejected).

---

## 3. Unresolved Scientific Implementation Details

The primary paper describes high-level algorithms but omits critical mathematical parameters and implementation details required for deterministic software execution.

### Decision 1: Z-score Normalization Convention & Epsilon
- **Mathematical Expression:** $z = \frac{x - \mu}{\sigma}$
- **Missing Detail:**
  - Population standard deviation (`ddof=0`, denominator $N$) vs. Sample standard deviation (`ddof=1`, denominator $N-1$).
  - Denominator stability epsilon $\epsilon$ or fallback if $\sigma = 0$.
- **Scientific Impact:** For $N=4097$, $\sqrt{4097/4096} \approx 1.000122$. In all 500 Bonn files, $\sigma \ge 18.6\,\mu\text{V}$, so $\sigma=0$ does not occur, but exact numerical values differ slightly between `ddof=0` and `ddof=1`.
- **Recommendation:** `ddof=0` (standard population z-score, matching `scipy.stats.zscore`) with guard $\epsilon = 10^{-12}$.

### Decision 2: Average Mutual Information (AMI) Estimator for Time Delay $\tau$
- **Paper Statement:** *"$\tau$: selected as the first minimum of the Average Mutual Information (AMI) curve to ensure minimal redundancy between delayed coordinates... typical values were $\tau = 10\text{--}15$ samples ($\approx 60\text{--}90$ ms)"* (Section 3.2).
- **Missing Detail:**
  - **Probability Estimator:** Equi-width histogram binning (Fraser & Swinney, 1986) vs. Equi-quantile / adaptive binning vs. Kernel density / k-NN.
  - **Number of Bins ($B$):** Common literature choices: Fixed 16 bins, fixed 32 bins, fixed 64 bins, or Sturges formula ($B = 1 + \lceil\log_2 N\rceil \approx 14$ bins).
  - **Search Range:** Maximum lag $\tau_{\max}$ evaluated (e.g. $\tau \in [1, 50]$ or $\tau \in [1, 100]$).
  - **Definition of "First Minimum":** Strict local minimum ($AMI(\tau-1) > AMI(\tau) < AMI(\tau+1)$) vs. smoothed minimum.
  - **Fallback:** What if AMI decays monotonically without a local minimum in the search window? (e.g., fallback to the typical range center $\tau=10$, or first lag where $AMI(\tau) \le 1/e \cdot AMI(0)$).
- **Recommendation:** Equi-width histogram with $B=32$ bins, lag range $\tau \in [1, 50]$, strict first local minimum, with fallback to first lag dropping below $1/e$ or $\tau=10$ if monotonic.

### Decision 3: False Nearest Neighbours (FNN) for Embedding Dimension $m$
- **Paper Statement:** *"m: chosen such that the False-Nearest-Neighbour (FNN) ratio < 1%... typical values were $m = 5 \pm 1$."* (Section 3.2).
- **Missing Detail:**
  - **Distance Metric:** Euclidean ($L_2$) vs Maximum norm ($L_\infty$).
  - **Kennel et al. (1992) Threshold Parameters:**
    1. Distance ratio threshold: $R_{\text{tol}}$ (standard literature uses $R_{\text{tol}} \in [10, 15]$; typically 15).
    2. Attractor size threshold: $A_{\text{tol}}$ (Kennel second criterion: $(R_{d+1} / R_A) > A_{\text{tol}}$, where $R_A = \text{std}(x)$ and $A_{\text{tol}} = 2$).
  - **Dimension Search Range:** $m \in [1, 10]$ vs $m \in [1, 15]$.
  - **Theiler Window:** Exclude temporal neighbors $|i - j| \le W$ (typically $W = \tau$ or $W = 1$) to prevent temporal autocorrelation from contaminating false neighbor statistics.
- **Recommendation:** Standard Kennel et al. (1992) FNN with Euclidean metric, $R_{\text{tol}}=15$, $A_{\text{tol}}=2$, Theiler window $W=\tau$, searching $m \in [1, 10]$, stopping at first $m$ where FNN ratio $< 1\%$ (fallback to $m=5$ if threshold not reached by $m=10$).

### Decision 4: Recurrence Distance Metric & Gray-Level Quantization Parameter $K$
- **Paper Statements:**
  - Distance: *"In many cases, Euclidean distance is considered... However, Euclidean distance has a higher computational cost than other distance measures, such as Manhattan distance and squared distance."* (Section 3.3).
  - Quantization: *"$\text{RP}_{\text{quantized}} = \text{round}\left( \frac{\text{RP}}{\max(\text{RP})} \cdot k \right)$ (Eq. 20)"* and *"(3) converted to a continuous recurrence distance image (GRP) and quantized to $K$ gray levels;"* (Section 3.5).
- **Missing Detail:**
  - The numerical value of $K$ is **never stated anywhere in the paper**.
  - Distance metric actually used for the EEG corpus is not explicitly finalized between Euclidean ($L_2$), Manhattan ($L_1$), or Squared Euclidean ($L_2^2$).
  - Discrete Level Range: $\text{round}\left( \frac{\text{RP}}{\max(\text{RP})} \cdot K \right)$ produces integer values in $\{0, 1, \dots, K\}$, which is $K+1$ discrete levels. In Eqs. (6)-(19), indices run from $1$ to $N_g$.
- **Recommendation:** Euclidean distance ($L_2$) for canonical phase-space geometry; $K=16$ or $K=32$ gray levels (standard for GLCM recurrence texture analysis); discrete levels mapped to $N_g$ gray levels.

### Decision 5: Total GLCM Feature Count (96 vs 40 Features)
- **Paper Discrepancy:**
  - Section 3.5 (p. 5) lists: *"(Energy, Entropy, Contrast, Homogeneity, Correlation concatenated across $\theta, d$)"* — which is 5 features $\times$ 8 GLCMs = **40 features**.
  - Section 3.5 continuation (p. 7, lines 276-278) states: *"There are 12 features for each of the 8 GLCMs and a total of 96 features. 10 feature subsets are selected for each case by the supervised multi-cluster feature selection (MCFS) method."*
  - Section 3.3.4–3.3.15 explicitly defines all 12 Haralick equations ($f_1$ to $f_{12}$): Energy, Entropy, Contrast, Homogeneity, Correlation, Dissimilarity, Variance, Max Probability, Sum Average, Cluster Shade, Cluster Prominence, Autocorrelation.
- **Evidence Synthesis:** The text on p. 7 explicitly explains that the full feature pool consists of **96 features** ($12 \times 8$), from which MCFS selects top-10. The 5 features mentioned on p. 5 were illustrative examples.
- **Recommendation:** Implement the full **96-feature set** ($12 \text{ Haralick features} \times 8 \text{ GLCMs}$), strictly matching the explicit statement *"12 features for each of the 8 GLCMs and a total of 96 features"*, with clear metadata labeling each of the 96 feature columns.

---

## 4. Proposed Minimal File Changes (Phase 4)

Once decisions are approved, implementation will proceed with the following modular, minimal files:
```text
scripts/
├── preprocessing/
│   ├── __init__.py
│   ├── normalization.py     # Per-recording z-score with approved convention & epsilon
│   ├── embedding.py         # AMI lag tau & Kennel FNN dimension m estimation
│   ├── grp.py               # Phase-space trajectory, GRP distance matrix, K-level quantization
│   ├── glcm.py              # 8 GLCMs (theta=0, 45; d=1..4) and 12 Haralick equations
│   └── pipeline.py          # Coordinated single-recording and batch preprocessing runner
tests/
├── test_normalization.py    # Unit tests for z-score math and immutability
├── test_embedding.py        # Unit tests for AMI and FNN with known signals (sine/Lorenz)
├── test_grp.py              # Unit tests for GRP construction and quantization bounds
└── test_glcm.py             # Unit tests for GLCM orientation, symmetry, and Haralick equations
```

---

## 5. Single-Recording Pilot Test Plan (Phase 5)

We will validate the pipeline on exactly one representative recording (e.g., `Z001.txt`):
1. **Raw File Invariance:** Verify SHA-256 of `data/bonn_eeg/extracted/A/Z001.txt` is identical before and after.
2. **Intermediate Array Dimensionality:**
   - Raw signal: $(4097,)$
   - Normalized $z$: $(4097,)$, mean $\approx 0.0$, std $\approx 1.0$
   - Delay $\tau$ and Dimension $m$: Compare with typical paper ranges ($\tau \approx 10\text{--}15$, $m \approx 4\text{--}6$)
   - Embedded trajectory $X$: $(N_t, m)$ where $N_t = 4097 - (m-1)\tau$
   - GRP distance matrix: $(N_t, N_t)$, symmetric, zero diagonal
   - Quantized GRP: $(N_t, N_t)$ integer array in range $[0, K]$
   - 8 GLCMs: each shape $(N_g, N_g)$, normalized probabilities $\sum p(i,j) = 1.0$
   - Extracted Feature Vector: Exactly 96 scalar floats (or 40 if Option B is chosen)
3. **Data Quality Checks:** Zero NaNs, zero Infs, no constant feature vectors.
4. **Execution Log:** Save diagnostics to `audit/logs/pilot_recording.log`.

---

## 6. Methodological Decisions & Reproducibility Notes

### False Nearest Neighbours
“The embedding dimension was selected using an FNN ratio threshold of less than 1%, with the current implementation applying the relative-distance criterion using `R_tol = 15.0`. The implementation does not apply the second absolute-distance criterion (`A_tol`) of the canonical two-criterion Kennel method. This was a deliberate implementation choice for this replication and is not asserted to be an explicitly reported parameter of the primary paper. The selected embedding dimension is estimated per recording and is not assumed to be constant across the dataset.”

### GRP quantization
“The continuous generalized recurrence-distance matrix was quantized to 16 gray levels indexed from 0 to 15 using `round(RP / max(RP) * (K - 1))`, with `K = 16`. The primary paper's printed equation multiplies by `K`; the implementation uses `K - 1` to obtain exactly 16 discrete levels compatible with a 16 × 16 GLCM. This is documented as a replication decision resolving an ambiguity in the paper, rather than a literal implementation of the printed equation.”

### GLCM symmetrization
“The GLCMs were symmetrized before probability normalization, following the existing implementation retained for this replication. The available description in the primary paper does not clearly establish whether symmetrization was used; therefore, this is recorded as an implementation choice rather than a verified author-specified parameter.”
