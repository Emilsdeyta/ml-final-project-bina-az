# Crack Detection Project — Final Deliverable

Concrete/pavement crack detection via CNN segmentation, mathematical regression on real
cracks, synthetic crack generation, and the inverse (curve-recovery) problem.

## Structure

```
crack_detection_project/
├── src/                              # Core, tested Python modules (importable, no bugs)
│   ├── dataset.py                    # Dataset class + leakage-safe splitting utilities
│   ├── model.py                      # U-Net (7.7M params, base_filters=32)
│   ├── losses_metrics.py             # BCE+Dice loss, IoU/Dice/Precision/Recall, tolerant IoU
│   ├── train.py                      # Resumable training loop (Kaggle-session-safe)
│   ├── curve_fit.py                  # Regression on REAL crack masks (skeleton -> polynomial)
│   ├── synthetic.py                  # Synthetic circle/sine crack image generator
│   └── inverse_fit.py                # Inverse problem: recover circle/sine params from pixels
├── notebooks/                        # Kaggle-ready, self-contained notebooks (GPU)
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_segmentation.ipynb   # <- main deliverable: training + benchmark
│   ├── 03_regression_real_cracks.ipynb
│   ├── 04_synthetic_cracks.ipynb
│   └── 05_inverse_problem.ipynb
└── PROJECT_LOG.md                    # Full debugging history & findings (source for report)
```

## How to run (Kaggle, GPU)

1. Create/open a Kaggle Notebook, attach the `concrete-crack-conglomerate` dataset,
   enable **GPU** (Settings → Accelerator → GPU T4).
2. Upload each notebook (or copy/paste) and run **top to bottom, in numeric order**
   (01 → 02 → 03 → 04 → 05). Each notebook is self-contained: the first cells
   `%%writefile` the `src/*.py` modules into `/kaggle/working/src`, so no separate file
   upload is required — the `.py` files here are provided as the readable/reviewable
   source, and the notebooks embed the exact same code.
3. **Notebook 02 is checkpoint/resume-safe.** If a Kaggle session is interrupted, simply
   re-run the notebook — it detects `/kaggle/working/checkpoint.pt` and continues from
   the last completed epoch automatically.
4. Notebook 05 depends on notebook 04's output (`synthetic/manifest.json`) — run 04 first.

**If your dataset path differs** from
`/kaggle/input/concrete-crack-conglomerate/Task Data 3/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection`,
update the `ROOT = "..."` line near the top of each notebook.

## What each notebook produces

| Notebook | Produces |
|---|---|
| 01 | Full EDA: source-mix, class imbalance, empty-mask %, sample visualizations |
| 02 | Trained model, loss/IoU curves, official test benchmark, leakage-aware benchmark, tolerant-IoU analysis, qualitative predictions, room-for-improvement notes |
| 03 | Parametric polynomial fits for ≥3 real crack masks, R², interpretation |
| 04 | 10 synthetic binary crack images (circle/sine, single & multi-function), saved masks + ground-truth params |
| 05 | Recovered circle/sine coefficients vs. ground truth, per-image comparison table |

## Key methodological finding (see PROJECT_LOG.md for full detail)

This dataset is built from large source photos cut into many overlapping crops. A
**naive filename-level train/val/test split leaks near-duplicate crops** between sets
and inflates IoU by roughly 0.10-0.20. All splitting in this project is done by
**source-photo ID** (grouping crops from the same original photo), which was verified to
give zero overlap. Both the "raw" official-test-set benchmark and a **leakage-corrected
"novel subset" benchmark** are reported in notebook 02 — report the novel-subset numbers
as the trustworthy generalization estimate; the raw numbers are included for
completeness/comparison only.

## Notes on scope

- Presentation (PPTX) and written report (Word) are intentionally **not** produced here,
  per current priority — the notebooks' markdown cells contain all the interpretation
  text needed to write both quickly once results are in.
- `src/` modules were unit-tested locally (13-sample subset, CPU) for correctness before
  being embedded in the notebooks; full-scale numbers (9899/1096 images, GPU) require
  running notebook 02 on Kaggle.
