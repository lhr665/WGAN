# Data

All arrays are NumPy `.npz`. Load them with `numpy.load`; no pickle is involved.

## `dataset_2035.npz` — main corpus

| key | shape | dtype | meaning |
|---|---|---|---|
| `X_train` | (2035, 6) | float32 | geometry: `L1 [mm], r2 [mm], t1 [deg], t2 [deg], w3 [mm], w4 [mm]` |
| `s_train` | (2035, 401) | float32 | return loss `S11 [dB]`, clipped at 0 dB |
| `g_train` | (2035, 401) | float32 | realised gain `[dBi]` |
| `freqs` | (401,) | float32 | frequency grid, 2.5–12.0 GHz |

Curves are Savitzky–Golay smoothed (window 15, third order) before clipping, which removes
solver jitter without moving the resonances.

## `split_indices.npz` — the frozen partition used in the paper

| key | shape | meaning |
|---|---|---|
| `train` | (1423,) | training rows |
| `validation` | (306,) | early-stopping rows |
| `final` | (306,) | held-out rows behind every reported metric |

The three blocks are disjoint and together cover all 2035 designs. Geometries are grouped
on `np.round(X, 3)`, so a design and any near-duplicate of it always land in the same
block; that grouping is what prevents the leakage audited in the paper.

**The indices are shipped as a fixed array rather than regenerated on the fly.** Reproduce
the paper by loading this file, not by re-splitting: the same grouping rule with a
different random seed yields different held-out designs and shifts the accuracies by a few
hundredths of a dB.

All normalisers and PCA bases are fitted on the `train` block only; fitting them on the
full corpus would leak the held-out rows.

Sanity check:

```python
import numpy as np
d  = np.load("data/dataset_2035.npz")
sp = np.load("data/split_indices.npz")
assert len(d["X_train"]) == 2035
assert (len(sp["train"]), len(sp["validation"]), len(sp["final"])) == (1423, 306, 306)
assert not (set(sp["train"]) & set(sp["final"]))
```

## `rogers_data.npz` / `rogers_split.npz` — substrate-transfer set

360 designs solved on a second substrate (`eps_r = 3.80`), same six geometry columns and
the same 401-point grid, split 252 / 54 / 54. These back the transfer results: fine-tuning
on 252 designs reaches an `S11` mean absolute error of 0.70 dB (R² = 0.875), matching a
model trained from scratch on 1423 designs (R² = 0.878).

## Reference accuracies on the 306 held-out designs

| model | `S11` MAE | `S11` R² | gain MAE | gain R² |
|---|---|---|---|---|
| dual-branch ResNet (Sec. III-A) | 0.85 dB | 0.86 | 0.69 dBi | 0.89 |
| shared trunk + decoupled heads | 0.77 dB | 0.88 | 0.64 dBi | 0.90 |
| deployed expert checkpoint (Sec. III-F) | 0.89 dB | 0.84 | 0.87 dBi | 0.84 |

The deployed checkpoint is the one embedded in the design loop and is the model behind the
error ledger in Sec. III-F; the dual-branch network is the architecture described in
Sec. II-B. Because the two differ, the two sections quote different numbers for the same
held-out designs.

## Raw exports

The `*.csv` files at the repository root are the untouched ANSYS HFSS exports the corpus
was built from, in long format (one row per frequency per design):

* `Terminal S Parameter Plot N.csv` — `L1, r2, t1, t2, w3, w4, Freq, dB(St(1,1))`
* `Gain Plot N.csv` — far-field gain, columns include `Phi`, `Theta`, `Freq`, `dB(GainTotal)`
