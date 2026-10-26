# Data

All arrays are NumPy `.npz`. Load them with `numpy.load`; no pickle is involved.

## `dataset_2035.npz` — main corpus

| key | shape | dtype | meaning |
|---|---|---|---|
| `X_train` | (2035, 6) | float32 | geometry, columns `L1 [mm], r2 [mm], t1 [deg], t2 [deg], w3 [mm], w4 [mm]` |
| `s_train` | (2035, 401) | float32 | return loss `S11 [dB]`, clipped at 0 dB |
| `g_train` | (2035, 401) | float32 | realised gain `[dBi]` |
| `freqs` | (401,) | float32 | frequency grid, 2.5-12.0 GHz |

The curves are Savitzky-Golay smoothed (window 15, third order) before clipping, which
removes solver jitter without moving the resonances.

## `split_indices.npz` — frozen partition

| key | shape | meaning |
|---|---|---|
| `train` | (1423,) | training rows |
| `validation` | (306,) | early-stopping rows |
| `final` | (306,) | held-out rows used for every reported metric |

The three blocks are disjoint and together cover all 2035 designs. They are rebuilt
exactly by grouping the geometries on their rounded coordinates
`np.round(X, 3)` and carving two blocks of 306 designs with `GroupShuffleSplit`
(`random_state=2025`), which leaves 1423 designs for training. All normalisers and the
PCA bases used by the surrogate are fitted on the `train` block only.

## `rogers_data.npz` / `rogers_split.npz` — substrate-transfer set

360 designs solved on a second substrate (`eps_r = 3.80`), same six geometry columns and
the same 401-point frequency grid, split 252 / 54 / 54. These support the fine-tuning
results: 252 designs are enough to match a model trained from scratch on 1423.

## Reproducing the reported accuracies

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

d = np.load("data/dataset_2035.npz")
sp = np.load("data/split_indices.npz")
X, S, G = d["X_train"], d["s_train"], d["g_train"]
tr, te = sp["train"], sp["final"]

# every scaler and PCA basis is fitted on the training rows only
sx = MinMaxScaler((-1, 1)).fit(X[tr])
ss = MinMaxScaler((-1, 1)).fit(S[tr])
sg = MinMaxScaler((-1, 1)).fit(G[tr])
pca_s = PCA(n_components=40, random_state=0).fit(ss.transform(S[tr]))
pca_g = PCA(n_components=40, random_state=0).fit(sg.transform(G[tr]))
```

A shared-trunk network trained on this split reaches an `S11` mean absolute error of
0.72 dB (R^2 = 0.884) over the 306 held-out designs; the paper reports 0.71 dB
(R^2 = 0.89) for the deployed expert.

## Raw exports

The `*.csv` files at the repository root are the untouched ANSYS HFSS exports the corpus
was built from, in long format (one row per frequency per design):

* `Terminal S Parameter Plot N.csv` — columns `L1, r2, t1, t2, w3, w4, Freq, dB(St(1,1))`
* `Gain Plot N.csv` — far-field gain, columns include `Phi`, `Theta`, `Freq`, `dB(GainTotal)`
