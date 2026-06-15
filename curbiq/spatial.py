r"""Spatial-statistics engine (pure numpy/scipy/h3).

Implements the rigorous, defensible hotspot toolkit used by ``hotspots.py``:

* **Getis-Ord Gi\***   — per-cell z-score; identifies statistically significant
  spatial clusters of high (hot) or low (cold) violation intensity.
* **Global Moran's I** — single headline measure of spatial autocorrelation,
  with an analytical z-score under the randomization assumption.
* **Local Moran's I**  — LISA quadrant labels (HH/LL/HL/LH) for cluster/outlier
  description.
* **Benjamini-Hochberg FDR** — multiple-comparison correction so "significant"
  hotspots survive testing thousands of cells at once (the same FDR approach
  ArcGIS applies to Optimized Hot Spot Analysis).

Spatial weights are H3 grid-disk contiguity (k rings). Gi\* includes the cell
itself (the "star"); Moran's I excludes it and is row-standardized.
"""
from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np
from scipy import sparse
from scipy.stats import norm

from curbiq import config as C


# --------------------------------------------------------------------------- #
# Lattice + spatial weights
# --------------------------------------------------------------------------- #
def build_active_lattice(cell_values: dict[str, float], k: int = C.GI_NEIGHBOR_K):
    """Universe of cells = populated cells ∪ their k-ring neighbours.

    Including the immediate neighbourhood (with value 0) lets a populated cell
    surrounded by empty space register as a genuine local peak.
    Returns ``(cells, x, idx)``: ordered cell list, value array, name->index map.
    """
    active: set[str] = set(cell_values)
    for c in list(cell_values):
        active.update(h3.grid_disk(c, k))
    cells = sorted(active)
    idx = {c: i for i, c in enumerate(cells)}
    x = np.zeros(len(cells), dtype=float)
    for c, v in cell_values.items():
        x[idx[c]] = v
    return cells, x, idx


def build_weights(cells: list[str], idx: dict[str, int], k: int,
                  include_self: bool, row_standardize: bool = False) -> sparse.csr_matrix:
    """Binary (optionally row-standardized) H3 grid-disk contiguity matrix."""
    rows, cols = [], []
    for c in cells:
        i = idx[c]
        for nb in h3.grid_disk(c, k):
            j = idx.get(nb)
            if j is None or (not include_self and j == i):
                continue
            rows.append(i)
            cols.append(j)
    n = len(cells)
    W = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    if row_standardize:
        deg = np.asarray(W.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        W = sparse.diags(1.0 / deg) @ W
    return W.tocsr()


# --------------------------------------------------------------------------- #
# Getis-Ord Gi*
# --------------------------------------------------------------------------- #
def getis_ord_gi_star(x: np.ndarray, W: sparse.csr_matrix) -> np.ndarray:
    """Gi* z-scores. ``W`` must be binary and include the diagonal (self)."""
    n = len(x)
    if n < 2:
        return np.zeros(n)
    wsum = np.asarray(W.sum(axis=1)).ravel()          # Σ_j w_ij
    wx = W.dot(x)                                      # Σ_j w_ij x_j
    xbar = x.mean()
    s = np.sqrt(max((x ** 2).mean() - xbar ** 2, 0.0))
    # Σ_j w_ij² == wsum for binary weights
    denom = s * np.sqrt(np.maximum((n * wsum - wsum ** 2) / (n - 1), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (wx - xbar * wsum) / denom
    z[~np.isfinite(z)] = 0.0
    return z


def z_to_p_two_sided(z: np.ndarray) -> np.ndarray:
    return 2.0 * norm.sf(np.abs(z))


def benjamini_hochberg(p: np.ndarray, q: float = C.FDR_Q):
    """Return (significant_mask, critical_p) controlling FDR at level ``q``."""
    n = len(p)
    if n == 0:
        return np.zeros(0, bool), 0.0
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, bool), 0.0
    kmax = np.max(np.where(below)[0])
    pcrit = float(ranked[kmax])
    return p <= pcrit, pcrit


def gi_confidence_band(z: float, significant: bool) -> str:
    """Human-readable confidence band for a Gi* z-score."""
    if not significant:
        return "not_significant"
    a = abs(z)
    if a >= C.GI_Z_BANDS["99.9%"]:
        lvl = "99.9%"
    elif a >= C.GI_Z_BANDS["99%"]:
        lvl = "99%"
    elif a >= C.GI_Z_BANDS["95%"]:
        lvl = "95%"
    else:
        lvl = "90%"   # FDR-significant but |z| below the 90% band -> lowest tier
    return f"{'hot' if z > 0 else 'cold'}_{lvl}"


# --------------------------------------------------------------------------- #
# Moran's I (global + local)
# --------------------------------------------------------------------------- #
@dataclass
class MoranResult:
    I: float
    expected: float
    variance: float
    z: float
    p: float


def global_morans_i(x: np.ndarray, W: sparse.csr_matrix) -> MoranResult:
    """Global Moran's I with analytical z under the randomization assumption."""
    n = len(x)
    if n < 4:                       # variance denominator has (n-1)(n-2)(n-3)
        return MoranResult(0.0, 0.0, 0.0, 0.0, 1.0)
    z = x - x.mean()
    Wco = W.tocoo()
    w, wi, wj = Wco.data, Wco.row, Wco.col

    S0 = w.sum()
    num = float((w * z[wi] * z[wj]).sum())
    den = float((z ** 2).sum())
    I = (n / S0) * (num / den) if S0 and den else 0.0
    EI = -1.0 / (n - 1)

    # S1 = 0.5 Σ_ij (w_ij + w_ji)^2 ;  S2 = Σ_i (rowsum_i + colsum_i)^2
    Wsym = (W + W.T).tocoo()
    S1 = 0.5 * float((Wsym.data ** 2).sum())
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    colsum = np.asarray(W.sum(axis=0)).ravel()
    S2 = float(((rowsum + colsum) ** 2).sum())

    b2 = n * float((z ** 4).sum()) / (den ** 2) if den else 0.0
    A = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    B = b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    denom = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
    var = (A - B) / denom - EI ** 2 if denom else 0.0
    zscore = (I - EI) / np.sqrt(var) if var > 0 else 0.0
    p = float(2.0 * norm.sf(abs(zscore)))
    return MoranResult(I=float(I), expected=float(EI), variance=float(var),
                       z=float(zscore), p=p)


def local_morans(x: np.ndarray, Wrs: sparse.csr_matrix,
                 permutations: int = C.MORAN_PERMUTATIONS,
                 seed: int = C.MORAN_SEED) -> dict:
    """Local Moran's I (LISA) with conditional-permutation pseudo-p.

    ``Wrs`` must be row-standardized and self-excluded. Returns per-cell::

        {I, lag, quadrant ('HH'|'HL'|'LH'|'LL'), p (pseudo, two-sided)}

    The permutation holds each cell's own value fixed and reshuffles the rest
    (conditional inference); pseudo-p = (#|I_perm| >= |I_obs| + 1)/(perms + 1).
    """
    n = len(x)
    zc = x - x.mean()
    s2 = float((zc ** 2).sum())
    if s2 == 0 or n < 3:
        zeros = np.zeros(n)
        return {"I": zeros, "lag": zeros, "quadrant": np.array(["LL"] * n),
                "p": np.ones(n)}
    lag = Wrs.dot(zc)
    I = n * zc * lag / s2
    quad = np.where(zc > 0,
                    np.where(lag > 0, "HH", "HL"),
                    np.where(lag > 0, "LH", "LL"))
    absI = np.abs(I)
    rng = np.random.default_rng(seed)
    ge = np.zeros(n)
    for _ in range(permutations):
        zp = rng.permutation(zc)                 # reshuffle neighbour pool
        Iperm = n * zc * Wrs.dot(zp) / s2  # own value zc held fixed
        ge += np.abs(Iperm) >= absI
    pseudo = (ge + 1.0) / (permutations + 1.0)
    return {"I": I, "lag": lag, "quadrant": quad, "p": pseudo}


def mann_kendall(y: np.ndarray, alpha: float = 0.05) -> dict:
    """Mann-Kendall monotonic-trend test on a per-cell time series.

    Used for emerging-hotspot typing. Returns trend label + S, z, p, Kendall tau.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return {"trend": "insufficient", "S": 0, "z": 0.0, "p": 1.0, "tau": 0.0}
    s = 0.0
    for k in range(n - 1):
        s += np.sign(y[k + 1:] - y[k]).sum()
    _, counts = np.unique(y, return_counts=True)
    tie = float((counts * (counts - 1) * (2 * counts + 5)).sum())
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / np.sqrt(var)
    elif s < 0:
        z = (s + 1) / np.sqrt(var)
    else:
        z = 0.0
    p = float(2.0 * norm.sf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    if p < alpha and z > 0:
        trend = "intensifying"
    elif p < alpha and z < 0:
        trend = "diminishing"
    else:
        trend = "stable"
    return {"trend": trend, "S": int(s), "z": float(z), "p": p, "tau": float(tau)}
