"""Fuzzy-logic building blocks: membership functions, fuzzy c-means, and a small Mamdani controller."""
import numpy as np


# ---------- membership functions ----------
def gaussian(x, centre, sigma):
    return np.exp(-((x - centre) ** 2) / (2 * sigma ** 2))


def triangle(x, a, b, c):
    """Triangular membership: 0 at a, 1 at b, 0 at c."""
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b > a else (x >= b).astype(float)
    right = (c - x) / (c - b) if c > b else (x <= b).astype(float)
    return np.clip(np.minimum(left, right), 0, 1)


# ---------- fuzzy c-means ----------
def fuzzy_c_means(X, c, m=2.0, iters=40, seed=0):
    """Soft clustering: every sample belongs to every cluster with a degree in [0,1].
    Returns (centres, U) where U[i,k] is the membership of sample i in cluster k."""
    rng = np.random.default_rng(seed)
    U = rng.random((len(X), c))
    U /= U.sum(1, keepdims=True)
    for _ in range(iters):
        Um = U ** m
        centres = (Um.T @ X) / Um.sum(0)[:, None]
        d = np.linalg.norm(X[:, None, :] - centres[None], axis=2) + 1e-9
        inv = d ** (-2.0 / (m - 1))
        U = inv / inv.sum(1, keepdims=True)
    return centres, U


# ---------- Mamdani controller: "how reliable is this reading?" ----------
# Inputs : strength  = membership of the winning character      (0..1)
#          gap       = strength(winner) - strength(runner-up)   (0..1)
# Output : reliability (0..100 %)
IN_SETS = {"LOW": (-0.4, 0.0, 0.4), "MEDIUM": (0.1, 0.4, 0.7), "HIGH": (0.4, 0.8, 1.2)}
GAP_SETS = {"SMALL": (-0.1, 0.0, 0.1), "MEDIUM": (0.03, 0.1, 0.2), "LARGE": (0.1, 0.25, 1.0)}
OUT_SETS = {"LOW": (-40, 0, 40), "MEDIUM": (20, 50, 80), "HIGH": (60, 100, 140)}
# rule table: (strength, gap) -> reliability
RULES = {("LOW", "SMALL"): "LOW", ("LOW", "MEDIUM"): "LOW", ("LOW", "LARGE"): "MEDIUM",
         ("MEDIUM", "SMALL"): "LOW", ("MEDIUM", "MEDIUM"): "MEDIUM", ("MEDIUM", "LARGE"): "HIGH",
         ("HIGH", "SMALL"): "MEDIUM", ("HIGH", "MEDIUM"): "HIGH", ("HIGH", "LARGE"): "HIGH"}
_Y = np.linspace(0, 100, 201)


def reliability(strength, gap):
    """Mamdani inference: min for AND, max to combine rules, centroid defuzzification."""
    s = {k: float(triangle(strength, *v)) for k, v in IN_SETS.items()}
    g = {k: float(triangle(gap, *v)) for k, v in GAP_SETS.items()}
    agg = np.zeros_like(_Y)
    for (sk, gk), ok in RULES.items():
        fire = min(s[sk], g[gk])                                   # AND = min
        agg = np.maximum(agg, np.minimum(fire, triangle(_Y, *OUT_SETS[ok])))  # clip, then OR = max
    if agg.sum() == 0:
        return 0.0
    return float((agg * _Y).sum() / agg.sum())                     # centroid


def reliability_label(r):
    return "HIGH" if r >= 65 else "MEDIUM" if r >= 35 else "LOW"
