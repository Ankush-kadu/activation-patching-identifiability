"""Null calibration for the held-out response-ratio profile.

Added 2026-08-27.  The first version of invariant.py compared the EXTREME of each trial's
profile (its min and max over the p train eigenvectors) against a POINTWISE envelope: the
2.5-97.5 percentiles of all null profile values pooled over eigen-indices.  That is an
extreme-vs-pointwise mismatch -- the maximum of p values is not distributed like a typical
value -- and it made "escapes at both ends" anti-conservative.  Three corrections:

  extreme_envelope   compare an extreme to the distribution of extremes: the 2.5th percentile
                     of the null MINIMA and the 97.5th percentile of the null MAXIMA.
  loo_envelope       build each trial's envelope from the OTHER trials' null replicates, so a
                     trial is never scored against a threshold its own nulls helped set.
  escape_binom_p     with a 2.5% nominal per-end escape rate, k escapes out of n trials is a
                     one-sided binomial tail -- the per-cell p-value.
  holm               Holm step-down adjustment across the model x task cells.

A label-permutation test was tried and dropped: permuting S/J labels destroys the support
difference that the design creates on purpose, so it tests "are these two point sets
distinguishable at all" (near-tautological here) rather than "does the procedure manufacture
extremes", which is what the same-family null answers.
"""
import numpy as np
from scipy.stats import binomtest

NOMINAL = 0.025          # per-end escape rate of a 2.5 / 97.5 extreme-value envelope


def extreme_envelope(null_profiles, lo_pct=2.5, hi_pct=97.5):
    """null_profiles: list of held-out profiles (one per null replicate)."""
    mins = np.array([min(p) for p in null_profiles])
    maxs = np.array([max(p) for p in null_profiles])
    return float(np.percentile(mins, lo_pct)), float(np.percentile(maxs, hi_pct))


def loo_envelope(nulls_per_trial, i):
    """Envelope for trial i built from every OTHER trial's null replicates."""
    others = [p for j, ps in enumerate(nulls_per_trial) if j != i for p in ps]
    return extreme_envelope(others if others else nulls_per_trial[i])


def escape_binom_p(k, n, p=NOMINAL):
    """One-sided binomial tail: P(at least k escapes | nominal rate p)."""
    if n == 0:
        return float("nan")
    return float(binomtest(int(k), int(n), p, alternative="greater").pvalue)


def holm(pvals):
    """Holm step-down adjusted p-values (family-wise error control), input order preserved."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
