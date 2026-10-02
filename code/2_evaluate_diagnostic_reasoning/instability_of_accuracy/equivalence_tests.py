"""
Equivalence testing (TOST) on the paired accuracy comparisons -- letter section 1.

The superiority tests in §4 established that accuracy separated 0 of 6 pairs.
That is a failure to reject, not evidence of equivalence. TOST turns the
defensible subset of it into a positive claim.

Reports, for each pair, the SMALLEST margin at which equivalence holds --
avoiding the usual criticism that the margin was chosen to fit the result.
Readers can then apply whatever threshold they consider clinically meaningful.

Equivalence at margin D holds iff the 90% CI lies entirely within (-D, +D).
That is algebraically identical to two one-sided t-tests at alpha = 0.05.
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 10_000
RNG = np.random.default_rng(20260814)

r = pd.read_pickle("r.pkl")

# One accuracy value per case x model: the mean of the 5 raters' correctness
# calls. Same unit of analysis as the superiority tests in §4.
tr = (r.groupby(["Case ID", "Diagnostician"])
        .agg(acc=("correct", "mean"), dx=("dx", "mean")).reset_index())
w = tr.pivot(index="Case ID", columns="Diagnostician", values="acc")
print(f"cases: {len(w)}   models: {list(w.columns)}")

rows = []
for a, b in itertools.combinations(w.columns, 2):
    d = (w[a] - w[b]).dropna()
    n = len(d)
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)

    t95 = stats.t.ppf(0.975, n - 1)          # superiority interval
    t90 = stats.t.ppf(0.95, n - 1)           # equivalence interval

    # Smallest margin at which TOST passes.
    bound = abs(mean) + t90 * se

    # Bootstrap check: paired differences are multiples of 0.2, so the
    # t-interval's normality assumption is worth verifying.
    boot = np.array([RNG.choice(d.values, n, replace=True).mean()
                     for _ in range(N_BOOT)])
    blo, bhi = np.percentile(boot, [5, 95])
    boot_bound = max(abs(blo), abs(bhi))

    rows.append({
        "pair": f"{a} vs {b}", "n": n, "delta": mean,
        "ci95_lo": mean - t95 * se, "ci95_hi": mean + t95 * se,
        "ci90_lo": mean - t90 * se, "ci90_hi": mean + t90 * se,
        "equiv_bound": bound, "equiv_bound_boot": boot_bound,
    })

res = pd.DataFrame(rows).sort_values("equiv_bound")
pd.set_option("display.width", 200)
print("\n--- equivalence bounds (smallest margin at which TOST passes) ---")
print(res.round(4).to_string(index=False))

# Holm adjustment, for symmetry with the superiority tests. The TOST p-value is
# the LARGER of the two one-sided p-values; recompute it at each candidate
# margin, since TOST significance is margin-dependent.
print("\n--- Holm-adjusted TOST at candidate margins ---")
for D in (0.05, 0.075, 0.10, 0.15):
    ps = []
    for _, row in res.iterrows():
        n = row["n"]
        se = (row["ci95_hi"] - row["delta"]) / stats.t.ppf(0.975, n - 1)
        p_lo = stats.t.sf((row["delta"] + D) / se, n - 1)      # H0: delta <= -D
        p_hi = stats.t.cdf((row["delta"] - D) / se, n - 1)     # H0: delta >= +D
        ps.append(max(p_lo, p_hi))
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(ps) - i) * ps[idx])
        adj[idx] = min(run, 1.0)
    passed = [res.iloc[i]["pair"] for i in range(len(ps)) if adj[i] < 0.05]
    print(f"margin +/-{D:.3f}: {len(passed)}/6 equivalent -> {passed}")

# Sensitivity: restrict to the 18 cases where all 5 raters agreed unanimously
# on all 4 models (§5). Rules out adjudication noise as the source of the null.
g = r.groupby(["Case ID", "Diagnostician"])["correct"].agg(["sum", "size"]).reset_index()
g["unanimous"] = (g["sum"] == 0) | (g["sum"] == g["size"])
clean = g.groupby("Case ID")["unanimous"].all()
clean = clean[clean].index
print(f"\n--- cleanly adjudicated subset: {len(clean)} cases ---")
wc = (g[g["Case ID"].isin(clean)]
      .assign(acc=lambda x: (x["sum"] == x["size"]).astype(int))
      .pivot(index="Case ID", columns="Diagnostician", values="acc"))
for a, b in itertools.combinations(wc.columns, 2):
    d = (wc[a] - wc[b]).dropna()
    if d.std(ddof=1) == 0:
        print(f"{a} vs {b}: identical on all {len(d)} cases")
        continue
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{a} vs {b}: delta={d.mean():+.3f}  equiv bound="
          f"{abs(d.mean()) + stats.t.ppf(0.95, len(d)-1) * se:.3f}")
