"""
Trace-level analysis of automated proxy features (letter section 2).  v2

§8 computed  groupby('model').mean()  BEFORE correlating, leaving n = 4.
The per-trace values already exist in trace_comparison_by_case.csv (120 rows).
This script runs the same question at the trace level, using §7's mixed model.

Question: within model and within case, does the feature predict the clinician
rating of THAT trace? If the correlation is only between models, the feature is
a house-style fingerprint, not a quality detector.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from itertools import permutations

# ---------------------------------------------------------------- load
t = pd.read_csv("pairwise_comparison_of_accuracy_run_and_reasoning_subset/trace_comparison_by_case.csv")          # 120 rows
r = pd.read_pickle("r.pkl")                              # 600 ratings

name_map = {"Anthropic Claude Opus 4.5": "Claude Opus 4.5",
            "OpenAI GPT-5.2": "GPT-5.2",
            "Google Gemini 3 Pro": "Gemini 3 Pro",
            "DeepSeek-V3.2": "DeepSeek-V3.2"}
r["model"] = r["Diagnostician"].map(name_map)

# Per-trace clinician rating: mean of the 5 raters. One row per case x model.
tr = (r.groupby(["Case ID", "model"])
        .agg(dx=("dx", "mean"), ext=("ext", "mean"), acc=("correct", "mean"))
        .reset_index().rename(columns={"Case ID": "case_id"}))

j = tr.merge(t, on=["case_id", "model"], how="inner")
print(f"joined: {len(j)} traces (expect 120)")
assert len(j) == 120, "join key mismatch -- check case_id dtype and model naming"

# Length-normalised variants: delib_run2 is a raw phrase count, so longer
# traces score higher mechanically. Normalising removes that confound.
j["codes_per_1k"] = j["codes_in_trace_run2"] / j["words_run2"] * 1000
j["delib_per_1k"] = j["delib_run2"] / j["words_run2"] * 1000
j["w100"] = j["words_run2"] / 100

FEATURES = ["codes_in_trace_run2", "delib_run2",
            "codes_per_1k", "delib_per_1k", "w100"]

# ------------------------------------------------- exact Spearman p, n = 4
def exact_spearman_p(rho, n=4):
    """Two-sided permutation p. scipy's default t-approximation is
    anti-conservative at n = 4 (returns 0.200 where the exact value is 0.333)."""
    base = np.arange(n)
    rhos = [stats.spearmanr(base, p).statistic for p in permutations(base)]
    return float(np.mean(np.abs(np.array(rhos)) >= abs(rho) - 1e-9))

print("\n--- §8 model-level rho, with exact p ---")
m = j.groupby("model")[["dx"] + FEATURES].mean()
for f in FEATURES:
    rho = stats.spearmanr(m["dx"], m[f]).statistic
    print(f"{f:22s} rho={rho:+.2f}  exact p={exact_spearman_p(rho):.3f}  n=4")

# ------------------------------------------------- (a) within-model Spearman
print("\n--- within-model Spearman (feature vs dx), n=30 per model ---")
for f in FEATURES:
    cells = []
    for mod, g in j.groupby("model"):
        rho, p = stats.spearmanr(g[f], g["dx"])
        cells.append(f"{mod.split()[0]}: {rho:+.2f}(p={p:.2f})")
    print(f"{f:22s} " + "  ".join(cells))

# ------------------------------------------------- (b) mixed model, per §7
print("\n--- mixed model: dx ~ feature + C(model) + (1|case_id) ---")
slopes = {}
for f in FEATURES:
    fit = smf.mixedlm(f"dx ~ {f} + C(model)", data=j, groups=j["case_id"]).fit(reml=True)
    b, se, p = fit.params[f], fit.bse[f], fit.pvalues[f]
    slopes[f] = b
    print(f"{f:22s} {b:+.4f}  95% CI [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}]  p={p:.4f}")

# ------------------------------------------------- (c) explanatory ceiling
# §7's logic: between-model spread in the feature x within-trace slope,
# against the observed 0.48-1.23 point rating gaps.
print("\n--- explanatory ceiling (cf. §7: 188 words -> +0.08 vs observed +1.23) ---")
for f in FEATURES:
    spread = m[f].max() - m[f].min()
    print(f"{f:22s} spread {spread:8.2f} -> predicts {slopes[f]*spread:+.3f} rating points")

# ------------------------------------------------- (d) ICD regex audit
# ICD_RE = r"\b([A-Z]\d{2}(?:\.\d+)?)\b" matches any capital letter + 2 digits,
# not just F-codes. "B12" (vitamin B12 deficiency -- routine in a psychiatric
# workup) matches. Quantify before relying on the count.
print("\n--- what is the code counter actually counting? ---")
print("Re-run trace_features() capturing the matched strings, then:")
print("  pd.Series(all_matches).value_counts().head(30)")
print("  share of matches with a leading F: <fill in>")
