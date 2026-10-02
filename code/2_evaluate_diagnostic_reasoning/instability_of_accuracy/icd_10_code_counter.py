"""
ICD-10 code counter audit -- letter section 2, block (d).

The feature `codes_in_trace_run2` comes from §8 of the analysis record:

    ICD_RE = re.compile(r"\\b([A-Z]\\d{2}(?:\\.\\d+)?)\\b")
    "icd_codes": len(set(ICD_RE.findall(t)))

Two problems to quantify:

  1. FALSE POSITIVES. `[A-Z]\\d{2}` matches any capital letter followed by two
     digits. "B12" matches -- and vitamin B12 deficiency is routine in a
     psychiatric workup. So does any lab or protocol abbreviation of that shape.
     Because the count is a set, one B12 mention adds one "candidate diagnosis".

  2. CONSTRUCT VALIDITY. The count measures code-WRITING, not diagnosis-
     CONSIDERING. Claude writes the longest traces (1088 words) and names the
     fewest codes (2.83); GPT-5.2 writes fewer words (900) and names three times
     as many (8.77). That is more consistent with a formatting difference than
     with a narrower differential. A regex cannot fully settle this, but the
     zero-code share and the words-per-code ratio are informative.

The question this script answers: does the -0.80 model-level ordering survive
restricting to plausible psychiatric diagnosis codes?

Outputs
-------
1. Every distinct matched string with its frequency, flagged F / other / suspect
2. Per-model false-positive share -- the number that matters most
3. Context windows around non-F matches, for eyeballing
4. Lowercase-code check (the regex is case-sensitive and misses "f32.1")
5. Zero-code trace share per model
6. Model-level Spearman recomputed on F-codes only
7. Trace-level mixed model recomputed on F-codes only
"""

import re
import json
import glob
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from itertools import permutations
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- loading
# Trace TEXT is not in trace_comparison_by_case.csv -- that file holds the
# already-computed counts. Per §8's provenance diagram the text lives in the
# raw model output JSON, `model_thoughts` field, run 2 (the generation the
# clinicians rated). Adapt this loader to your file layout.

def load_traces():
    """Return DataFrame with columns: case_id, model, trace_text."""
    rows = []
    folder = Path("/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/code/2_evaluate_diagnostic_reasoning/instability_of_accuracy/pairwise_comparison_of_accuracy_run_and_reasoning_subset/reasoning_traces/reasoning_subset")
    for path in folder.glob("*.json"):        # <-- ADAPT
        with open(path) as fh:
            blob = json.load(fh)
        for rec in (blob if isinstance(blob, list) else [blob]):
            rows.append({
                "case_id": rec["case_id"],                          # <-- ADAPT
                "model": path.stem.split("_")[2],                              # <-- ADAPT
                "trace_text": rec.get("model_thoughts") or "",      # <-- ADAPT
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No traces loaded -- adapt load_traces() to your layout.")
    return df


traces = load_traces()
print(f"loaded {len(traces)} traces across {traces.model.nunique()} models\n")

# ---------------------------------------------------------------- the regex
ICD_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b")          # as used in §8
ICD_RE_LOWER = re.compile(r"\b([a-z]\d{2}(?:\.\d+)?)\b")    # what it misses

# ICD-10 chapter letters that carry psychiatric or neurocognitive diagnoses.
# F = mental and behavioural; G = nervous system (dementias, epilepsy).
PSYCH_LETTERS = {"F"}
CLINICAL_LETTERS = {"F", "G"}

# Shapes that look like codes but are near-certainly not diagnoses in this
# corpus. Extend after reading the frequency table -- do not treat as complete.
KNOWN_NON_CODES = {"B12", "B6", "D3", "T3", "T4", "O2", "CO2", "K10", "Q10"}

# ------------------------------------------------- 1. what is being matched
all_hits = []
for _, row in traces.iterrows():
    for code in set(ICD_RE.findall(row.trace_text)):     # set: mirrors §8
        all_hits.append({"case_id": row.case_id, "model": row.model, "code": code})

hits = pd.DataFrame(all_hits)
hits["letter"] = hits.code.str[0]
hits["is_f"] = hits.letter.isin(PSYCH_LETTERS)
hits["is_clinical"] = hits.letter.isin(CLINICAL_LETTERS)
hits["is_known_non_code"] = hits.code.isin(KNOWN_NON_CODES)

print("=== 1. distinct matched strings, by frequency ===")
freq = hits.code.value_counts()
for code, n in freq.items():
    tag = "F" if code[0] in PSYCH_LETTERS else ("clin" if code[0] in CLINICAL_LETTERS else "OTHER")
    if code in KNOWN_NON_CODES:
        tag += " <-- known non-code"
    print(f"  {code:10s} {n:5d}  [{tag}]")

print(f"\ntotal set-matches: {len(hits)}")
print(f"  F-codes:        {hits.is_f.sum():5d}  ({hits.is_f.mean():.1%})")
print(f"  F or G:         {hits.is_clinical.sum():5d}  ({hits.is_clinical.mean():.1%})")
print(f"  known non-code: {hits.is_known_non_code.sum():5d}  ({hits.is_known_non_code.mean():.1%})")

# ------------------------------------------------- 2. per-model contamination
# THE KEY TABLE. If the false-positive rate is uniform across models it mostly
# adds noise. If it concentrates in one model -- especially GPT-5.2, whose 8.77
# drives the -0.80 -- then the ordering is an artifact.
print("\n=== 2. per-model breakdown (mean distinct codes per trace) ===")
per_model = (hits.groupby(["model", "case_id"])
                 .agg(all_codes=("code", "size"),
                      f_codes=("is_f", "sum"),
                      clin_codes=("is_clinical", "sum"))
                 .groupby("model").mean().round(2))
per_model["non_f_share"] = (1 - per_model.f_codes / per_model.all_codes).round(3)
print(per_model.to_string())
print("\n§8 reported: Claude 2.83 | Gemini 3.97 | DeepSeek 3.50 | GPT-5.2 8.77")
print("If all_codes does not reproduce those, the loader is reading the wrong run.\n")

# ------------------------------------------------- 3. context for non-F hits
print("=== 3. context around non-F matches (sample of 25) ===")
ctx = []
for _, row in traces.iterrows():
    for m in ICD_RE.finditer(row.trace_text):
        if m.group(1)[0] not in PSYCH_LETTERS:
            a, b = max(0, m.start() - 45), min(len(row.trace_text), m.end() + 45)
            ctx.append((row.model, m.group(1),
                        row.trace_text[a:b].replace("\n", " ")))
for model, code, snippet in ctx[:25]:
    print(f"  [{model[:12]:12s}] {code:8s} …{snippet}…")
print(f"  ({len(ctx)} non-F occurrences total)")

# ------------------------------------------------- 4. lowercase misses
lower = sum(len(set(ICD_RE_LOWER.findall(t))) for t in traces.trace_text)
print(f"\n=== 4. lowercase code-shaped strings the regex misses: {lower} ===")
print("If material, the count is also sensitive to each model's capitalisation habits.")

# ------------------------------------------------- 5. zero-code traces
print("\n=== 5. traces naming no codes at all ===")
traces["n_all"] = traces.trace_text.apply(lambda t: len(set(ICD_RE.findall(t))))
traces["n_f"] = traces.trace_text.apply(
    lambda t: len({c for c in set(ICD_RE.findall(t)) if c[0] in PSYCH_LETTERS}))
traces["words"] = traces.trace_text.str.split().str.len()
zero = traces.groupby("model").agg(
    zero_code_share=("n_all", lambda s: (s == 0).mean()),
    zero_f_share=("n_f", lambda s: (s == 0).mean()),
    words_per_code=("words", "mean")).round(3)
zero["words_per_code"] = (zero.words_per_code /
                          traces.groupby("model").n_all.mean()).round(1)
print(zero.to_string())
print("A high zero-code share means the feature is not measuring differential")
print("breadth for that model -- it is measuring whether the model writes codes.\n")

# ------------------------------------------------- 6/7. does -0.80 survive?
r = pd.read_pickle("r.pkl")
name_map = {"Anthropic Claude Opus 4.5": "claude-opus-4-5-20251101",
            "OpenAI GPT-5.2": "gpt-5.2",
            "Google Gemini 3 Pro": "gemini-3-pro-preview",
            "DeepSeek-V3.2": "deepseek-reasoner"}
r["model"] = r["Diagnostician"].map(name_map)
tr = (r.groupby(["Case ID", "model"]).agg(dx=("dx", "mean")).reset_index()
        .rename(columns={"Case ID": "case_id"}))
j = tr.merge(traces[["case_id", "model", "n_all", "n_f", "words"]],
             on=["case_id", "model"], how="inner")
print(f"joined {len(j)} traces to ratings (expect 120)")


def exact_p(rho, n=4):
    base = np.arange(n)
    rhos = [stats.spearmanr(base, p).statistic for p in permutations(base)]
    return float(np.mean(np.abs(np.array(rhos)) >= abs(rho) - 1e-9))


print("\n=== 6. model-level Spearman: original count vs F-only ===")
m = j.groupby("model")[["dx", "n_all", "n_f"]].mean()
for col, label in [("n_all", "all code-shaped"), ("n_f", "F-codes only")]:
    rho = stats.spearmanr(m["dx"], m[col]).statistic
    print(f"  {label:18s} rho={rho:+.2f}  exact p={exact_p(rho):.3f}")

print("\n=== 7. trace-level mixed model: dx ~ codes + C(model) + (1|case_id) ===")
for col, label in [("n_all", "all code-shaped"), ("n_f", "F-codes only")]:
    fit = smf.mixedlm(f"dx ~ {col} + C(model)", data=j, groups=j["case_id"]).fit(reml=True)
    b, se = fit.params[col], fit.bse[col]
    print(f"  {label:18s} {b:+.4f}  95% CI [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}]  p={fit.pvalues[col]:.4f}")

print("""
--- how to read this ---
The letter's claim is that the between-model ordering reflects presentation
style rather than reasoning quality. Three outcomes:

  (a) F-only reproduces rho = -0.80 and the trace-level null holds.
      Nothing changes. Rename the feature "distinct ICD-10 F-codes named"
      and report the F-only figures, since they are the defensible ones.

  (b) F-only weakens or flips the ordering.
      The -0.80 was partly non-diagnostic matches. Say so -- it strengthens
      the section's argument, since the feature is even less of a proxy than
      the letter currently claims.

  (c) Non-F matches concentrate in one model.
      The strongest version of the letter's own point: the count was reading
      that model's habit of writing lab values, not its differential. Report
      the per-model false-positive share directly.

If the counts turn out too noisy to characterise cleanly, drop the code-count
feature from section 2 and rest it on self-correction density (95% CI -0.04 to
+0.03, the cleanest null in the set) and trace length. Neither depends on this
regex.
""")