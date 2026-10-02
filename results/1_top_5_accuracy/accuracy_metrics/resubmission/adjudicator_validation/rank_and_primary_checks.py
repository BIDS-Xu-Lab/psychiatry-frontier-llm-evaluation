"""
rank_and_primary_checks.py — two follow-ups to the adjudicator validation.

1. Rank breakdown: clinician "yes" rate by where the adjudicator found a match
   (rank 1 / rank 2+ / no match), using (a) any reference and (b) the primary
   (first-listed) reference only.
2. Primary-reference sensitivity: agreement with the clinician majority when the
   adjudicator verdict counts only matches to the primary reference, at top-1 and
   top-5, next to the any-reference versions used so far.

No API calls: verdicts come from the *_comparisons.csv logs (use the rescored folder).
ref_index 0 is the first reference listed for the vignette; for references written
as alternatives ("X or Y", "X vs Y") that is the first alternative.

Usage
  python rank_and_primary_checks.py --ratings processed_full_list.csv \
      --adjdir adjudication_logs_rescored --setting 90:1 --outdir rank_primary_checks
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy import stats

RENAME = {"Claude Opus 4.5": "Anthropic Claude Opus 4.5", "GPT-5.2": "OpenAI GPT-5.2"}
B = 5000
RNG = np.random.default_rng(20261002)


def stats2x2(ref, adj):
    ref, adj = np.asarray(ref).astype(int), np.asarray(adj).astype(int)
    a = ((ref == 1) & (adj == 1)).sum(); b = ((ref == 1) & (adj == 0)).sum()
    c = ((ref == 0) & (adj == 1)).sum(); d = ((ref == 0) & (adj == 0)).sum()
    n = a + b + c + d; po = (a + d) / n
    pr, pa = (a + b) / n, (a + c) / n
    pe = pr * pa + (1 - pr) * (1 - pa)
    return {"agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan, "pabak": 2 * po - 1,
            "clin_yes_adj_no": int(b), "clin_no_adj_yes": int(c),
            "mcnemar_p": stats.binomtest(int(c), int(b + c), 0.5).pvalue if b + c else 1.0}


def boot_ci(df, col):
    cases = df.case_id.unique(); groups = {k: g for k, g in df.groupby("case_id")}
    vals = []
    for _ in range(B):
        s = pd.concat([groups[k] for k in RNG.choice(cases, len(cases), replace=True)])
        vals.append(((s[col] == s.clin_majority)).mean())
    return np.percentile(vals, [2.5, 97.5])


def where(rank):
    return "no match" if pd.isna(rank) else ("rank 1" if rank == 1 else "rank 2+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--adjdir", required=True)
    ap.add_argument("--setting", default="90:1", help="threshold:pass, e.g. 90:1 or off:1")
    ap.add_argument("--outdir", default="rank_primary_checks")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    thr, pas = a.setting.split(":")

    # clinicians
    r = pd.read_csv(a.ratings)
    r["yes"] = (r.diagnosis_match.astype(str).str.strip().str.lower() == "yes").astype(int)
    pairs = r.groupby(["case_id", "model_name"]).yes.agg(n_yes="sum", clin_yes_rate="mean").reset_index()
    pairs["clin_majority"] = (pairs.n_yes >= 3).astype(int)

    # adjudicator comparisons for the chosen setting
    files = glob.glob(os.path.join(a.adjdir, f"*_thr{thr}_pass{pas}_comparisons.csv"))
    assert files, f"no *_thr{thr}_pass{pas}_comparisons.csv in {a.adjdir}"
    c = pd.concat(pd.read_csv(f) for f in files)
    c["model_name"] = c["model"].replace(RENAME)
    c["hit"] = c["verdict"] == True                                            # noqa: E712
    hits = c[c.hit]
    first_any = hits.groupby(["case_id", "model_name"])["rank"].min().rename("rank_any")
    first_pri = hits[hits.ref_index == 0].groupby(["case_id", "model_name"])["rank"].min().rename("rank_primary")
    m = pairs.merge(first_any, on=["case_id", "model_name"], how="left") \
             .merge(first_pri, on=["case_id", "model_name"], how="left")
    assert len(m) == 120, f"expected 120 pairs, got {len(m)}"

    m["top1_any"] = (m.rank_any == 1).astype(int)
    m["top1_primary"] = (m.rank_primary == 1).astype(int)
    m["top5_any"] = m.rank_any.notna().astype(int)
    m["top5_primary"] = m.rank_primary.notna().astype(int)

    # 1. rank breakdown
    print(f"=== 1. Rank breakdown (setting {a.setting}) ===")
    for col, label in [("rank_any", "any reference"), ("rank_primary", "primary reference only")]:
        t = m.assign(where=m[col].map(where)).groupby("where").agg(
            pairs=("clin_yes_rate", "size"), clinician_yes_rate=("clin_yes_rate", "mean"),
            majority_yes=("clin_majority", "mean")).reindex(["rank 1", "rank 2+", "no match"]).round(3)
        print(f"\n-- match to {label} --\n{t.to_string()}")
        t.to_csv(os.path.join(a.outdir, f"rank_breakdown_{col}.csv"))

    # 2. agreement under each definition
    print("\n=== 2. Agreement with clinician majority ===")
    rows = []
    for col in ["top1_any", "top1_primary", "top5_any", "top5_primary"]:
        s = stats2x2(m.clin_majority, m[col]); lo, hi = boot_ci(m, col)
        rows.append({"adjudicator_measure": col, **s, "agree_lo": lo, "agree_hi": hi})
    agr = pd.DataFrame(rows)
    print(agr.round(3).to_string(index=False))
    agr.to_csv(os.path.join(a.outdir, "agreement_by_definition.csv"), index=False)

    # per-model accuracy under each definition
    pm = m.groupby("model_name")[["clin_yes_rate", "clin_majority", "top1_any", "top1_primary"]].mean().round(3)
    pm.columns = ["clin_rater_mean", "clin_majority", "adj_top1_any", "adj_top1_primary"]
    print(f"\n-- per model --\n{pm.to_string()}")
    pm.to_csv(os.path.join(a.outdir, "per_model_top1.csv"))

    # pairs where the primary-only rule changes the top-1 verdict
    ch = m[m.top1_any != m.top1_primary][["case_id", "model_name", "n_yes", "clin_majority",
                                          "rank_any", "rank_primary", "top1_any", "top1_primary"]]
    print(f"\n-- pairs whose top-1 verdict changes under primary-only ({len(ch)}) --")
    print(ch.to_string(index=False) if len(ch) else "none")
    ch.to_csv(os.path.join(a.outdir, "changed_pairs.csv"), index=False)


if __name__ == "__main__":
    main()