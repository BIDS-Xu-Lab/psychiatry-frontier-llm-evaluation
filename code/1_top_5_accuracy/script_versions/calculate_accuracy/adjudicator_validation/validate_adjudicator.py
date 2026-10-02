"""
validate_adjudicator.py — Editorial point 4 / R2-7b.

Compares GPT-5-mini adjudication with the clinicians. --metric chooses which
adjudicator verdict is compared: top1 (rank-1 prediction matches; closest to the
Task 2 instruction "captured at least the primary diagnosis") or hit_rate (top-5) with the five clinicians on the 120 rated vignette-model pairs.

Inputs
  --ratings     processed_full_list.csv (one row per rater x case x model;
                columns case_id, model_name, annotator, diagnosis_match, model_diagnosis)
  --adjudicator one or more *_cases.csv files from evaluate_accuracy_logged.py,
                run on the SAME outputs the clinicians rated (run 2). Use the
                exact model_name strings from the ratings file as --model-label.
  --canonical   "threshold:pass" used for the headline numbers, e.g. 90:1
  --outdir      where tables go

Outputs (all CSV): provenance_check, agreement_by_setting, agreement_by_model,
agreement_by_unanimity, loo_envelope, model_level_accuracy, self_consistency,
disagreements_for_review. Prints a short summary.
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

RNG = np.random.default_rng(20260924)
B = 5000


# ------------------------------------------------------------------ helpers
def parse_preds(s):
    if not isinstance(s, str):
        return []
    out = []
    for line in s.strip().split("\n"):
        line = line.strip()                      # ratings file indents lines 2-5
        m = re.match(r"\d+\.\s+(.*)", line)
        out.append(m.group(1).strip() if m else line)
    return out


def agreement_stats(ref, adj):
    """ref, adj: 0/1 arrays. Reference = clinicians."""
    ref, adj = np.asarray(ref), np.asarray(adj)
    a = int(((ref == 1) & (adj == 1)).sum())   # both correct
    b = int(((ref == 1) & (adj == 0)).sum())   # clinicians yes, adjudicator no
    c = int(((ref == 0) & (adj == 1)).sum())   # clinicians no, adjudicator yes
    d = int(((ref == 0) & (adj == 0)).sum())
    n = a + b + c + d
    po = (a + d) / n
    p_ref, p_adj = (a + b) / n, (a + c) / n
    pe = p_ref * p_adj + (1 - p_ref) * (1 - p_adj)
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    mcnemar_p = stats.binomtest(c, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return {"n": n, "both_yes": a, "clin_yes_adj_no": b, "clin_no_adj_yes": c, "both_no": d,
            "agreement": po, "kappa": kappa, "pabak": 2 * po - 1,
            "sensitivity": a / (a + b) if a + b else np.nan,
            "specificity": d / (c + d) if c + d else np.nan,
            "ppv": a / (a + c) if a + c else np.nan,
            "npv": d / (b + d) if b + d else np.nan,
            "clin_rate": p_ref, "adj_rate": p_adj,
            "mcnemar_exact_p": mcnemar_p}


def cluster_boot(df, fn, keys=("agreement", "kappa", "pabak", "sensitivity", "specificity"), desc="bootstrap"):
    """Percentile CIs, resampling vignettes (cases) with replacement."""
    cases = df["case_id"].unique()
    groups = {c: g for c, g in df.groupby("case_id")}
    draws = {k: [] for k in keys}
    for _ in tqdm(range(B), desc=desc, leave=False):
        samp = pd.concat([groups[c] for c in RNG.choice(cases, len(cases), replace=True)])
        s = fn(samp)
        for k in keys:
            draws[k].append(s[k])
    return {f"{k}_lo": np.nanpercentile(v, 2.5) for k, v in draws.items()} | \
           {f"{k}_hi": np.nanpercentile(v, 97.5) for k, v in draws.items()}


def summarise(df, ref_col, adj_col="adj", desc="bootstrap"):
    fn = lambda x: agreement_stats(x[ref_col], x[adj_col])
    return {**fn(df), **cluster_boot(df, fn, desc=desc)}


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--adjudicator", nargs="+", required=True)
    ap.add_argument("--canonical", default="90:1")
    ap.add_argument("--outdir", default="adjudicator_validation")
    ap.add_argument("--metric", choices=["top1", "hit_rate"], default="top1",
                    help="adjudicator verdict to compare with clinicians (default top1)")
    ap.add_argument("--rename", nargs="*", default=[],
                    help='map adjudicator labels to ratings model_name, e.g. "GPT-5.2=OpenAI GPT-5.2"')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda name, df: df.to_csv(os.path.join(a.outdir, f"{name}.csv"), index=False)

    # ---- clinicians ----------------------------------------------------
    r = pd.read_csv(a.ratings)
    r["yes"] = (r["diagnosis_match"].astype(str).str.strip().str.lower() == "yes").astype(int)
    assert len(r) == 600 and r.groupby(["case_id", "model_name"]).size().eq(5).all(), \
        "expected 5 raters x 120 pairs"
    pairs = (r.groupby(["case_id", "model_name"])
               .agg(n_yes=("yes", "sum"), clin_mean=("yes", "mean")).reset_index())
    pairs["clin_majority"] = (pairs["n_yes"] >= 3).astype(int)
    pairs["unanimous"] = pairs["n_yes"].isin([0, 5])
    raters = sorted(r["annotator"].unique())
    wide = r.pivot_table(index=["case_id", "model_name"], columns="annotator", values="yes").reset_index()

    # ---- adjudicator -----------------------------------------------------
    adj = pd.concat([pd.read_csv(f) for f in a.adjudicator], ignore_index=True)
    adj = adj.rename(columns={"model": "model_name"})
    mapping = dict(x.split("=", 1) for x in a.rename)
    if mapping:
        adj["model_name"] = adj["model_name"].replace(mapping)
        print("[rename]", "; ".join(f"{k} -> {v}" for k, v in mapping.items()))
    only_adj = sorted(set(adj.model_name) - set(pairs.model_name))
    only_rat = sorted(set(pairs.model_name) - set(adj.model_name))
    if only_adj or only_rat:
        raise SystemExit("Model names don't match.\n"
                         f"  only in adjudicator files: {only_adj}\n"
                         f"  only in ratings file:      {only_rat}\n"
                         'Fix with --rename "adjudicator label=ratings name" (one per model).')
    adj["setting"] = adj["fuzzy_threshold"].astype(str) + ":" + adj["pass_id"].astype(str)
    missing = set(map(tuple, pairs[["case_id", "model_name"]].values)) - \
              set(map(tuple, adj[adj.setting == a.canonical][["case_id", "model_name"]].values))
    assert not missing, f"canonical setting lacks {len(missing)} rated pairs, e.g. {sorted(missing)[:3]} " \
                        "(check --model-label strings match model_name)"

    # ---- 1. provenance: did the adjudicator score what clinicians saw? -----
    canon = adj[adj.setting == a.canonical]
    prov = r.merge(canon[["case_id", "model_name", "y_pred"]], on=["case_id", "model_name"])
    prov["clin_list"] = prov["model_diagnosis"].apply(parse_preds)
    prov["adj_list"] = prov["y_pred"].apply(json.loads)
    norm = lambda l: [re.sub(r"\s+", " ", x).strip().lower() for x in l]
    prov["same_output"] = [norm(x) == norm(y) for x, y in zip(prov.clin_list, prov.adj_list)]
    out("provenance_check", prov.loc[~prov.same_output,
        ["case_id", "model_name", "annotator", "model_diagnosis", "y_pred"]])
    n_bad = int((~prov.same_output).sum())
    print(f"[provenance] rater rows whose differential differs from the adjudicated one: {n_bad} / {len(prov)}")
    if n_bad > 30:
        print("  !! Most rows differ — the adjudicator file is probably from the wrong generation run.")

    # ---- 2. agreement, every setting, vs majority and per-rater pooled -------
    rows = []
    for s, g in adj.groupby("setting"):
        m = pairs.merge(g[["case_id", "model_name", a.metric]].rename(columns={a.metric: "adj"}),
                        on=["case_id", "model_name"])
        if len(m) != 120:
            continue
        m["adj"] = m["adj"].astype(int)
        rows.append({"setting": s, "reference": "majority (>=3/5)",
                     **summarise(m, "clin_majority", desc=f"bootstrap: setting {s}")})
    by_setting = pd.DataFrame(rows)
    out("agreement_by_setting", by_setting)

    m = pairs.merge(canon[["case_id", "model_name", a.metric, "first_match_rank", "first_match_tier",
                           "n_llm_errors", "y_true", "y_pred"]].rename(columns={a.metric: "adj"}),
                    on=["case_id", "model_name"])
    m["adj"] = m["adj"].astype(int)

    # ---- 3. stratified (canonical setting) ----------------------------------
    out("agreement_by_model", pd.DataFrame(
        [{"model_name": k, **summarise(g, "clin_majority", desc=f"bootstrap: {k}")}
         for k, g in m.groupby("model_name")]))
    out("agreement_by_unanimity", pd.DataFrame(
        [{"unanimous": k, **agreement_stats(g["clin_majority"], g["adj"])} for k, g in m.groupby("unanimous")]))

    # ---- 4. leave-one-out envelope ------------------------------------------
    w = wide.merge(m[["case_id", "model_name", "adj"]], on=["case_id", "model_name"])
    loo = []
    for rr in raters:
        others = [x for x in raters if x != rr]
        votes = w[others].sum(axis=1)
        decided = votes != 2                       # 2-2 ties among four: excluded
        ref = (votes[decided] >= 3).astype(int)
        loo.append({"held_out": rr, "n_decided": int(decided.sum()), "n_ties": int((~decided).sum()),
                    "rater_agreement": (w.loc[decided, rr].astype(int) == ref).mean(),
                    "adjudicator_agreement": (w.loc[decided, "adj"] == ref).mean()})
    loo = pd.DataFrame(loo)
    out("loo_envelope", loo)

    # ---- 5. model-level accuracy: does the adjudicator shift a model? ---------
    ml = m.groupby("model_name").agg(clin_rater_mean=("clin_mean", "mean"),
                                     clin_majority=("clin_majority", "mean"),
                                     adjudicator=("adj", "mean")).reset_index()
    ml["adj_minus_rater_mean"] = ml["adjudicator"] - ml["clin_rater_mean"]
    ml["adj_minus_majority"] = ml["adjudicator"] - ml["clin_majority"]
    out("model_level_accuracy", ml)

    # ---- 6. self-consistency across passes (same threshold) -------------------
    sc = []
    for thr, g in adj.groupby("fuzzy_threshold"):
        piv = g.pivot_table(index=["case_id", "model_name"], columns="pass_id", values=a.metric)
        if piv.shape[1] < 2:
            continue
        sc.append({"fuzzy_threshold": thr, "n_passes": piv.shape[1],
                   "pairs_with_any_flip": int((piv.nunique(axis=1) > 1).sum()),
                   "n_pairs": len(piv)})
    out("self_consistency", pd.DataFrame(sc))

    # ---- 7. disagreement sheet for review ------------------------------------
    print(f"[metric] adjudicator verdict compared: {a.metric}")
    dis = m[m["adj"] != m["clin_majority"]].copy()
    dis["direction"] = np.where(dis["adj"] == 1, "adjudicator YES, clinicians NO",
                                "adjudicator NO, clinicians YES")
    dis["coder_extraction_or_matching"] = ""   # EXTRACTION_MISS / MATCHING_JUDGMENT
    dis["coder_category"] = ""
    dis["coder_notes"] = ""
    out("disagreements_for_review", dis[["case_id", "model_name", "n_yes", "unanimous", "direction",
        "first_match_rank", "first_match_tier", "n_llm_errors", "y_true", "y_pred",
        "coder_extraction_or_matching", "coder_category", "coder_notes"]])

    # ---- summary ---------------------------------------------------------------
    h = by_setting[by_setting.setting == a.canonical].iloc[0]
    print(f"\n[headline, {a.canonical}] agreement {h.agreement:.3f} "
          f"({h.agreement_lo:.3f}-{h.agreement_hi:.3f}); kappa {h.kappa:.3f}; PABAK {h.pabak:.3f}")
    print(f"  discordant: adj YES/clin NO = {h.clin_no_adj_yes}, adj NO/clin YES = {h.clin_yes_adj_no}; "
          f"McNemar exact p = {h.mcnemar_exact_p:.3f}")
    print(f"[LOO] raters {loo.rater_agreement.min():.3f}-{loo.rater_agreement.max():.3f}; "
          f"adjudicator {loo.adjudicator_agreement.min():.3f}-{loo.adjudicator_agreement.max():.3f}")
    print(ml.round(3).to_string(index=False))
    print(f"[errors] canonical pairs with any LLM error: {int((m.n_llm_errors > 0).sum())}")


if __name__ == "__main__":
    main()