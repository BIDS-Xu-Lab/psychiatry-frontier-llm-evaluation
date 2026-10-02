"""
rescore_dropped_preambles.py — re-rank existing adjudication logs after removing
non-diagnosis lines (preambles, blank lines) from model differentials.

No API calls: every verdict is taken from the *_comparisons.csv logs. Only cases
whose differential had more than 5 lines are touched; a line is treated as a
non-diagnosis line if it is blank or ends with ":" (e.g. "Based on ... here are
the top 5 most likely diagnoses:"). A case is only re-scored if exactly 5 lines
remain; otherwise it is left as is and reported.

Usage
  python rescore_dropped_preambles.py --indir adjudication_logs --outdir adjudication_logs_rescored
Writes, for every run, a *_cases.csv (same columns) and *_comparisons.csv (dropped
lines removed, ranks renumbered) into --outdir, plus rescore_report.csv.
"""
import argparse, glob, json, os
import pandas as pd


def is_non_diagnosis(p):
    p = "" if pd.isna(p) else str(p).strip()
    return p == "" or p.endswith(":")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    report = []
    for cf in sorted(glob.glob(os.path.join(a.indir, "*_cases.csv"))):
        tag = os.path.basename(cf)[: -len("_cases.csv")]
        kf = os.path.join(a.indir, f"{tag}_comparisons.csv")
        cases = pd.read_csv(cf)
        comps = pd.read_csv(kf)
        cases["n_unnumbered_lines_dropped"] = 0

        for i, row in cases[cases["n_preds"] > 5].iterrows():
            preds = json.loads(row["y_pred"])
            keep = [j for j, p in enumerate(preds, 1) if not is_non_diagnosis(p)]   # 1-based ranks
            if len(keep) != 5:
                report.append({"run": tag, "case_id": row.case_id, "status": f"left unchanged ({len(keep)} lines kept)"})
                continue
            new_rank = {old: new for new, old in enumerate(keep, 1)}
            sel = comps["case_id"] == row["case_id"]
            c = comps[sel].copy()
            c = c[c["rank"].isin(keep)]
            c["rank"] = c["rank"].map(new_rank)
            comps = pd.concat([comps[~sel], c], ignore_index=True)

            hit = c[c["verdict"] == True]                                         # noqa: E712
            n_refs = int(row["n_refs"])
            first = hit["rank"].min() if len(hit) else None
            first_tier = (hit.sort_values("rank").iloc[0]["tier"] if len(hit) else None)
            cases.loc[i, ["n_preds", "top1", "hit_rate", "recall", "mrr", "first_match_rank",
                          "first_match_tier", "n_llm_errors", "flag_not_five_preds",
                          "n_unnumbered_lines_dropped", "y_pred"]] = [
                5, 1.0 if first == 1 else 0.0, 1.0 if len(hit) else 0.0,
                hit["ref_index"].nunique() / n_refs, (1 / first) if first else 0.0,
                first, first_tier, int((c["tier"] == "LLM_ERROR").sum()), False,
                len(preds) - 5, json.dumps([preds[j - 1] for j in keep])]
            report.append({"run": tag, "case_id": row.case_id,
                           "status": f"re-scored: first match rank {row.first_match_rank} -> {first}, "
                                     f"top1 {row.top1} -> {1.0 if first == 1 else 0.0}"})

        cases.to_csv(os.path.join(a.outdir, f"{tag}_cases.csv"), index=False)
        comps.sort_values(["case_id", "rank", "ref_index"]).to_csv(
            os.path.join(a.outdir, f"{tag}_comparisons.csv"), index=False)
        meta = os.path.join(a.indir, f"{tag}_meta.json")
        if os.path.exists(meta):
            m = json.load(open(meta))
            m["rescored_dropped_preambles"] = True
            m["summary"] = cases[["top1", "hit_rate", "recall", "mrr"]].mean().round(4).to_dict()
            json.dump(m, open(os.path.join(a.outdir, f"{tag}_meta.json"), "w"), indent=2)

    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(a.outdir, "rescore_report.csv"), index=False)
    print(rep.to_string(index=False) if len(rep) else "No cases with more than 5 lines; nothing changed.")


if __name__ == "__main__":
    main()