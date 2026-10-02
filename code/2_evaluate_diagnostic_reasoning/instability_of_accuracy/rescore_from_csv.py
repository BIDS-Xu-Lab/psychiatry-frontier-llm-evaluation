#!/usr/bin/env python3
"""
Run-to-run stability rescoring from adjudication_pairs.csv, using the study's
OWN evaluation logic (rapidfuzz token_set_ratio >= 90, then gpt-5-mini).

Reads the CSV emitted by run_comparison.py rather than the raw JSON, so no
re-parsing of model outputs is needed.

    model, run, case_id, rank, predicted_diagnosis, predicted_code, reference_diagnosis

Metric definitions match evaluate_accuracy.py exactly:
    top1      1.0 if the rank-1 prediction matched ANY reference diagnosis
    hit_rate  1.0 if any prediction matched any reference diagnosis  ("top-5")
    recall    fraction of reference diagnoses matched by some prediction
    mrr       1 / rank of first matching prediction

USAGE
    python rescore_from_csv.py --pairs adjudication_pairs.csv --outdir out/
    python rescore_from_csv.py --pairs adjudication_pairs.csv --fuzzy-threshold 100
    python rescore_from_csv.py --pairs adjudication_pairs.csv --limit 40   # dry run

OUTPUTS
    run_stability_scored.csv     per model/case, both runs, all four metrics
    run_stability_summary.txt    flip rates and adjudication-tier accounting
    flagged_fuzzy_matches.csv    fuzzy auto-matches that bypassed the LLM
    unscorable_cases.csv         case/run combinations absent from the CSV
"""

import re, os, json, csv, argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    base_url="https://ai-kwj9-llm-evaluation.openai.azure.com/openai/v1/",
    api_key=os.environ.get("OPENAI_FOUNDRY_API_KEY"),
)

# ---------------------------------------------------------------------------
# Reference parsing — unchanged from evaluate_accuracy.py
# ---------------------------------------------------------------------------

def parse_ground_truth_diagnoses(diagnosis_str) -> list:
    if not isinstance(diagnosis_str, str):
        return []
    s = diagnosis_str.replace("\\n", "\n")
    if re.search(r"^\d+\.", s.strip()):
        out = []
        for line in s.strip().split("\n"):
            m = re.match(r"\d+\.\s+(.*)", line)
            out.append(m.group(1).strip() if m else line.strip())
        return [d for d in out if d]
    return [d.strip() for d in re.split(r";", s) if d.strip()]


def rebuild_prediction(diagnosis, code, include_code=True):
    """
    evaluate_accuracy.py passed the whole line after the numbering, e.g.
    'Fetishistic Disorder - ICD-10 F65.0'. The CSV splits text and code, so
    rejoin them to give the matcher the same input the original run saw.
    The adjudicator prompt explicitly refers to the ICD-10 codes.
    """
    d = str(diagnosis).strip()
    c = str(code).strip() if code is not None else ""
    if include_code and c and c.lower() != "nan":
        return f"{d} - ICD-10 {c}"
    return d


# ---------------------------------------------------------------------------
# Near-miss detection for fuzzy auto-matches
# ---------------------------------------------------------------------------

SHORT_TOKEN_RE = re.compile(r"^(i{1,3}|iv|v|1|2|3|a|b|c)$", re.I)
ICD_RE = re.compile(r"\b[a-z]\d{2}(?:\.\d+)?\b", re.I)


def _tokens(s):
    s = ICD_RE.sub(" ", s.lower())
    s = re.sub(r"\bicd[\s-]*10\b", " ", s)
    drop = {"disorder", "type", "and", "with", "the", "of"}
    return {w for w in re.findall(r"[a-z0-9]+", s) if w not in drop}


def differing_tokens(t, p):
    ta, pa = _tokens(t), _tokens(p)
    return (ta - pa) | (pa - ta)


# ---------------------------------------------------------------------------
# Evaluator — same logic and prompt as the study, plus instrumentation
# ---------------------------------------------------------------------------

class HybridEvaluator:
    def __init__(self, fuzzy_threshold=90, llm_model="gpt-5-mini"):
        self.fuzzy_threshold = fuzzy_threshold
        self.llm_model = llm_model
        self.cache = {}
        self.llm_calls = 0
        self.llm_errors = 0
        self.fuzzy_matches = 0
        self.llm_matches = 0
        self.flagged = []

    def check_match(self, true_diag, pred_diag):
        """Returns (is_match, tier) with tier in {'fuzzy','llm','cache','error'}."""
        t = true_diag.lower().strip()
        p = pred_diag.lower().strip()

        score = fuzz.token_set_ratio(t, p)
        if score >= self.fuzzy_threshold:
            self.fuzzy_matches += 1
            diff = differing_tokens(t, p)
            if diff and all(SHORT_TOKEN_RE.match(d) for d in diff):
                self.flagged.append({
                    "true": true_diag, "pred": pred_diag,
                    "ratio": round(score, 1),
                    "differing_tokens": "|".join(sorted(diff)),
                })
            return True, "fuzzy"

        key = f"{t} || {p}"
        if key in self.cache:
            return self.cache[key], "cache"

        is_match, ok = self._ask_llm(t, p)
        if not ok:
            self.llm_errors += 1
            return False, "error"
        self.cache[key] = is_match
        self.llm_calls += 1
        if is_match:
            self.llm_matches += 1
        return is_match, "llm"

    def _ask_llm(self, t, p):
        prompt = f"""

        Your task is to act as a strict medical adjudicator specializing in psychiatry and identify whether the predicted diagnosis is clinically equivalent to (or a valid subclass of) the true diagnosis. Your standards are exacting, and you must consider the nuances of each diagnosis carefully. As much as possible, adhere to the diagnostic language laid out in the DSM-5-TR, and utilize the included ICD-10 F-codes to aid your determination.

        True Diagnosis: "{t}"
        Predicted Diagnosis: "{p}"

        Return JSON ONLY: {{ "match": <true/false> }}
        """

        class DiagnosisMatch(BaseModel):
            match: bool

        try:
            response = client.responses.parse(
                model=self.llm_model,
                input=[{"role": "user", "content": prompt}],
                text_format=DiagnosisMatch,
            )
            text = None
            for item in response.output:
                content = getattr(item, "content", None)
                if content:
                    text = content[0].text
                    break
            if text is None:
                return False, False
            return json.loads(text).get("match", False), True
        except Exception as e:
            print(f"  LLM error ({t[:30]} vs {p[:30]}): {e}")
            return False, False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_group(group, ev, include_code=True):
    """group: rows for one (model, case_id, run), any order."""
    g = group.sort_values("rank")
    y_true = parse_ground_truth_diagnoses(g["reference_diagnosis"].iloc[0])
    y_pred = [rebuild_prediction(r.predicted_diagnosis, r.predicted_code, include_code)
              for r in g.itertuples()]

    if not y_true or not y_pred:
        return None

    found, first_rank = set(), None
    for rank_idx, pred_item in enumerate(y_pred):
        hit = False
        for true_idx, true_item in enumerate(y_true):
            m, _ = ev.check_match(true_item, pred_item)
            if m:
                hit = True
                found.add(true_idx)
        if hit and first_rank is None:
            first_rank = rank_idx + 1

    return {
        "top1": 1.0 if first_rank == 1 else 0.0,
        "hit_rate": 1.0 if found else 0.0,
        "recall": len(found) / len(y_true),
        "mrr": (1 / first_rank) if first_rank else 0.0,
        "n_true": len(y_true),
        "n_pred": len(y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="adjudication_pairs.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--fuzzy-threshold", type=int, default=90,
                    help="90 reproduces the study pipeline; 100 disables the fuzzy tier")
    ap.add_argument("--no-code", action="store_true",
                    help="Score on diagnosis text alone, without appending the ICD code")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N case/run groups (dry run)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pairs)
    need = {"model", "run", "case_id", "rank", "predicted_diagnosis",
            "predicted_code", "reference_diagnosis"}
    missing_cols = need - set(df.columns)
    if missing_cols:
        raise SystemExit(f"CSV is missing columns: {sorted(missing_cols)}")

    print(f"Loaded {len(df)} rows from {args.pairs}")
    print(f"Fuzzy threshold: {args.fuzzy_threshold}"
          f"{'  (fuzzy tier DISABLED — all comparisons go to the LLM)' if args.fuzzy_threshold > 100 or args.fuzzy_threshold == 100 else ''}")
    print(f"ICD codes included in matched string: {not args.no_code}\n")

    groups = dict(list(df.groupby(["model", "case_id", "run"])))

    # Identify case/run combinations absent from the CSV (content filter,
    # empty output, or parse failure upstream).
    models = sorted(df["model"].unique())
    unscorable = []
    for m in models:
        for c in sorted(df[df["model"] == m]["case_id"].unique()):
            for r in ("run1", "run2"):
                if (m, c, r) not in groups:
                    unscorable.append({"model": m, "case_id": c, "run": r,
                                       "note": "absent from CSV — check raw output "
                                               "(content filter / empty / parse failure)"})
    if unscorable:
        pd.DataFrame(unscorable).to_csv(outdir / "unscorable_cases.csv", index=False)
        print(f"{len(unscorable)} case/run combination(s) absent from the CSV:")
        for u in unscorable:
            print(f"  {u['model']}  case {u['case_id']}  {u['run']}")
        print()

    ev = HybridEvaluator(fuzzy_threshold=args.fuzzy_threshold)
    scored = {}
    keys = sorted(groups)
    if args.limit:
        keys = keys[:args.limit]
    for k in tqdm(keys, desc="Scoring"):
        s = score_group(groups[k], ev, include_code=not args.no_code)
        if s:
            scored[k] = s

    # Pair the runs
    rows = []
    for m in models:
        for c in sorted(df[df["model"] == m]["case_id"].unique()):
            a, b = scored.get((m, c, "run1")), scored.get((m, c, "run2"))
            rows.append({
                "model": m, "case_id": c,
                "scorable": bool(a and b),
                **{f"{k}_run1": (a[k] if a else None) for k in
                   ("top1", "hit_rate", "recall", "mrr")},
                **{f"{k}_run2": (b[k] if b else None) for k in
                   ("top1", "hit_rate", "recall", "mrr")},
            })
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "run_stability_scored.csv", index=False)

    # ---- summary ----
    lines = ["RUN-TO-RUN STABILITY — study hybrid evaluator "
             f"(fuzzy>={args.fuzzy_threshold}, then {ev.llm_model})",
             "=" * 70, ""]
    for m, g in out.groupby("model"):
        s = g[g.scorable]
        n = len(s)
        excl = len(g) - n
        if n == 0:
            continue
        f1 = int((s.top1_run1 != s.top1_run2).sum())
        fh = int((s.hit_rate_run1 != s.hit_rate_run2).sum())
        lines += [
            f"{m}   (n={n} scorable, {excl} excluded)",
            f"  Top-1     : run1 {s.top1_run1.mean():.3f}   run2 {s.top1_run2.mean():.3f}   "
            f"flipped {f1}/{n} ({100*f1/n:.1f}%)",
            f"  Top-5/hit : run1 {s.hit_rate_run1.mean():.3f}   run2 {s.hit_rate_run2.mean():.3f}   "
            f"flipped {fh}/{n} ({100*fh/n:.1f}%)",
            f"  Recall@5  : run1 {s.recall_run1.mean():.3f}   run2 {s.recall_run2.mean():.3f}",
            f"  MRR       : run1 {s.mrr_run1.mean():.3f}   run2 {s.mrr_run2.mean():.3f}",
            "",
        ]

    tot = out[out.scorable]
    if len(tot):
        lines += [
            "OVERALL",
            f"  Top-1 flipped     : {int((tot.top1_run1!=tot.top1_run2).sum())}/{len(tot)} "
            f"({100*(tot.top1_run1!=tot.top1_run2).mean():.1f}%)",
            f"  Top-5 flipped     : {int((tot.hit_rate_run1!=tot.hit_rate_run2).sum())}/{len(tot)} "
            f"({100*(tot.hit_rate_run1!=tot.hit_rate_run2).mean():.1f}%)",
            "",
        ]

    lines += [
        "ADJUDICATION TIERS",
        f"  Matches from fuzzy tier : {ev.fuzzy_matches}",
        f"  Matches from LLM tier   : {ev.llm_matches}",
        f"  LLM calls made          : {ev.llm_calls}",
        f"  LLM errors              : {ev.llm_errors}"
        + ("   <-- scored as non-match; the original pipeline did this silently"
           if ev.llm_errors else ""),
        "",
    ]

    if ev.flagged:
        uniq, seen = [], set()
        for f in ev.flagged:
            k = (f["true"].lower(), f["pred"].lower())
            if k not in seen:
                seen.add(k)
                uniq.append(f)
        pd.DataFrame(ev.flagged).to_csv(outdir / "flagged_fuzzy_matches.csv", index=False)
        lines += [
            f"!! {len(ev.flagged)} fuzzy auto-matches ({len(uniq)} unique) differ only by a "
            "short discriminating token",
            "   These were scored as correct WITHOUT reaching the LLM adjudicator.",
            "",
        ]
        for f in uniq[:25]:
            lines.append(f"   ratio {f['ratio']:5.1f}   {f['true']}   ==   {f['pred']}"
                         f"   (differs by: {f['differing_tokens']})")
        if len(uniq) > 25:
            lines.append(f"   ... and {len(uniq)-25} more in flagged_fuzzy_matches.csv")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    (outdir / "run_stability_summary.txt").write_text(report)
    print(f"Wrote outputs to {outdir}/")
    print("""
NEXT
  1. Review flagged_fuzzy_matches.csv. Any Bipolar I vs II pair there was
     counted correct without adjudication.
  2. Re-run with --fuzzy-threshold 100 and diff the accuracy figures. The
     difference is what the fuzzy shortcut contributed to the reported numbers.
  3. unscorable_cases.csv lists case/run combinations with no rows. Confirm
     each against the raw JSON and state the denominator convention in Methods.
""")


if __name__ == "__main__":
    main()
