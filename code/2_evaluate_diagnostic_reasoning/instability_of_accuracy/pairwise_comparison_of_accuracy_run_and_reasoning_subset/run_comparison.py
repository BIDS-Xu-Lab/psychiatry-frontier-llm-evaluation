#!/usr/bin/env python3
"""
Run-to-run stability comparison for LLM diagnostic outputs.

Compares two independent generation runs (the 196-case accuracy run and the
30-case reasoning-subset run) that used identical prompts and model aliases.

Outputs:
  1. Top-1 agreement between runs (by ICD code and by normalized text)
  2. Overlap of the 5-item differential (set intersection, Jaccard, positional)
  3. Correctness flips: cases where top-1/top-5 correctness differs between runs
  4. All of the above stratified by model
  5. adjudication_pairs.csv for authoritative scoring with your GPT-5-mini adjudicator

Usage:
    python run_comparison.py --config config.json
    python run_comparison.py --demo structure.txt     # test the parser on a sample
"""

import json, re, argparse, itertools, sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG — edit these paths, or supply a config.json with the same keys
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "run1_label": "accuracy_run_196",
    "run2_label": "reasoning_subset_30",
    "files": {
        # model name : {run1: path, run2: path}
        "DeepSeek-V3.2":       {"run1": "data/deepseek_196.json",  "run2": "data/deepseek_30.json"},
        "Claude Opus 4.5":     {"run1": "data/claude_196.json",    "run2": "data/claude_30.json"},
        "Gemini 3 Pro":        {"run1": "data/gemini_196.json",    "run2": "data/gemini_30.json"},
        "GPT-5.2":             {"run1": "data/gpt52_196.json",     "run2": "data/gpt52_30.json"},
    },
    # Field names in the JSON objects
    "id_field":        "case_id",
    "reference_field": "diagnosis",
    "output_field":    "model_diagnosis",
    "trace_field":     "model_thoughts",
    "source_field":    "source",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches: "1. Fetishistic Disorder - ICD-10 F65.0"
#          "2) Adjustment Disorder, With Anxiety — F43.22"
#          "3. Some Disorder (F41.1)"
LINE_RE = re.compile(
    r"^\s*(\d+)\s*[\.\)\:]\s*(.+?)\s*$"
)
CODE_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b")


def normalize_text(s):
    """Lowercase, strip punctuation/specifiers, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"icd[- ]?10", " ", s)
    s = re.sub(r"\b[a-z]\d{2}(\.\d+)?\b", " ", s)      # strip codes
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    # Drop common specifier words that don't change the underlying condition
    for w in ["disorder", "unspecified", "other", "specified", "with", "without",
              "single", "episode", "moderate", "severe", "mild", "recurrent",
              "type", "nos", "due", "to"]:
        s = re.sub(rf"\b{w}\b", " ", s)
    return " ".join(s.split())


def parse_differential(raw):
    """
    Parse a model_diagnosis string into an ordered list of dicts:
        [{rank, text, code, norm}, ...]
    Robust to missing codes and varied dash characters.
    """
    if not raw:
        return []
    out = []
    for line in str(raw).split("\n"):
        m = LINE_RE.match(line)
        if not m:
            continue
        rank = int(m.group(1))
        body = m.group(2)
        code_m = CODE_RE.search(body)
        code = code_m.group(1) if code_m else None
        # Text = everything before the dash/code
        text = re.split(r"\s+[-–—]\s+", body)[0]
        text = re.sub(r"\(.*?\)", "", text).strip()
        out.append({
            "rank": rank,
            "text": text,
            "code": code,
            "norm": normalize_text(text),
        })
    out.sort(key=lambda d: d["rank"])
    return out


def load_run(path, cfg):
    """Load a JSON file into {case_id: record}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {path}")
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("results", data.get("data", list(data.values())))
    out = {}
    for rec in data:
        cid = rec.get(cfg["id_field"])
        if cid is None:
            continue
        out[cid] = {
            "case_id": cid,
            "reference": rec.get(cfg["reference_field"]),
            "source": rec.get(cfg["source_field"]),
            "differential": parse_differential(rec.get(cfg["output_field"])),
            "trace": rec.get(cfg["trace_field"]) or "",
            "raw_output": rec.get(cfg["output_field"]),
        }
    return out


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compare_case(a, b):
    """Compare one case's differential across two runs."""
    da, db = a["differential"], b["differential"]
    if not da or not db:
        return None

    codes_a = [d["code"] for d in da if d["code"]]
    codes_b = [d["code"] for d in db if d["code"]]
    norms_a = [d["norm"] for d in da if d["norm"]]
    norms_b = [d["norm"] for d in db if d["norm"]]

    # Top-1
    top1_code = (codes_a[0] == codes_b[0]) if (codes_a and codes_b) else None
    top1_text = (norms_a[0] == norms_b[0]) if (norms_a and norms_b) else None

    # Set overlap (order-independent)
    sa, sb = set(codes_a), set(codes_b)
    inter_code = len(sa & sb)
    jacc_code = len(sa & sb) / len(sa | sb) if (sa | sb) else None

    sna, snb = set(norms_a), set(norms_b)
    inter_text = len(sna & snb)
    jacc_text = len(sna & snb) / len(sna | snb) if (sna | snb) else None

    # Positional agreement: same code at same rank
    pos_match = sum(1 for i in range(min(len(codes_a), len(codes_b)))
                    if codes_a[i] == codes_b[i])

    return {
        "case_id": a["case_id"],
        "source": a.get("source"),
        "n_a": len(da), "n_b": len(db),
        "top1_code_match": top1_code,
        "top1_text_match": top1_text,
        "set_intersection_code": inter_code,
        "jaccard_code": jacc_code,
        "set_intersection_text": inter_text,
        "jaccard_text": jacc_text,
        "positional_matches": pos_match,
        "top1_a": da[0]["text"] if da else None,
        "top1_b": db[0]["text"] if db else None,
        "codes_a": ";".join(codes_a),
        "codes_b": ";".join(codes_b),
        "trace_len_a": len(a["trace"].split()),
        "trace_len_b": len(b["trace"].split()),
    }


def heuristic_correct(differential, reference, top_n):
    """
    APPROXIMATE correctness by normalized text containment.
    NOT authoritative — use the LLM adjudicator for reported numbers.
    Returns True/False/None.
    """
    if not differential or not reference:
        return None
    ref = normalize_text(reference)
    if not ref:
        return None
    ref_tokens = set(ref.split())
    for d in differential[:top_n]:
        if not d["norm"]:
            continue
        if ref in d["norm"] or d["norm"] in ref:
            return True
        dt = set(d["norm"].split())
        if ref_tokens and dt and len(ref_tokens & dt) / len(ref_tokens) >= 0.75:
            return True
    return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pct(x, n):
    return f"{100*x/n:.1f}%" if n else "n/a"


def summarize(rows, model, run1_label, run2_label):
    n = len(rows)
    if n == 0:
        print(f"  {model}: no paired cases")
        return {}
    t1c = [r["top1_code_match"] for r in rows if r["top1_code_match"] is not None]
    t1t = [r["top1_text_match"] for r in rows if r["top1_text_match"] is not None]
    inter = [r["set_intersection_code"] for r in rows]
    jacc = [r["jaccard_code"] for r in rows if r["jaccard_code"] is not None]
    pos = [r["positional_matches"] for r in rows]

    s = {
        "model": model,
        "n_pairs": n,
        "top1_code_agree": sum(t1c)/len(t1c) if t1c else None,
        "top1_text_agree": sum(t1t)/len(t1t) if t1t else None,
        "mean_overlap_of_5": sum(inter)/len(inter) if inter else None,
        "mean_jaccard": sum(jacc)/len(jacc) if jacc else None,
        "mean_positional": sum(pos)/len(pos) if pos else None,
        "identical_lists": sum(1 for r in rows if r["set_intersection_code"] == 5
                               and r["positional_matches"] == 5),
    }
    print(f"\n  {model}  (n = {n} paired cases)")
    print(f"    Top-1 same diagnosis (ICD code)   : {s['top1_code_agree']*100:.1f}%" if s['top1_code_agree'] is not None else "    Top-1 (code): n/a")
    print(f"    Top-1 same diagnosis (text)       : {s['top1_text_agree']*100:.1f}%" if s['top1_text_agree'] is not None else "    Top-1 (text): n/a")
    print(f"    Mean shared dx out of 5           : {s['mean_overlap_of_5']:.2f}")
    print(f"    Mean Jaccard (5-item lists)       : {s['mean_jaccard']:.3f}")
    print(f"    Mean same-code-at-same-rank       : {s['mean_positional']:.2f} / 5")
    print(f"    Fully identical differentials     : {s['identical_lists']}/{n} ({pct(s['identical_lists'], n)})")
    return s


def correctness_flips(pairs_by_model, cfg):
    print("\n" + "="*78)
    print("CORRECTNESS FLIPS  (heuristic matcher — see caveat below)")
    print("="*78)
    allrows = []
    for model, pairs in pairs_by_model.items():
        n = len(pairs)
        if n == 0:
            continue
        flips1 = flips5 = 0
        a1 = b1 = a5 = b5 = 0
        for a, b in pairs:
            ca1 = heuristic_correct(a["differential"], a["reference"], 1)
            cb1 = heuristic_correct(b["differential"], b["reference"], 1)
            ca5 = heuristic_correct(a["differential"], a["reference"], 5)
            cb5 = heuristic_correct(b["differential"], b["reference"], 5)
            if ca1 is not None and cb1 is not None:
                a1 += ca1; b1 += cb1; flips1 += (ca1 != cb1)
            if ca5 is not None and cb5 is not None:
                a5 += ca5; b5 += cb5; flips5 += (ca5 != cb5)
        print(f"\n  {model} (n={n})")
        print(f"    top-1 correct: run1 {a1}/{n} ({pct(a1,n)})  run2 {b1}/{n} ({pct(b1,n)})  FLIPPED {flips1} ({pct(flips1,n)})")
        print(f"    top-5 correct: run1 {a5}/{n} ({pct(a5,n)})  run2 {b5}/{n} ({pct(b5,n)})  FLIPPED {flips5} ({pct(flips5,n)})")
        allrows.append((model, n, flips1, flips5))
    if allrows:
        tot = sum(r[1] for r in allrows)
        f1 = sum(r[2] for r in allrows); f5 = sum(r[3] for r in allrows)
        print(f"\n  OVERALL: top-1 flipped {f1}/{tot} ({pct(f1,tot)}) | top-5 flipped {f5}/{tot} ({pct(f5,tot)})")
    print("""
  CAVEAT: these use approximate string matching and will disagree with your
  GPT-5-mini adjudicator. Treat as a screen only. Use adjudication_pairs.csv
  below to produce the numbers you would actually report.""")


def write_adjudication_file(pairs_by_model, out_path):
    """Emit every (run, case, rank, diagnosis, reference) for authoritative scoring."""
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "run", "case_id", "rank", "predicted_diagnosis",
                    "predicted_code", "reference_diagnosis"])
        for model, pairs in pairs_by_model.items():
            for a, b in pairs:
                for run_label, rec in (("run1", a), ("run2", b)):
                    for d in rec["differential"]:
                        w.writerow([model, run_label, rec["case_id"], d["rank"],
                                    d["text"], d["code"] or "", rec["reference"]])
    print(f"\n  Wrote {out_path} — score this with your GPT-5-mini adjudicator,")
    print("  then recompute flips from the adjudicated verdicts.")


def write_case_detail(pairs_by_model, out_path):
    import csv
    rows = []
    for model, pairs in pairs_by_model.items():
        for a, b in pairs:
            r = compare_case(a, b)
            if r:
                r["model"] = model
                rows.append(r)
    if not rows:
        return
    keys = ["model"] + [k for k in rows[0] if k != "model"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {out_path} — per-case detail for manual inspection.")




# ---------------------------------------------------------------------------
# Reasoning-trace comparison
# ---------------------------------------------------------------------------

STOP = set("""a an the and or but if then than that this these those of in on at to for
with from by as is are was were be been being it its he she his her they them their we
us our you your i me my not no so such can could would should may might will just also
about into over under more most less least other another each any all some which who whom
whose what when where why how there here up down out off again further once do does did
doing have has had having""".split())

DELIB = ["wait", "hmm", "actually", "let me reconsider", "on second thought",
         "but then", "however", "although", "revisit", "double-check",
         "double check", "rule out", "alternatively", "need to check"]

# Phrases suggesting the model is explaining an answer already supplied to it,
# rather than reasoning toward one. See notes in the report.
POSTHOC = [
    "the answer provided", "the response provided", "the answer lists",
    "the response lists", "the provided answer", "the given answer",
    "explain the thought process", "explain my thought process",
    "explain the reasoning behind", "justify the diagnos",
    "the thinking should reflect", "the diagnoses provided",
    "the model concluded", "why those were chosen", "how those were prioritized",
]


def content_tokens(text):
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return [t for t in toks if t not in STOP and len(t) > 2]


def bigrams(toks):
    return set(zip(toks, toks[1:]))


def cosine_tfidf(a_toks, b_toks):
    """Simple TF-IDF-free cosine on raw term frequencies (no sklearn dependency)."""
    from math import sqrt
    from collections import Counter
    ca, cb = Counter(a_toks), Counter(b_toks)
    keys = set(ca) | set(cb)
    if not keys:
        return None
    dot = sum(ca[k] * cb[k] for k in keys)
    na = sqrt(sum(v * v for v in ca.values()))
    nb = sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else None


def count_markers(text, markers):
    t = (text or "").lower()
    return sum(t.count(m) for m in markers)


def compare_traces(a, b):
    ta, tb = a["trace"] or "", b["trace"] or ""
    if not ta and not tb:
        return None
    wa, wb = ta.split(), tb.split()
    toks_a, toks_b = content_tokens(ta), content_tokens(tb)
    sa, sb = set(toks_a), set(toks_b)
    bga, bgb = bigrams(toks_a), bigrams(toks_b)

    codes_a = set(CODE_RE.findall(ta))
    codes_b = set(CODE_RE.findall(tb))

    return {
        "case_id": a["case_id"],
        "words_run1": len(wa),
        "words_run2": len(wb),
        "word_ratio": (len(wb) / len(wa)) if wa else None,
        "unigram_jaccard": (len(sa & sb) / len(sa | sb)) if (sa | sb) else None,
        "bigram_jaccard": (len(bga & bgb) / len(bga | bgb)) if (bga | bgb) else None,
        "cosine": cosine_tfidf(toks_a, toks_b),
        "codes_in_trace_run1": len(codes_a),
        "codes_in_trace_run2": len(codes_b),
        "code_overlap_in_trace": len(codes_a & codes_b),
        "delib_run1": count_markers(ta, DELIB),
        "delib_run2": count_markers(tb, DELIB),
        "posthoc_run1": count_markers(ta, POSTHOC),
        "posthoc_run2": count_markers(tb, POSTHOC),
    }


def summarize_traces(pairs_by_model):
    print("\n" + "=" * 78)
    print("REASONING-TRACE COMPARISON")
    print("=" * 78)
    print("""
  NOTE ON INTERPRETATION
  These are SURFACE measures. High lexical overlap does not establish that two
  traces are of equal clinical quality, and low overlap does not establish that
  they differ in quality. Only clinician rating can answer that. What these
  measures CAN do is tell you whether the two runs produced recognizably similar
  material, and flag whether they were elicited the same way.
""")
    all_rows = []
    for model, pairs in pairs_by_model.items():
        rows = [r for r in (compare_traces(a, b) for a, b in pairs) if r]
        if not rows:
            print(f"\n  {model}: no traces to compare")
            continue
        n = len(rows)
        def mean(k):
            vals = [r[k] for r in rows if r[k] is not None]
            return sum(vals) / len(vals) if vals else None
        w1, w2 = mean("words_run1"), mean("words_run2")
        print(f"\n  {model}  (n = {n})")
        print(f"    Mean trace length      : run1 {w1:.0f} words | run2 {w2:.0f} words")
        print(f"    Unigram Jaccard        : {mean('unigram_jaccard'):.3f}")
        print(f"    Bigram Jaccard         : {mean('bigram_jaccard'):.3f}")
        print(f"    Term-frequency cosine  : {mean('cosine'):.3f}")
        print(f"    ICD codes in trace     : run1 {mean('codes_in_trace_run1'):.1f} | "
              f"run2 {mean('codes_in_trace_run2'):.1f} | shared {mean('code_overlap_in_trace'):.1f}")
        print(f"    Deliberation markers   : run1 {mean('delib_run1'):.1f} | run2 {mean('delib_run2'):.1f}")

        ph1 = sum(1 for r in rows if r["posthoc_run1"] > 0)
        ph2 = sum(1 for r in rows if r["posthoc_run2"] > 0)
        flag = "  <-- INVESTIGATE" if (ph1 > n * 0.1 or ph2 > n * 0.1) else ""
        print(f"    Post-hoc framing phrases in: run1 {ph1}/{n} traces | run2 {ph2}/{n} traces{flag}")

        for r in rows:
            r["model"] = model
        all_rows.extend(rows)

    # Global post-hoc warning
    tot = len(all_rows)
    if tot:
        p1 = sum(1 for r in all_rows if r["posthoc_run1"] > 0)
        p2 = sum(1 for r in all_rows if r["posthoc_run2"] > 0)
        if p1 > tot * 0.1 or p2 > tot * 0.1:
            print("""
  ***  WARNING  ***
  A substantial share of traces contain phrasing that suggests the model was
  explaining a diagnosis already present in its input, rather than reasoning
  toward one (e.g. "the answer provided", "explain the thought process").

  If this is concentrated in ONE run, the two runs did not elicit reasoning the
  same way, and traces from that run are NOT interchangeable with the other's.
  Consequences:
    - The blinded study must draw traces from the same elicitation method that
      produced the originally rated traces.
    - The Methods description of how reasoning was obtained needs to match what
      each run actually did.
  Inspect posthoc_examples.txt before proceeding.
""")
    return all_rows


def write_trace_outputs(all_rows, pairs_by_model, outdir):
    import csv
    if all_rows:
        keys = ["model"] + [k for k in all_rows[0] if k != "model"]
        with open(Path(outdir) / "trace_comparison_by_case.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"  Wrote trace_comparison_by_case.csv")

    # Dump examples of post-hoc framing for manual review
    lines = []
    for model, pairs in pairs_by_model.items():
        for a, b in pairs:
            for run_label, rec in (("run1", a), ("run2", b)):
                t = (rec["trace"] or "")
                low = t.lower()
                for ph in POSTHOC:
                    if ph in low:
                        i = low.index(ph)
                        lines.append(
                            f"[{model} | {run_label} | case {rec['case_id']}] "
                            f"...{t[max(0,i-120):i+200]}...\n")
                        break
    if lines:
        (Path(outdir) / "posthoc_examples.txt").write_text("\n".join(lines))
        print(f"  Wrote posthoc_examples.txt ({len(lines)} traces flagged)")


# ---------------------------------------------------------------------------
# Demo mode: test the parser on the structure.txt sample
# ---------------------------------------------------------------------------

def demo(path):
    txt = Path(path).read_text()
    blocks = re.findall(r'"model_diagnosis"\s*:\s*"((?:[^"\\]|\\.)*)"', txt)
    print(f"Found {len(blocks)} model_diagnosis blocks in {path}\n")
    parsed = []
    for i, b in enumerate(blocks, 1):
        raw = b.encode().decode("unicode_escape")
        d = parse_differential(raw)
        parsed.append(d)
        print(f"--- Block {i} ---")
        for item in d:
            print(f"  {item['rank']}. {item['text']:45s} code={item['code']}")
        print()
    if len(parsed) >= 2:
        a = {"case_id": "demo", "differential": parsed[0], "trace": "", "reference": None, "source": None}
        b = {"case_id": "demo", "differential": parsed[1], "trace": "", "reference": None, "source": None}
        r = compare_case(a, b)
        print("--- Comparison of first two blocks ---")
        for k, v in r.items():
            print(f"  {k:24s} {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to config.json")
    ap.add_argument("--demo", help="Run parser demo on a sample text file")
    ap.add_argument("--outdir", default=".", help="Where to write CSVs")
    args = ap.parse_args()

    if args.demo:
        demo(args.demo)
        return

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))

    print("="*78)
    print("RUN-TO-RUN STABILITY COMPARISON")
    print(f"  {cfg['run1_label']}  vs  {cfg['run2_label']}")
    print("="*78)

    pairs_by_model = {}
    summaries = []

    for model, paths in cfg["files"].items():
        try:
            r1 = load_run(paths["run1"], cfg)
            r2 = load_run(paths["run2"], cfg)
        except FileNotFoundError as e:
            print(f"\n  SKIPPING {model}: {e}")
            continue

        shared = sorted(set(r1) & set(r2))
        print(f"\n  {model}: run1 has {len(r1)} cases, run2 has {len(r2)}, "
              f"{len(shared)} in common")
        if len(shared) != 30:
            print(f"    !! expected 30 shared cases, got {len(shared)} — check the data")

        # Sanity: identical vignette + reference across runs
        mismatched_ref = [c for c in shared if r1[c]["reference"] != r2[c]["reference"]]
        if mismatched_ref:
            print(f"    !! {len(mismatched_ref)} cases have differing reference diagnoses "
                  f"between runs: {mismatched_ref[:5]}")

        pairs_by_model[model] = [(r1[c], r2[c]) for c in shared]

    print("\n" + "="*78)
    print("DIFFERENTIAL STABILITY")
    print("="*78)
    for model, pairs in pairs_by_model.items():
        rows = [compare_case(a, b) for a, b in pairs]
        rows = [r for r in rows if r]
        s = summarize(rows, model, cfg["run1_label"], cfg["run2_label"])
        if s:
            summaries.append(s)

    if summaries:
        print("\n" + "-"*78)
        print("  Expected ordering if this reflects sampling temperature:")
        print("    Gemini 3 Pro (temp=1) least stable ... DeepSeek-V3.2 (temp=0) most stable")
        print("    Claude / GPT-5.2 omit temperature (reasoning models).")
        print("  If DeepSeek is NOT the most stable, suspect alias drift between runs")
        print("  or a data-matching problem — investigate before reporting.")
        print("-"*78)

    trace_rows = summarize_traces(pairs_by_model)

    correctness_flips(pairs_by_model, cfg)

    outdir = Path(args.outdir)
    write_adjudication_file(pairs_by_model, outdir / "adjudication_pairs.csv")
    write_case_detail(pairs_by_model, outdir / "run_comparison_by_case.csv")
    write_trace_outputs(trace_rows, pairs_by_model, outdir)

    print("\n" + "="*78)
    print("NEXT STEPS")
    print("="*78)
    print("""
  1. Check the access timestamps for both runs. If they are weeks apart, alias
     drift is a competing explanation for any instability you find here.
  2. Diff the actual prompt strings used in each run. If they differ at all,
     this measures prompt sensitivity, not model stochasticity.
  3. Score adjudication_pairs.csv with GPT-5-mini and recompute the flip rates
     from those verdicts — those are the numbers to report.
  4. Compare the flip rate against the between-model spread in Table 1
     (top-5 accuracy ranges 0.730-0.801, a 7-point spread). If run-to-run
     flips are of similar magnitude, the Table 1 ranking is not stable.
""")


if __name__ == "__main__":
    main()
