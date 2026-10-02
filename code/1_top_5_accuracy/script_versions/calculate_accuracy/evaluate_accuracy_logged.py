"""
evaluate_accuracy_logged.py
Same scoring logic as evaluate_accuracy.py (same parsing, same fuzzy tier, same
gpt-5-mini prompt, same metric definitions), with four changes:

  1. Every reference x prediction comparison is logged: tier that decided it,
     fuzzy score, LLM verdict, whether it came from cache, raw LLM text, errors.
  2. Failed LLM calls are retried, then recorded as ERROR instead of silently
     scored False. Case-level metrics still treat ERROR as a non-match (the
     original convention) but every case carries an n_llm_errors count.
  3. --fuzzy-threshold accepts a number or "off" (every pair goes to the LLM).
  4. --pass-id labels repeated adjudication passes; --cases restricts to a
     list of case IDs (e.g. the 30 clinician-rated vignettes).

Usage
  python evaluate_accuracy_logged.py --input run2_claude.json --model-label "Claude Opus 4.5" \
      --fuzzy-threshold 90 --pass-id 1 --cases rated_case_ids.txt --outdir adjudication_logs/

Input: JSON list of cases (as written by the generation scripts) or a CSV with
columns case_id, diagnosis, model_diagnosis.
"""
import argparse, hashlib, json, os, re, time
from datetime import datetime, timezone

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

COL_TRUE, COL_PRED = "diagnosis", "model_diagnosis"
LLM_MODEL = "gpt-5-mini"

PROMPT = """

        Your task is to act as a strict medical adjudicator specializing in psychiatry and identify whether the predicted diagnosis is clinically equivalent to (or a valid subclass of) the true diagnosis. Your standards are exacting, and you must consider the nuances of each diagnosis carefully. As much as possible, adhere to the diagnostic language laid out in the DSM-5-TR, and utilize the included ICD-10 F-codes to aid your determination.

        True Diagnosis: "{t}"
        Predicted Diagnosis: "{p}"

        Return JSON ONLY: {{ "match": <true/false> }}
        """  # verbatim from evaluate_accuracy.py, so verdicts stay comparable


# ---------------------------------------------------------------- parsing
# Ground-truth parsing is identical to evaluate_accuracy.py (semicolon splitting is
# flagged, not changed). Prediction parsing differs in one documented way: see below.
def parse_ground_truth_diagnoses(s) -> list:
    if not isinstance(s, str):
        return []
    if re.search(r"^\d+\.", s.strip()):
        out = []
        for line in s.strip().split("\n"):
            m = re.match(r"\d+\.\s+(.*)", line)
            out.append(m.group(1).strip() if m else line.strip())
        return out
    return [d.strip() for d in re.split(r";", s) if d.strip()]


def parse_model_predicted_diagnoses(s) -> list:
    """
    Same as the original for clean numbered lists. One correction: if the output
    contains numbered lines, unnumbered lines (preambles such as "Based on the
    vignette ... here are the top 5:", blank lines) are dropped, so they cannot
    occupy ranks. Outputs with no numbered lines are parsed exactly as before.
    """
    if not isinstance(s, str):
        return []
    lines = [l.strip() for l in s.strip().split("\n")]
    numbered = [re.match(r"\d+\.\s+(.*)", l) for l in lines]
    if any(numbered):
        return [m.group(1).strip() for m in numbered if m]
    return [l for l in lines]


def n_dropped_lines(s) -> int:
    """How many lines the corrected parser dropped (0 for clean lists)."""
    if not isinstance(s, str):
        return 0
    lines = [l.strip() for l in s.strip().split("\n")]
    if any(re.match(r"\d+\.\s+", l) for l in lines):
        return sum(1 for l in lines if not re.match(r"\d+\.\s+", l))
    return 0


# -------------------------------------------------------------- evaluator
class LoggedHybridEvaluator:
    def __init__(self, fuzzy_threshold, client=None, max_attempts=4, model=LLM_MODEL):
        self.fuzzy_threshold = fuzzy_threshold          # None = fuzzy tier off
        self.client = client
        self.model = model                               # OpenAI model or Azure deployment name
        self.max_attempts = max_attempts
        self.cache = {}                                  # same per-run cache as original
        self.llm_calls = 0
        self.llm_errors = 0

    def check_match(self, true_diag, pred_diag) -> dict:
        t, p = true_diag.lower().strip(), pred_diag.lower().strip()
        score = fuzz.token_set_ratio(t, p)
        rec = {"fuzzy_score": round(score, 2), "tier": None, "verdict": None,
               "from_cache": False, "llm_raw": None, "llm_error": None, "attempts": 0}

        if self.fuzzy_threshold is not None and score >= self.fuzzy_threshold:
            rec.update(tier="FUZZY", verdict=True)
            return rec

        key = f"{t} || {p}"
        if key in self.cache:
            rec.update(self.cache[key], from_cache=True)
            return rec

        llm = self._ask_llm(t, p)
        self.cache[key] = llm
        rec.update(llm)
        return rec

    def _ask_llm(self, t, p) -> dict:
        from pydantic import BaseModel

        class DiagnosisMatch(BaseModel):
            match: bool

        last_err = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.client.responses.parse(
                    model=self.model,
                    input=[{"role": "user", "content": PROMPT.format(t=t, p=p)}],
                    text_format=DiagnosisMatch,
                )
                self.llm_calls += 1
                parsed = getattr(resp, "output_parsed", None)
                raw = getattr(resp, "output_text", None)
                if parsed is None:  # fall back to the original indexing
                    raw = resp.output[1].content[0].text
                    parsed = DiagnosisMatch(**json.loads(raw))
                return {"tier": "LLM", "verdict": bool(parsed.match), "llm_raw": raw,
                        "llm_error": None, "attempts": attempt}
            except Exception as e:  # noqa: BLE001 — logged, not swallowed
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))
        self.llm_errors += 1
        return {"tier": "LLM_ERROR", "verdict": None, "llm_raw": None,
                "llm_error": last_err, "attempts": self.max_attempts}


# ---------------------------------------------------------------- client
def make_client():
    """
    Azure OpenAI if AZURE_OPENAI_ENDPOINT is set, otherwise OpenAI directly.
    Reads from the environment or a .env file in the working directory.

    Azure:  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (https://<resource>.openai.azure.com),
            AZURE_OPENAI_DEPLOYMENT (your deployment name for gpt-5-mini),
            optional AZURE_OPENAI_API_VERSION. Without it the v1 endpoint is used;
            with it, the versioned AzureOpenAI client.
    OpenAI: OPENAI_API_KEY.
    Returns (client, model_or_deployment, provider_description).
    """
    from dotenv import load_dotenv
    load_dotenv()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        key = os.environ.get("AZURE_OPENAI_API_KEY")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not key or not deployment:
            raise SystemExit("AZURE_OPENAI_ENDPOINT is set but AZURE_OPENAI_API_KEY or "
                             "AZURE_OPENAI_DEPLOYMENT is missing.")
        version = os.environ.get("AZURE_OPENAI_API_VERSION")
        if version:
            from openai import AzureOpenAI
            client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=version)
            desc = f"azure:{endpoint} (api_version {version})"
        else:
            from openai import OpenAI
            client = OpenAI(base_url=endpoint.rstrip("/") + "/openai/v1/", api_key=key)
            desc = f"azure-v1:{endpoint}"
        return client, deployment, desc
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("No AZURE_OPENAI_ENDPOINT and no OPENAI_API_KEY found.")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), LLM_MODEL, "openai"


# ------------------------------------------------------------------ main
def load_cases(path):
    if path.endswith(".json"):
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return pd.read_csv(path)


def evaluate(cases_df, evaluator, model_label, pass_id):
    comps, case_rows = [], []
    for _, row in tqdm(cases_df.iterrows(), total=len(cases_df),
                       desc=f"{model_label} pass {pass_id}", unit="case"):
        y_true = parse_ground_truth_diagnoses(row[COL_TRUE])
        y_pred = parse_model_predicted_diagnoses(row[COL_PRED])
        if not y_true:
            continue
        found, first_rank, n_err = set(), None, 0
        first_tier = None
        for rank, pred in enumerate(y_pred, 1):
            pred_correct = False
            for ti, true in enumerate(y_true):
                r = evaluator.check_match(true, pred)
                comps.append({"model": model_label, "pass_id": pass_id, "case_id": row["case_id"],
                              "rank": rank, "ref_index": ti, "reference": true, "prediction": pred, **r})
                if r["verdict"] is True:
                    pred_correct = True
                    found.add(ti)
                elif r["tier"] == "LLM_ERROR":
                    n_err += 1
            if pred_correct and first_rank is None:
                first_rank = rank
                first_tier = next(c["tier"] for c in reversed(comps)
                                  if c["rank"] == rank and c["verdict"] is True
                                  and c["case_id"] == row["case_id"])
        raw_ref = row[COL_TRUE] if isinstance(row[COL_TRUE], str) else ""
        case_rows.append({
            "model": model_label, "pass_id": pass_id, "case_id": row["case_id"],
            "n_refs": len(y_true), "n_preds": len(y_pred),
            "top1": 1.0 if first_rank == 1 else 0.0,
            "hit_rate": 1.0 if found else 0.0,           # "top-5"
            "recall": len(found) / len(y_true),
            "mrr": 1 / first_rank if first_rank else 0.0,
            "first_match_rank": first_rank, "first_match_tier": first_tier,
            "n_llm_errors": n_err,
            "flag_semicolon_split": (";" in raw_ref) and not re.search(r"^\d+\.", raw_ref.strip()),
            "flag_not_five_preds": len(y_pred) != 5,
            "n_unnumbered_lines_dropped": n_dropped_lines(row[COL_PRED]),
            "y_true": json.dumps(y_true), "y_pred": json.dumps(y_pred),
        })
    return pd.DataFrame(comps), pd.DataFrame(case_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--fuzzy-threshold", default="90", help='number, or "off"')
    ap.add_argument("--pass-id", default="1")
    ap.add_argument("--cases", help="optional text file, one case_id per line")
    ap.add_argument("--outdir", default="adjudication_logs")
    a = ap.parse_args()

    client, model_name, provider = make_client()

    thr = None if a.fuzzy_threshold.lower() == "off" else float(a.fuzzy_threshold)
    cases = load_cases(a.input)
    if a.cases:
        keep = {int(x) for x in open(a.cases).read().split()}
        cases = cases[cases["case_id"].astype(int).isin(keep)]

    ev = LoggedHybridEvaluator(thr, client, model=model_name)
    print(f"Evaluating {a.model_label}: {len(cases)} cases, fuzzy threshold {a.fuzzy_threshold}, "
          f"pass {a.pass_id} | adjudicator {model_name} via {provider}")
    comps, case_df = evaluate(cases, ev, a.model_label, a.pass_id)
    print(f"Done. {ev.llm_calls} LLM calls, {ev.llm_errors} errors after retry.")
    comps["fuzzy_threshold"] = a.fuzzy_threshold
    case_df["fuzzy_threshold"] = a.fuzzy_threshold

    os.makedirs(a.outdir, exist_ok=True)
    tag = f"{re.sub(r'[^A-Za-z0-9]+', '_', a.model_label)}_thr{a.fuzzy_threshold}_pass{a.pass_id}"
    comps.to_csv(os.path.join(a.outdir, f"{tag}_comparisons.csv"), index=False)
    case_df.to_csv(os.path.join(a.outdir, f"{tag}_cases.csv"), index=False)
    meta = {
        "model_label": a.model_label, "input": a.input, "fuzzy_threshold": a.fuzzy_threshold,
        "pass_id": a.pass_id, "n_cases": len(case_df), "llm_model": model_name, "provider": provider,
        "llm_calls": ev.llm_calls, "llm_errors_after_retry": ev.llm_errors,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "summary": case_df[["top1", "hit_rate", "recall", "mrr"]].mean().round(4).to_dict(),
    }
    with open(os.path.join(a.outdir, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()