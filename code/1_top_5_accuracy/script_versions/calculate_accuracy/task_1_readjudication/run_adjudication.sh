#!/usr/bin/env bash
# run_adjudication.sh — all adjudication runs for the validation (editorial point 4)
#
#   4 models x (fuzzy 90, passes 1-3  +  fuzzy off, pass 1) = 16 runs
#
# Usage:   bash run_adjudication.sh            # JOBS defaults to 4
#          JOBS=8 bash run_adjudication.sh     # more in parallel
#
# - Runs in parallel with xargs -P (works with macOS bash 3.2 and Linux).
# - Each run writes its own log to $LOGDIR/<tag>.log; the terminal shows one line
#   per run as it starts and finishes (tqdm bars go to the logs, since parallel
#   bars would interleave). Watch one live with: tail -f adjudication_logs/run_logs/<tag>.log
# - Resumable: a run whose *_meta.json already exists is skipped, so re-running
#   after an interruption only does what's missing.
# - Rate limits: failed calls retry with backoff inside the evaluator. If the logs
#   show many errors, lower JOBS.
# - If $RATINGS exists and every run succeeded, validate_adjudicator.py runs at the end.

set -euo pipefail

# ============================ EDIT THIS SECTION ============================
# Task 2 model-output files (the ones the reasoning traces came from), and the
# EXACT model_name string each one has in processed_full_list.csv. Same order.
INPUTS=(
  "/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/model_generated_diagnoses/main_experiment/predicted_diagnoses_claude-opus-4-5-20251101_20251215_225418.json"
  "/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/model_generated_diagnoses/main_experiment/predicted_diagnoses_deepseek-reasoner_20251215_215332.json"
  "/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/model_generated_diagnoses/main_experiment/predicted_diagnoses_gemini-3-pro-preview_20251217_184205.json"
  "/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/model_generated_diagnoses/main_experiment/predicted_diagnoses_gpt-5.2_20251218_122902.json"
)
LABELS=(
  "Claude Opus 4.5"
  "DeepSeek-V3.2"
  "Google Gemini 3 Pro"
  "GPT-5.2"
)
# ==========================================================================

JOBS="${JOBS:-4}"

# Python: use the pyenv environment's interpreter directly (no activation needed,
# which also makes it reliable inside parallel subshells).
PYENV_ENV="${PYENV_ENV:-mh-eval}"
if [ -z "${PYTHON:-}" ]; then
  if command -v pyenv >/dev/null 2>&1 && [ -x "$(pyenv root)/versions/$PYENV_ENV/bin/python" ]; then
    PYTHON="$(pyenv root)/versions/$PYENV_ENV/bin/python"
  else
    echo "Could not find pyenv environment '$PYENV_ENV'. Set PYTHON=/path/to/python or PYENV_ENV=<name>."
    exit 1
  fi
fi
echo "Using Python: $PYTHON"
 
# Paths — edit these defaults, or override on the command line (see header).
EVALUATOR="${EVALUATOR:-/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/code/1_top_5_accuracy/script_versions/calculate_accuracy/evaluate_accuracy_logged.py}"
VALIDATOR="${VALIDATOR:-/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/code/1_top_5_accuracy/script_versions/calculate_accuracy/adjudicator_validation/validate_adjudicator.py}"
OUTDIR="${OUTDIR:-/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/accuracy_metrics/resubmission/task_1_readjudication}"
LOGDIR="$OUTDIR/run_logs"
RATINGS="${RATINGS:-/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/2_evaluate_diagnostic_reasoning/clinician_annotated_reasoning_traces/processed_full_list.csv}"
CANONICAL="${CANONICAL:-90:1}"

# (threshold pass) combinations, in the order they should be scheduled
SETTINGS=("90 1" "off 1" "90 2" "off 2" "90 3" "off 3")

# ------------------------------------------------------------- checks
[ "${#INPUTS[@]}" -eq "${#LABELS[@]}" ] || { echo "INPUTS and LABELS differ in length"; exit 1; }
[ -f "$EVALUATOR" ] || { echo "Missing $EVALUATOR"; exit 1; }
for f in "${INPUTS[@]}"; do [ -f "$f" ] || { echo "Missing input: $f"; exit 1; }; done
if [ -z "${AZURE_OPENAI_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f .env ]; then
  echo "Warning: no AZURE_OPENAI_API_KEY / OPENAI_API_KEY and no .env in $(pwd); the evaluator will fail to authenticate."
fi
mkdir -p "$LOGDIR"

# Same tag the evaluator builds: non-alphanumeric runs -> "_"
tag_for() { printf '%s_thr%s_pass%s' "$(printf '%s' "$1" | sed -E 's/[^A-Za-z0-9]+/_/g')" "$2" "$3"; }

# ------------------------------------------------------ build job list
JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT
n_jobs=0; n_skip=0
for s in "${SETTINGS[@]}"; do
  set -- $s; thr="$1"; pass="$2"
  for i in "${!INPUTS[@]}"; do
    input="${INPUTS[$i]}"; label="${LABELS[$i]}"
    tag="$(tag_for "$label" "$thr" "$pass")"
    if [ -f "$OUTDIR/${tag}_meta.json" ]; then
      echo "[skip] $tag (already done)"; n_skip=$((n_skip + 1)); continue
    fi
    # One self-contained command per job; NUL-separated so labels with spaces survive.
    printf '%s\0' "echo \"[start] $tag\"; \
if \"$PYTHON\" \"$EVALUATOR\" --input \"$input\" --model-label \"$label\" \
--fuzzy-threshold $thr --pass-id $pass --outdir \"$OUTDIR\" > \"$LOGDIR/$tag.log\" 2>&1; \
then echo \"[done]  $tag\"; else echo \"[FAIL]  $tag  (see $LOGDIR/$tag.log)\"; exit 1; fi" >> "$JOBFILE"
    n_jobs=$((n_jobs + 1))
  done
done

echo "Running $n_jobs runs ($n_skip skipped), $JOBS at a time. Logs: $LOGDIR/"
start=$(date +%s)

status=0
if [ "$n_jobs" -gt 0 ]; then
  xargs -0 -n 1 -P "$JOBS" bash -c < "$JOBFILE" || status=$?
fi

elapsed=$(( $(date +%s) - start ))
echo "Finished in $((elapsed / 60)) min $((elapsed % 60)) s."

# ------------------------------------------------------------ summary
"$PYTHON" - "$OUTDIR" <<'PY'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_meta.json"))):
    m = json.load(open(f))
    rows.append((m["model_label"], m["fuzzy_threshold"], m["pass_id"], m["n_cases"],
                 m["llm_calls"], m["llm_errors_after_retry"], m["summary"]["hit_rate"]))
print(f"\n{'model':22s} {'thr':>4s} {'pass':>4s} {'cases':>5s} {'calls':>6s} {'errors':>6s} {'top-5':>6s}")
for r in rows:
    print(f"{r[0]:22s} {str(r[1]):>4s} {str(r[2]):>4s} {r[3]:5d} {r[4]:6d} {r[5]:6d} {r[6]:6.3f}")
if any(r[5] for r in rows):
    print("\n!! Some runs have LLM errors after retry. Re-run those (delete their *_meta.json) before validating.")
PY

if [ "$status" -ne 0 ]; then
  echo "One or more runs failed; fix and re-run this script (finished runs are skipped)."
  exit "$status"
fi

# --------------------------------------------------- optional validation
if [ -f "$RATINGS" ] && [ -f "$VALIDATOR" ]; then
  echo -e "\nAll runs complete. Running $VALIDATOR ..."
  "$PYTHON" "$VALIDATOR" --ratings "$RATINGS" --adjudicator "$OUTDIR"/*_cases.csv \
      --canonical "$CANONICAL" --outdir adjudicator_validation
else
  echo "Skipping validation ($RATINGS or $VALIDATOR not found here)."
fi
