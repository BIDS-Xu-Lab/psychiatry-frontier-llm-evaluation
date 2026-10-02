#!/usr/bin/env bash
# run_validation.sh — run validate_adjudicator.py on all adjudication runs.
#
# Usage:   bash run_validation.sh
# Overrides (before "bash"): PYENV_ENV, PYTHON, VALIDATOR, ADJ_DIR, RATINGS,
#                            RESULTS_DIR, CANONICALS, EXPECTED
#
# Runs once per canonical setting (default: 90:1 and off:1). Each run uses ALL
# *_cases.csv files (for the by-setting agreement and self-consistency tables)
# and the canonical one for the headline, per-model, LOO and disagreement outputs.

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Run this with bash:  bash run_validation.sh   (not zsh, sh, or source)"
  return 1 2>/dev/null || exit 1
fi
set -euo pipefail

# ============================ EDIT THIS SECTION ============================
ADJ_DIR="${ADJ_DIR:-"/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/accuracy_metrics/resubmission/adjudicator_validation_rescored/"}"                 # where the *_cases.csv files are
RATINGS="${RATINGS:-"/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/2_evaluate_diagnostic_reasoning/clinician_annotated_reasoning_traces/processed_full_list.csv"}"            # clinician ratings
RESULTS_DIR="${RESULTS_DIR:-adjudicator_validation_results}"
VALIDATOR="${VALIDATOR:-"/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/code/1_top_5_accuracy/script_versions/calculate_accuracy/adjudicator_validation/validate_adjudicator.py"}"
CANONICALS="${CANONICALS:-90:1 off:1}"                   # headline settings to analyse
METRICS="${METRICS:-top1 hit_rate}"                      # top1 = primary comparison
EXPECTED="${EXPECTED:-24}"                               # 4 models x 6 runs
# Adjudicator label -> model_name in the ratings file (only where they differ)
RENAME=(
  "Claude Opus 4.5=Anthropic Claude Opus 4.5"
  "GPT-5.2=OpenAI GPT-5.2"
)
# ==========================================================================
 
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
 
# ------------------------------------------------------------- checks
[ -f "$VALIDATOR" ] || { echo "Missing validator: $VALIDATOR"; exit 1; }
[ -f "$RATINGS" ]   || { echo "Missing ratings file: $RATINGS"; exit 1; }
[ -d "$ADJ_DIR" ]   || { echo "Missing adjudication folder: $ADJ_DIR"; exit 1; }
 
FILES=()
while IFS= read -r -d '' f; do FILES+=("$f"); done \
  < <(find "$ADJ_DIR" -maxdepth 1 -name '*_cases.csv' -print0 | sort -z)
 
echo "Found ${#FILES[@]} *_cases.csv files in $ADJ_DIR"
if [ "${#FILES[@]}" -ne "$EXPECTED" ]; then
  echo "Expected $EXPECTED. Check the folder (or set EXPECTED=${#FILES[@]} if that's intended)."
  exit 1
fi
 
mkdir -p "$RESULTS_DIR"
 
# ---------------------------------------------------------------- run
for metric in $METRICS; do
  for canon in $CANONICALS; do
    out="$RESULTS_DIR/${metric}_canonical_${canon/:/_pass}"
    mkdir -p "$out"
    echo -e "\n========== $metric, canonical $canon -> $out =========="
    "$PYTHON" "$VALIDATOR" --ratings "$RATINGS" --adjudicator "${FILES[@]}" --metric "$metric" \
        --canonical "$canon" --outdir "$out" --rename "${RENAME[@]}" 2>&1 | tee "$out/summary.txt"
  done
done
 
echo -e "\nDone. Results in $RESULTS_DIR/ (each folder has summary.txt plus the CSV tables)."