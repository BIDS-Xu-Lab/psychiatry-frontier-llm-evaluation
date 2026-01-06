# Calculate accuracy metrics for model-predicted diagnoses against ground truth (n=196) using hybrid fuzzy + LLM approach
import re
from rapidfuzz import fuzz
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import json
import pandas as pd
from tqdm import tqdm

# Define constants for DataFrame columns of interest
COL_TRUE = 'diagnosis'
COL_PRED = 'model_diagnosis'

# Load API key from environment variable
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Helper functions: Parse ground truth diagnoses and model-predicted diagnoses strings from DataFrame into lists
def parse_ground_truth_diagnoses(diagnosis_str) -> list:
    """
    Converts '1. Diagnosis A\n2. Diagnosis B' into ['Diagnosis A', 'Diagnosis B']
    """
    # Handle empty or non-string inputs
    if not isinstance(diagnosis_str, str): return []

    # Check if the diagnosis string is a numbered list (starts with "1." or similar)
    if re.search(r'^\d+\.', diagnosis_str.strip()):
        # Split by newline and map each diagnosis into a list
        diagnoses = []
        for line in diagnosis_str.strip().split('\n'):
            # Remove the numbering and any leading/trailing whitespace
            match = re.match(r'\d+\.\s+(.*)', line)  # Regex to capture text after numbering
            if match:
                diagnoses.append(match.group(1).strip())  # Add the cleaned diagnosis to the list
            else:
                diagnoses.append(line.strip())  # If not numbered, just add the line as-is
        return diagnoses
    else:
        # If not a numbered list, split by semicolons
        diagnoses = re.split(r';', diagnosis_str)
        return [diag.strip() for diag in diagnoses if diag.strip()]


def parse_model_predicted_diagnoses(model_diagnoses_str) -> list:
    """
    Converts '1. Diagnosis A\n2. Diagnosis B' into ['Diagnosis A', 'Diagnosis B']
    """
    # Handle empty or non-string inputs
    if not isinstance(model_diagnoses_str, str): return []

    diagnoses = []
    for line in model_diagnoses_str.strip().split('\n'):
        # Remove the numbering and any leading/trailing whitespace
        match = re.match(r'\d+\.\s+(.*)', line)
        if match:
            diagnoses.append(match.group(1).strip())
        else:
            # If not numbered, just add the line
            diagnoses.append(line.strip())
    return diagnoses


# Compare ground truth and predicted diagnoses using hybrid fuzzy + LLM approach for one case
# Define hybrid evaluator class for one case
class HybridEvaluator:
    def __init__(self, fuzzy_threshold=90, llm_model="gpt-5-mini"):
        self.fuzzy_threshold = fuzzy_threshold
        self.llm_model = llm_model
        self.cache = {}  # The cache prevents paying for the same comparison twice. Structure: {"True Term || Pred Term": True/False}
        self.llm_calls = 0

    def check_match(self, true_diag, pred_diag):
        """
        Returns True if match, False if not.
        Uses Fuzzy first, then falls back to LLM.
        """
        # 1. Normalize strings for easier comparison
        t = true_diag.lower().strip()
        p = pred_diag.lower().strip()

        # 2. TIER 1: Fuzzy String Matching (Free & Fast)
        # token_set_ratio handles reordering (e.g. "Type 2 Diabetes" == "Diabetes Type 2")
        fuzzy_score = fuzz.token_set_ratio(t, p)
        if fuzzy_score >= self.fuzzy_threshold:
            return True

        # 3. TIER 2: LLM Judge (Semantic)
        # Only runs if fuzzy score is low (e.g., < 95)
        # Check cache first
        cache_key = f"{t} || {p}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Call the LLM to act as a strict medical adjudicator
        # print(f"Fuzzy threshold exceeded. Invoking LLM for: '{t}' vs '{p}'")
        is_match = self._ask_llm(t, p)

        # Update Cache
        self.cache[cache_key] = is_match
        self.llm_calls += 1
        return is_match

    def _ask_llm(self, t, p):
        # Define prompt for LLM-as-a-judge
        prompt = f"""

        Your task is to as a strict medical adjudicator specializing in psychiatry and identify whether the predicted diagnosis is clinically equivalent to (or a valid subclass of) the true diagnosis. Your standards are exacting, and you must consider the nuances of each diagnosis carefully. As much as possible, adhere to the diagnostic language laid out in the DSM-5-TR, and utilize the included ICD-10 F-codes to aid your determination.

        True Diagnosis: "{t}"
        Predicted Diagnosis: "{p}"

        Return JSON ONLY: {{ "match": <true/false> }}
        """

        # Define response schema
        class DiagnosisMatch(BaseModel):
            match: bool  # True if match, False if not

        # Call the LLM judge
        try:
            response = client.responses.parse(
                model="gpt-5-mini",
                input=[
                    {
                          "role": "user",
                          "content": prompt
                    }
                ],
                text_format=DiagnosisMatch,
            )
            result = json.loads(response.output[1].content[0].text)
            return result.get("match", False)  # Default to False if key missing
        except Exception as e:
            print(f"LLM Error: {e}")
            return False


# Load cases from JSON to Pandas DataFrame
model_results_path = "../../../../results/top_5_accuracy/predicted_diagnoses/"
models = os.listdir(model_results_path)

# Evaluate all models inside the folder
for model in models:
    # Load model results to a Pandas DataFrame
    results_path = os.path.join(model_results_path, model)

    with open(results_path, 'r') as f:
        cases = json.load(f)

    cases_df = pd.DataFrame(cases)

    # Initialize Evaluator
    evaluator = HybridEvaluator(fuzzy_threshold=90, llm_model="gpt-5-mini")

    results = []

    # Iterate through DataFrame
    print(f"Starting evaluation for {model}...")
    for index, row in tqdm(cases_df.iterrows(), total=len(cases_df)):

        # 1. Parse Data
        y_true = parse_ground_truth_diagnoses(row[COL_TRUE])
        y_pred = parse_model_predicted_diagnoses(row[COL_PRED])

        # If no ground truth, skip
        if not y_true:
            continue

        # 2. Analyze Matches
        # We map which TRUE diagnoses were found in the PRED list
        found_indices = set()
        first_match_rank = None  # For MRR

        # Iterate through predictions (Order matters for Rank!)
        for rank_idx, pred_item in enumerate(y_pred):
            current_rank = rank_idx + 1  # 1-based rank

            # Check against ALL true items
            is_this_pred_correct = False

            for true_idx, true_item in enumerate(y_true):
                # THE HYBRID CHECK
                if evaluator.check_match(true_item, pred_item):
                    is_this_pred_correct = True
                    found_indices.add(true_idx)

            # If this prediction was a match, and it's the first one we've seen...
            if is_this_pred_correct and first_match_rank is None:
                first_match_rank = current_rank

        # 3. Calculate Metrics

        # Hybrid Recall@5: % of true diagnoses found
        recall_score = len(found_indices) / len(y_true)

        # Hybrid Hit Rate: Did we find at least one?
        hit_rate = 1.0 if len(found_indices) > 0 else 0.0

        # Hybrid MRR: 1 / Rank of first match
        mrr_score = (1 / first_match_rank) if first_match_rank else 0.0

        # Hybrid Top-1: Did the very first prediction match *any* truth?
        # We can check if Rank 1 was the first match
        top1_score = 1.0 if first_match_rank == 1 else 0.0

        results.append({
            "case_id": row['case_id'],
            "y_true": y_true,
            "y_pred": y_pred,
            "hybrid_top1": top1_score,
            "hybrid_hit_rate": hit_rate,
            "hybrid_recall": recall_score,
            "hybrid_mrr": mrr_score
        })

    results_df = pd.DataFrame(results)
    final_df = pd.concat([cases_df.reset_index(drop=True), results_df], axis=1)

    print(f"Done! Made {evaluator.llm_calls} calls to LLM.")

    # 1. Aggregate Statistics
    stats = {
        "Metric": ["Top-1 Accuracy", "Top-5 Hit Rate", "Recall@5", "MRR"],
        "Score": [
            results_df['hybrid_top1'].mean(),
            results_df['hybrid_hit_rate'].mean(),
            results_df['hybrid_recall'].mean(),
            results_df['hybrid_mrr'].mean()
        ]
    }
    stats_df = pd.DataFrame(stats)

    # Display nicely formatted percentages
    print("\n=== FINAL DIAGNOSTIC PERFORMANCE (Mean Scores) ===")
    stats_df.style.format({"Score": "{:.2%}"})

    # Export stats to CSV
    stats_df.to_csv(f"{model}_diagnostic_performance_summary.csv", index=False)
    print(f"\nSaved performance summary to '{model}_diagnostic_performance_summary.csv'")

    # 2. Inspecting Failures
    # Return rows where Hit Rate was 0 (Total Misses)
    misses = final_df[final_df['hybrid_hit_rate'] == 0]
    print(f"\nTotal Cases Completely Missed: {len(misses)}")
    if len(misses) > 0:
        print("Example Miss:")
        print(misses[[COL_TRUE, COL_PRED]].iloc[0])

    # 3. Export to CSV
    final_df.to_csv(f"{model}_diagnostic_evaluation_results_detailed.csv", index=False)
    print(f"\nSaved detailed results to '{model}_diagnostic_evaluation_results_detailed.csv'")
