import pandas as pd, glob
D = "/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/accuracy_metrics/resubmission/adjudicator_validation/"
r = pd.read_csv("/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/2_evaluate_diagnostic_reasoning/clinician_annotated_reasoning_traces/processed_full_list.csv")
r["yes"] = (r.diagnosis_match.str.strip().str.lower() == "yes").astype(int)
pairs = r.groupby(["case_id", "model_name"]).yes.agg(["sum", "mean"]).reset_index()
k = pd.concat(pd.read_csv(f) for f in glob.glob(f"{D}/*_thr90_pass1_cases.csv")).rename(columns={"model": "model_name"})
k["model_name"] = k.model_name.replace({"Claude Opus 4.5": "Anthropic Claude Opus 4.5", "GPT-5.2": "OpenAI GPT-5.2"})
m = pairs.merge(k, on=["case_id", "model_name"])
m["where"] = m.first_match_rank.map(lambda x: "no match" if pd.isna(x) else ("rank 1" if x == 1 else "rank 2+"))
print(m.groupby("where").agg(pairs=("mean", "size"), clinician_yes_rate=("mean", "mean"),
                             majority_yes=("sum", lambda s: (s >= 3).mean())).round(3))
maj = (m["sum"] >= 3).astype(int)
for col in ["top1", "hit_rate"]:
    print(col, "agreement with clinician majority:", round((m[col].astype(int) == maj).mean(), 3))