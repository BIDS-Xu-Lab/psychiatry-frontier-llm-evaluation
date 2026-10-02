import pandas as pd, glob
D = "/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/accuracy_metrics/resubmission/adjudicator_validation"
k = pd.concat(pd.read_csv(f) for f in glob.glob(f"{D}/*_thr90_pass1_cases.csv"))
print(k.groupby("model").flag_not_five_preds.sum())
print(k[k.flag_not_five_preds][["model", "case_id", "n_preds", "first_match_rank", "top1"]].to_string(index=False))