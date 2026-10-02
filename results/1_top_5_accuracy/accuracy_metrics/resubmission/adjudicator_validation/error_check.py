import pandas as pd, glob
D = "/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/1_top_5_accuracy/accuracy_metrics/resubmission/adjudicator_validation/"
c = pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/*_comparisons.csv")])
e = c[c.tier == "LLM_ERROR"]
print(len(e), "errored comparisons\n")
print(e.llm_error.str[:150].value_counts(), "\n")                       # what kind of error
print(e.groupby(["model", "case_id", "reference", "prediction"]).size()  # same pairs every pass?
        .sort_values(ascending=False).head(15).to_string(), "\n")

k = pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/*_cases.csv")])
risky = k[(k.n_llm_errors > 0) & (k.hit_rate == 0)]
print(len(risky), "case-runs where an error could have changed the top-5 verdict:")
print(risky[["model", "fuzzy_threshold", "pass_id", "case_id"]].to_string(index=False))