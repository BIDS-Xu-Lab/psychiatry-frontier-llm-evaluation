#!/usr/bin/env python3
"""
Join the coded blind sheet to the key and produce the cross-tabulations
needed for section 3 of the rebuttal letter.

    python analyze_coding.py --coded coding_sheet_BLIND.csv --key coding_sheet_KEY.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats

NON_EXACT = ["SYNONYM", "SPECIFIER", "ETIOLOGY", "PARTIAL", "ALTERNATIVE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coded", default="coding_sheet_BLIND.csv")
    ap.add_argument("--key", default="coding_sheet_KEY.csv")
    args = ap.parse_args()

    coded = pd.read_csv(args.coded)
    key = pd.read_csv(args.key)
    df = coded.merge(key, on="row_key", how="inner")

    df["CODE_relation"] = df["CODE_relation"].astype(str).str.strip().str.upper()
    blank = (df.CODE_relation == "") | (df.CODE_relation == "NAN")
    if blank.any():
        print(f"WARNING: {blank.sum()} rows are uncoded. Finish them first.\n")
        df = df[~blank]

    print(f"Coded rows: {len(df)}\n")

    # ---- 1. The main cross-tab ----
    print("=" * 66)
    print("1. RELATION x STATUS  (counts)")
    print("=" * 66)
    ct = pd.crosstab(df.CODE_relation, df.status)
    for c in ["SPLIT", "unanimous_YES", "unanimous_NO"]:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[["SPLIT", "unanimous_YES", "unanimous_NO"]]
    print(ct.to_string())

    print("\ncolumn percentages (what each status is made of):")
    print((100 * ct / ct.sum()).round(1).to_string())

    # ---- 2. The claim in one number ----
    print("\n" + "=" * 66)
    print("2. THE CLAIM: disagreement concentrates in non-verbatim matches")
    print("=" * 66)
    df["non_exact"] = df.CODE_relation.isin(NON_EXACT)
    for st in ["SPLIT", "unanimous_YES"]:
        s = df[df.status == st]
        if len(s):
            print(f"  {st:15s}  non-exact: {s.non_exact.sum():3d}/{len(s):3d} "
                  f"= {100*s.non_exact.mean():.1f}%")
    print("\n  The letter's sentence holds if the first figure is much larger.")

    # ---- 3. Association test, with the caveat ----
    sp = df[df.status.isin(["SPLIT", "unanimous_YES"])]
    if len(sp):
        tab = pd.crosstab(sp.non_exact, sp.status)
        if tab.shape == (2, 2):
            odds, p = stats.fisher_exact(tab.values)
            print(f"\n  Fisher exact (unclustered): OR = {odds:.2f}, p = {p:.4g}")
            print("  CAVEAT: the 120 rows are 30 cases x 4 models, so rows sharing a")
            print("  case are not independent and this p-value is optimistic. For the")
            print("  manuscript, refit as a mixed model clustered on case:")
            print("      glmer(split ~ non_exact + (1|case_id), family=binomial)")
            print("  For the letter, the descriptive contrast above is sufficient.")

    # ---- 4. Rubric vs clinical ambiguity ----
    print("\n" + "=" * 66)
    print("3. RUBRIC vs CLINICAL AMBIGUITY among disputed traces")
    print("=" * 66)
    s = df[df.status == "SPLIT"].copy()
    s["rubric"] = s.CODE_rubric_ambiguity.astype(str).str.strip().str.upper().eq("Y")
    if len(s):
        print(f"  attributable to unclear instructions : {s.rubric.sum()}/{len(s)} "
              f"({100*s.rubric.mean():.1f}%)")
        print(f"  attributable to clinical judgment    : {(~s.rubric).sum()}/{len(s)} "
              f"({100*(~s.rubric).mean():.1f}%)")
        print("\n  Report these separately. Rubric ambiguity is fixable with clearer")
        print("  instructions; clinical ambiguity is a property of the reference")
        print("  standard, which is the point section 3 makes.")

    # ---- 5. By model, for the supplement ----
    print("\n" + "=" * 66)
    print("4. BY MODEL")
    print("=" * 66)
    print(pd.crosstab(df.model, df.status).to_string())

    df.to_csv("coding_joined.csv", index=False)
    print("\nWrote coding_joined.csv")


if __name__ == "__main__":
    main()
