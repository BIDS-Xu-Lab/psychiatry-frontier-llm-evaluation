"""
case_mix_and_length.py — Editorial point 1 (case mix, vignette length) / R1-3 / R2-4.

First pass only. Grouping is rule-based from the reference-diagnosis text and
must be checked by a clinician (see the review sheet). Nothing here is final.

Inputs
  --detailed   any Task 1 *_diagnostic_evaluation_results_detailed.csv (has all
               196 vignettes, references, and source; identical across models)
  --rated      ambiguity_rating_sheet_with_vignettes_MSS.csv (gives the 30 rated IDs)
Outputs (in --outdir)
  vignette_case_mix_review.csv   one row per vignette, with blank columns for the clinician
  table_length.csv, table_case_mix_primary.csv, table_case_mix_any.csv, table_reference_flags.csv
"""
import argparse, ast, os, re
import pandas as pd

LITERATURE = {"dsm_5_tr_clinical_cases", "case_reports_in_psychiatry", "nejm",
              "jama_psychiatry", "jama"}

# Specifiers that would otherwise trigger the wrong chapter; masked first.
SPECIFIERS = [r"with (mood-(in)?congruent )?psychotic (features|symptoms)", r"with psychosis",
              r"without psychotic features", r"with catatonia", r"with panic( attacks)?",
              r"tic[- ]related", r"with suicidal ideation", r"high level of concern about suicide",
              r"(bipolar|depressive) type", r"\(mdd\)|\(ptsd\)|\(ocd\)|\(dmdd\)|\(asd\)|\(adhd\)|\(npsle\)|\(ied\)|\(arfid\)|\(sad\)|\(mad\)|\(iad\)"]

# (chapter, patterns) in PRIORITY order: earlier chapters claim text first.
# Chapters follow DSM-5-TR. Substance/medication-induced disorders go to the chapter
# of their presentation, as in DSM-5-TR; use/intoxication/withdrawal go to Substance.
RULES = [
    ("No diagnosis", [r"no diagnosis", r"normative stress reaction"]),
    ("Personality", [r"[\w/-]+ personality disorder"]),
    ("Paraphilic", [r"sexual sadism", r"fetishistic", r"pedophilic", r"voyeur", r"exhibitionis"]),
    ("Somatic symptom & related", [r"somatic symptom", r"illness anxiety", r"conversion",
        r"functional neurological", r"functional seizures", r"psychogenic nonepileptic",
        r"factitious", r"pseudocyesis"]),
    ("Medical (non-psychiatric)", [r"vitamin b12", r"lymphoma", r"cushing", r"lupus|npsle",
        r"encephalitis", r"leukoencephalopathy", r"hemorrhage", r"concussion", r"\btbi\b",
        r"vertigo", r"serotonin syndrome", r"neurosyphilis", r"pediatric autoimmune|pandas",
        r"charles bonnet", r"covid", r"cortical dysplasia", r"cri-du-chat"]),
    ("Bipolar & related", [r"delirious mania", r"bipolar[^;|()\n]*", r"cyclothymic",
        r"mood disorder due to[^;|()\n]*manic[^;|()\n]*"]),
    ("Neurocognitive", [r"delirium", r"neurocognitive", r"alzheimer", r"lewy", r"dementia",
        r"frontotemporal", r"pick.s disease", r"korsakov"]),
    ("Substance-related & addictive", [r"\b[\w/-]+ use disorder", r"\babuse\b", r"dependence",
        r"withdrawal", r"intoxication", r"gambling", r"substance related",
        r"alcohol use", r"hallucinogen persist\w* perception"]),
    ("Schizophrenia spectrum & psychotic", [r"schizo\w*[^;|()\n]*", r"psychotic disorder",
        r"psychosis", r"catatoni\w*", r"cataonia", r"delusional disorder"]),
    ("Trauma & stressor-related", [r"adjustment disorder[^;|,()\n]*(, acute stressor)?",
        r"post[- ]?traumatic", r"ptsd", r"acute stress", r"disinhibited social engagement",
        r"reactive attachment", r"(prolonged|complicated) grief"]),
    ("Depressive", [r"depressi\w*", r"dysthymi\w*", r"premenstrual dysphoric",
        r"disruptive mood dysregulation|dmdd", r"\bmdd\b", r"\(mad\)"]),
    ("Obsessive-compulsive & related", [r"obsessive[- ]compulsive", r"\bocd\b", r"hoarding",
        r"body dysmorphic", r"trichotillomania", r"excoriation|skin-picking"]),
    ("Dissociative", [r"dissociative", r"depersonali[sz]ation"]),
    ("Anxiety", [r"anxiety", r"panic disorder", r"phobia", r"agoraphobia", r"\(sad\)"]),
    ("Feeding & eating", [r"anorexia", r"bulimia", r"binge", r"\bpica\b",
        r"avoidant/restrictive food|arfid", r"eating disorder"]),
    ("Elimination", [r"enuresis", r"encopresis"]),
    ("Sleep-wake", [r"insomnia", r"hypersomnolence", r"sleep apnea", r"restless legs",
        r"rapid eye movement sleep|rem sleep"]),
    ("Sexual dysfunction", [r"sexual interest|sexual dysfunction|erectile"]),
    ("Gender dysphoria", [r"gender dysphoria"]),
    ("Disruptive, impulse-control & conduct", [r"oppositional", r"conduct disorder",
        r"intermittent explosive|\(ied\)"]),
    ("Neurodevelopmental", [r"autis\w*", r"intellectual", r"learning disorder",
        r"attention[- ]defici\w*|adhd", r"tourette", r"global developmental delay",
        r"neurodevelopmental"]),
    ("Other (conditions for further study)", [r"suicidal behavior disorder"]),
]


def classify(text):
    """Return [(position, chapter), ...] for every chapter matched in one reference string."""
    t = text.lower()
    for sp in SPECIFIERS:
        t = re.sub(sp, lambda m: " " * len(m.group(0)), t)
    hits = []
    for chapter, pats in RULES:
        for p in pats:
            for m in re.finditer(p, t):
                if m.group(0).strip():
                    hits.append((m.start(), chapter))
                    t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    return sorted(hits)


def flags(raw, refs, primary_chapter, n_in_first):
    f = []
    low = raw.lower()
    if re.search(r"\bor\b|\bvs\.?\b|\+/-", low):
        f.append("alternatives in reference (or / vs / +/-)")
    if re.search(r"provisional|rule out|history of", low):
        f.append("provisional / rule-out / history")
    if any(len({c for _, c in classify(r)}) > 1 for r in refs):
        f.append("one reference string spans 2+ chapters (check: may be several diagnoses scored as one)")
    if primary_chapter == "Medical (non-psychiatric)":
        f.append("primary reference is not a mental disorder")
    if re.search(r"diagnosis on admission|final diagnos|ms\.|mr\.", low):
        f.append("parsing anomaly (admission/final or two patients)")
    return "; ".join(f)


def describe(s):
    return pd.Series({"n": len(s), "median": s.median(), "IQR": f"{s.quantile(.25):.0f}–{s.quantile(.75):.0f}",
                      "min": s.min(), "max": s.max()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detailed", required=True)
    ap.add_argument("--rated", required=True)
    ap.add_argument("--outdir", default="case_mix")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    d = pd.read_csv(a.detailed)
    rated = set(pd.read_csv(a.rated, encoding="mac_roman")["case_id"])
    d["refs"] = d["y_true"].apply(ast.literal_eval)
    d["origin"] = d["source"].map(lambda s: "Literature" if s in LITERATURE else "Clinician-authored")
    d["subset"] = d["case_id"].map(lambda c: "Rated (30)" if c in rated else "Not rated (166)")
    d["words"] = d["vignette"].str.split().str.len()

    rows = []
    for _, r in d.iterrows():
        per_ref = [classify(x) for x in r.refs]
        first = per_ref[0] if per_ref else []
        primary = first[0][1] if first else "UNMATCHED"
        any_ch = sorted({c for h in per_ref for _, c in h})
        rows.append({"case_id": r.case_id, "source": r.source, "origin": r.origin, "subset": r.subset,
                     "words": r.words, "n_refs_parsed": len(r.refs),
                     "references": " || ".join(x.replace("\n", " ") for x in r.refs),
                     "primary_chapter_auto": primary, "all_chapters_auto": "; ".join(any_ch),
                     "unmatched_ref": any(not h for h in per_ref),
                     "reference_flags": flags(r.diagnosis if isinstance(r.diagnosis, str) else "",
                                              r.refs, primary, len(first)),
                     # ---- for the clinician ----
                     "CONFIRM_primary_chapter": "", "CONFIRM_all_chapters": "",
                     "TEMPORAL_COURSE": "", "COMMON_or_RARE": "", "notes": ""})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.outdir, "vignette_case_mix_review.csv"), index=False)

    # ---- tables ----
    length = pd.concat({
        "All": describe(out.words),
        **{k: describe(g.words) for k, g in out.groupby("origin")},
        **{k: describe(g.words) for k, g in out.groupby("subset")},
    }, axis=1).T
    length.to_csv(os.path.join(a.outdir, "table_length.csv"))

    prim = pd.crosstab(out.primary_chapter_auto, out.origin, margins=True, margins_name="Total")
    prim["Rated (30)"] = out[out.subset == "Rated (30)"].primary_chapter_auto.value_counts()
    prim = prim.fillna(0).astype(int)
    prim.loc["Total", "Rated (30)"] = int((out.subset == "Rated (30)").sum())
    prim = prim.sort_values("Total", ascending=False)
    prim.to_csv(os.path.join(a.outdir, "table_case_mix_primary.csv"))

    long = out.assign(ch=out.all_chapters_auto.str.split("; ")).explode("ch").reset_index(drop=True)
    anyt = pd.crosstab(long.ch, long.origin, margins=True, margins_name="Total").sort_values("Total", ascending=False)
    anyt.to_csv(os.path.join(a.outdir, "table_case_mix_any.csv"))

    fl = out.assign(f=out.reference_flags.str.split("; ")).explode("f").reset_index(drop=True)
    fl = fl[fl.f.fillna("") != ""]
    flt = pd.crosstab(fl.f, fl.origin, margins=True, margins_name="Total")
    flt.to_csv(os.path.join(a.outdir, "table_reference_flags.csv"))

    print(length.to_string(), "\n")
    print(prim.to_string(), "\n")
    print(anyt.to_string(), "\n")
    print(flt.to_string(), "\n")
    print("unmatched references:", out[out.unmatched_ref][["case_id", "references"]].to_string(index=False))
    print("multi-reference vignettes:", (out.n_refs_parsed > 1).sum(), "of", len(out))


if __name__ == "__main__":
    main()
