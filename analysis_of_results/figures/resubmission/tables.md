# Tables — NEJM AI resubmission

All values are VERIFIED from `COMPLETE_ANALYSIS_DOCUMENT_v2.md` unless marked.
`[PENDING]` = not yet computed. `⟨…⟩` = note to you, not table content.
Markdown pastes into Word as a table via Paste Special → or through pandoc.

Supplement numbering below is a **proposal** — one scheme, sections and tables
aligned — to resolve the R2-14 inconsistency. Renumber to taste, but keep one scheme.

---

## MAIN TEXT

### Table 1. Diagnostic accuracy of four large language models on 196 psychiatric case vignettes

| Model | Top-1 accuracy (95% CI) | Top-5 accuracy (95% CI) | Recall@5 (95% CI) | Mean reciprocal rank (95% CI) | Adjudication spread, top-5* |
|---|---|---|---|---|---|
| Claude Opus 4.5 | 0.638 (0.568–0.702) | 0.801 (0.740–0.851) | 0.731 [PENDING] | 0.710 [PENDING] | [PENDING] |
| DeepSeek-V3.2 | 0.566 (0.496–0.634) | 0.770 (0.707–0.824) | 0.691 [PENDING] | 0.657 [PENDING] | [PENDING] |
| GPT-5.2 | 0.607 (0.537–0.673) | 0.735 (0.669–0.792) | 0.684 [PENDING] | 0.665 [PENDING] | [PENDING] |
| Gemini 3 Pro | 0.566 (0.496–0.634) | 0.730 (0.663–0.787) | 0.662 [PENDING] | 0.630 [PENDING] | [PENDING] |

Accuracy determined by automated adjudication (GPT-5-mini), fuzzy string-matching tier disabled; values with the tier enabled are in Table S5. Correctness against the primary reference diagnosis for each vignette.
\* Range of top-5 accuracy across repeated adjudication passes over identical model outputs (Methods).

⟨The top-1 and top-5 intervals shown are **Wilson score intervals** computed from the integer counts implied by the reported proportions (125/196, 157/196, etc. — all four models reconcile to whole numbers, which is a useful check). They are placeholders for the bootstrap intervals your Methods promise; a case-level percentile bootstrap will land within ±0.005 of these. Recall@5 and MRR are not proportions and need the pipeline. If the bootstrap and Wilson disagree by more than that, something is off in one of them.⟩

---

### Table 2. Clinician-adjudicated accuracy and clinician-rated reasoning quality on the 30-vignette subset

| Model | Clinician-adjudicated top-5 accuracy | Diagnostic reasoning, mean ± SD (0–4) | Data extraction, mean ± SD (0–4) |
|---|---|---|---|
| Claude Opus 4.5 | 0.827 | 3.60 ± 0.45 | 3.67 ± 0.38 |
| Gemini 3 Pro | 0.820 | 3.12 ± 0.60 | 3.15 ± 0.69 |
| DeepSeek-V3.2 | 0.827 | 2.90 ± 0.63 | 2.97 ± 0.61 |
| GPT-5.2 | 0.767 | 2.37 ± 0.52 | 2.21 ± 0.50 |

Five psychiatrists, blinded to model identity, each rated all 120 vignette–model pairs (600 ratings). Accuracy is the proportion of the 150 rater determinations per model (30 vignettes × 5 raters) in which the reference diagnosis was judged present in the differential. SD is across the 30 per-vignette means. Rows are ordered by diagnostic reasoning score. All five raters independently produced this rank ordering. Pairwise within-case comparisons with Holm correction are shown in Figure 2 and Table S3; interrater agreement in Table S2.

⟨This is the dissociation table. Ordering by reasoning puts 0.827 in rows 1 and 3 with 0.70 points between them — that's the point. SDs are rounded from the current Table 2 (0.446 → 0.45 etc.); three decimals on a 0–4 scale overstate precision.⟩

---

## SUPPLEMENT — proposed unified numbering

| Section | Content | Tables / figures |
|---|---|---|
| S1 | Vignette screening criteria, instructions, per-vignette scores | Table S1 |
| S2 | Full prompts, verbatim | — |
| S3 | Pairwise comparisons and equivalence testing | Table S3A, S3B |
| S4 | Hyperparameters by provider | Table S4 |
| S5 | Adjudication pipeline: fuzzy tier, refusals, repeated passes | Table S5A, S5B |
| S6 | Clinician rating rubric | — |
| S7 | Comparison with trainee clinicians | Table S7 |
| S8 | Memorization robustness | Figure S8, Table S8A, S8B |
| S9 | Qualitative commentary and safety concerns | Table S9A, S9B, S9C |
| S10 | Computable trace features | Table S10A, S10B |
| S11 | Run-to-run reproducibility | Table S11A, S11B |

⟨Current body references: Sections S1, S2, S4, S5, S6, S7 and Tables S1A/B, S2A/C, S3, S4 and Figure S1. Under this scheme the body edits are: "Table S1A/S1B, Figure S1" → S8; "Table S2A/B/C" → S9; "Table S3" → S3A; "Table S4" (zero-code shares) → S10B. Everything else keeps its number.⟩

---

### Table S2. Interrater agreement among five psychiatrists on the 30-vignette subset

| Measure | Statistic | Value (95% CI) |
|---|---|---|
| Diagnostic correctness | Mean pairwise Cohen's κ, 10 rater pairs | 0.626 (range 0.46–0.83) |
| Diagnostic correctness | Mean pairwise raw agreement | 0.767 |
| Diagnostic correctness | Traces with non-unanimous verdict | 28 / 120 (23.3%) |
| Diagnostic reasoning | ICC(A,1), single rater | 0.41 |
| Diagnostic reasoning | ICC(A,k), five-rater average | 0.773 (0.702–0.831) |
| Data extraction | ICC(A,1), single rater | 0.51 |
| Data extraction | ICC(A,k), five-rater average | 0.840 (0.790–0.881) |
| Diagnostic reasoning, variance components | Case / rater / residual | 0.129 / 0.002 / 0.625 |

⟨Label is **mean pairwise Cohen's κ**, per §7 — not Fleiss's κ as in the body. Fix the body to match, or compute Fleiss's and report that instead; don't keep both labels for one number. Single-rater ICCs are consistent with the averaged values by Spearman–Brown (0.41 → 0.777; 0.51 → 0.839).⟩

---

### Table S3A. Pairwise within-case comparisons of the four models on the 30-vignette subset

| Comparison | Δ reasoning quality (95% CI) | Holm-adjusted *P* | Δ clinician-adjudicated accuracy (95% CI) | Holm-adjusted *P* |
|---|---|---|---|---|
| Claude Opus 4.5 vs GPT-5.2 | +1.23 (1.04 to 1.43) | <0.001 | +0.060 (−0.079 to 0.199) | n.s. |
| Gemini 3 Pro vs GPT-5.2 | +0.75 (0.52 to 0.99) | <0.001 | +0.053 (−0.052 to 0.159) | n.s. |
| Claude Opus 4.5 vs DeepSeek-V3.2 | +0.70 (0.50 to 0.90) | <0.001 | 0.000 (−0.071 to 0.071) | n.s. |
| DeepSeek-V3.2 vs GPT-5.2 | +0.53 (0.27 to 0.80) | 0.0006 | +0.060 (−0.090 to 0.210) | n.s. |
| Claude Opus 4.5 vs Gemini 3 Pro | +0.48 (0.29 to 0.67) | 0.0001 | +0.007 (−0.090 to 0.104) | n.s. |
| DeepSeek-V3.2 vs Gemini 3 Pro | −0.22 (−0.52 to 0.08) | 0.138 | +0.007 (−0.117 to 0.130) | n.s. |

Differences are first model minus second. Paired *t*-intervals on per-case means (df = 29), confirmed by bootstrap percentile intervals. Six comparisons per measure, Holm–Bonferroni corrected.

⟨The accuracy Holm-adjusted *P* values aren't in the record as numbers — only "0 of 6 separated." Fill from the pipeline or drop the column and keep the CIs, which all cross zero.⟩

---

### Table S3B. Equivalence testing of accuracy differences by two one-sided tests

| Comparison | Δ accuracy | 90% CI | Smallest margin at which equivalence holds | Bootstrap margin |
|---|---|---|---|---|
| Claude Opus 4.5 vs DeepSeek-V3.2 | 0.000 | −0.059 to 0.059 | **0.059** (0.10 after Holm) | 0.060 |
| Claude Opus 4.5 vs Gemini 3 Pro | +0.007 | −0.074 to 0.087 | 0.087 | 0.087 |
| DeepSeek-V3.2 vs Gemini 3 Pro | +0.007 | −0.096 to 0.109 | 0.109 | 0.107 |
| Gemini 3 Pro vs GPT-5.2 | +0.053 | −0.034 to 0.141 | 0.141 | 0.140 |
| Claude Opus 4.5 vs GPT-5.2 | +0.060 | −0.055 to 0.175 | 0.175 | 0.173 |
| DeepSeek-V3.2 vs GPT-5.2 | +0.060 | −0.064 to 0.184 | 0.184 | 0.180 |

| Fixed margin | Pairs equivalent |
|---|---|
| ±0.050 | 0 / 6 |
| ±0.075 | 0 / 6 |
| ±0.100 | 1 / 6 (Claude Opus 4.5 vs DeepSeek-V3.2) |
| ±0.150 | 3 / 6 |

The 90% CI is the TOST-equivalent interval; equivalence at margin *m* holds when the interval lies within ±*m*. The Claude Opus 4.5 vs DeepSeek-V3.2 pair was selected for the primary claim because its accuracies matched, which is post hoc; the Holm-adjusted margin is therefore the one reported in the text.

---

### Table S5A. Adjudication pipeline: matching tiers and content-filter refusals

| Item | Value |
|---|---|
| Unique predicted diagnosis strings adjudicated | 926 |
| Accepted by fuzzy string-matching tier | 22 (2.4%) |
| Case–model pairs withheld by provider content filter, per run | 9–10 (≈8%) |
| Handling of withheld outputs | Scored as non-matches |

⟨Add a row per model for accuracy with the fuzzy tier enabled vs disabled once the pipeline is re-run — the Methods promise both.⟩

### Table S5B. Adjudication variance: repeated passes over identical model outputs, 30-vignette subset

| Model / run | Top-5, threshold 90 | Top-5, threshold 100 | Change (points) | Note |
|---|---|---|---|---|
| Claude Opus 4.5, run 1 | 0.833 | 0.867 | +3.4 | rises under tighter threshold |
| GPT-5.2, run 1 | 0.793 | 0.828 | +3.5 | rises under tighter threshold |
| Claude Opus 4.5, run 2 | 0.867 | 0.800 | −6.7 | |
| DeepSeek-V3.2, run 1 | 0.900 | 0.833 | −6.7 | |
| DeepSeek-V3.2, run 2 | 0.867 | 0.733 | −13.4 | |
| GPT-5.2, run 2 | 0.897 | 0.828 | −6.9 | |
| Gemini 3 Pro, both runs | — | — | 0.0 | |
| **Mean absolute change** | | | **5.1** | |

Tightening the matching threshold cannot raise accuracy under the metric's construction; the two rises indicate adjudicator non-determinism between passes rather than a threshold effect.

⟨This is the existing two-pass comparison. The 3–5 passes per threshold committed in the letter would replace this table with per-model mean and range across passes.⟩

---

### Table S7. Diagnostic accuracy of the best-performing model and two psychiatry residents on the 30-vignette subset (formerly Table 3)

| | Top-1 accuracy | Top-5 accuracy | Recall@5 | Mean reciprocal rank |
|---|---|---|---|---|
| Claude Opus 4.5 | 0.733 | 0.867 | 0.776 | 0.800 |
| Psychiatry residents (*n* = 2, mean) | 0.617 | 0.767 | 0.656 | 0.683 |

Accuracy determined by automated adjudication, identically for model and residents. Note that Claude Opus 4.5's clinician-adjudicated top-5 accuracy on the same vignettes was 0.827 (Table 2); the difference reflects the adjudication method, not the outputs.

⟨That last sentence is your first adjudicator-validation datum, stated where a reader would otherwise notice the discrepancy.⟩

---

### Table S10A. Computable trace features: model-level means and associations with reasoning ratings

| Model | Reasoning (0–4) | Words per trace | Distinct ICD-10 codes per trace | Self-correction phrases per trace | Traces naming no code | Codes per trace, when any | Words per code |
|---|---|---|---|---|---|---|---|
| Claude Opus 4.5 | 3.60 | 1,088 | 2.83 | 0.73 | 33.3% | 4.25 | 384 |
| Gemini 3 Pro | 3.12 | 431 | 3.97 | 1.40 | 23.3% | 5.17 | 109 |
| DeepSeek-V3.2 | 2.90 | 620 | 3.50 | 3.77 | 53.3% | 7.50 | 177 |
| GPT-5.2 | 2.37 | 900 | 8.77 | 3.07 | 0.0% | 8.77 | 103 |

| Feature | Model-level Spearman ρ (*n* = 4) | Exact *P* |
|---|---|---|
| ICD-10 codes per trace | −0.80 | 0.333 |
| Self-correction phrases per trace | −0.80 | 0.333 |
| Self-correction phrases per 1,000 words | −0.80 | 0.333 |
| ICD-10 codes per 1,000 words | −0.40 | 0.750 |
| Trace length | +0.20 | 0.917 |

Features computed on the generation run rated by clinicians. At *n* = 4 the smallest attainable two-sided *P* is 0.083; model-level associations are descriptive. Of 572 distinct code-shaped strings across traces, 97.9% were ICD-10 Chapter F codes and none were non-codes.

### Table S10B. Computable trace features: within-model associations across 120 traces

| Feature | β per unit (95% CI) | *P* | Between-model spread | Predicted reasoning gap | Observed gaps |
|---|---|---|---|---|---|
| ICD-10 codes per trace | +0.019 (−0.007 to 0.045) | 0.145 | 5.93 | +0.12 | 0.48–1.23 |
| Self-correction phrases per trace | +0.042 (0.009 to 0.075) | 0.014 | 3.03 | +0.13 | |
| ICD-10 codes per 1,000 words | −0.012 (−0.019 to −0.006) | 0.0001 | 11.39 | −0.14 | |
| Self-correction phrases per 1,000 words | −0.009 (−0.043 to 0.026) | 0.634 | 5.40 | −0.05 | |
| Words (per 100) | +0.041 (0.021 to 0.060) | <0.0001 | 6.57 | +0.27 | |

Linear mixed model, `reasoning ~ feature + model + (1 | case)`. Predicted gap is the within-model slope multiplied by the largest between-model difference in the feature: the most any feature could explain of the observed between-model reasoning differences if the within-model relationship held across models.

⟨Note the third row: codes per 1,000 words is significantly *negative* within model — same sign as the model-level ρ, not a reversal. The Results sentence "null or reversed" needs to become "null, reversed, or too small to account for the observed differences." The last two columns are the honest version of the claim.⟩

---

### Table S11A. Run-to-run agreement of model outputs on identical prompts, 30-vignette subset

| Model | Case–model pairs | Same top-1 diagnosis | Mean shared diagnoses (of 5) | Identical five-item lists |
|---|---|---|---|---|
| Claude Opus 4.5 | 30 | 93.3% | 3.80 | 2 (6.7%) |
| Gemini 3 Pro | 30 | 90.0% | 3.60 | 2 (6.7%) |
| GPT-5.2 | 29 | 72.4% | 3.59 | 6 (20.7%) |
| DeepSeek-V3.2 | 30 | 53.3% | 2.27 | 0 (0%) |
| **All** | **119** | **77.3%** | **3.31** | **10 (8.4%)** |

Two generation runs ≤4 days apart, identical prompts and parameters; one GPT-5.2 output withheld by a content filter in one run. Only Claude Opus 4.5 was accessed via a dated model snapshot; for the other three, run-to-run variation and provider-side model updates cannot be separated. DeepSeek-V3.2 was the only model configured at temperature 0.

### Table S11B. Effect of regeneration on correctness verdicts and accuracy, 30-vignette subset

| | Top-1 verdict changed | Top-5 verdict changed |
|---|---|---|
| Matching threshold 90 | 17 / 119 (14.3%) | 15 / 119 (12.6%) |
| Matching threshold 100 | 18 / 119 (15.1%) | 19 / 119 (16.0%) |

| Model | Top-5, run 1 | Top-5, run 2 | Shift (points) | *n* |
|---|---|---|---|---|
| GPT-5.2 | 0.793 | 0.897 | +10.4 | 29 |
| Gemini 3 Pro | 0.733 | 0.800 | +6.7 | 30 |
| Claude Opus 4.5 | 0.833 | 0.867 | +3.4 | 30 |
| DeepSeek-V3.2 | 0.900 | 0.867 | −3.3 | 30 |
| Between-model spread | 16.7 | 9.7 | | |
| Best-performing model | DeepSeek-V3.2 | GPT-5.2 | | |

Accuracy by automated adjudication at threshold 90. The GPT-5.2 shift of +10.4 points corresponds to three cases out of 29.

⟨That last sentence is the denominator the Discussion currently omits.⟩

---

## Not buildable from the record

| Table | Why |
|---|---|
| S1 (screening scores) | Per-vignette data not in the record |
| S4 (hyperparameters) | In the current supplement already; carry over |
| S8A/B (memorization) | In the current supplement already; carry over |
| S9A–C (qualitative) | In the current supplement already; carry over. S9B/C need the three Gemini safety-concern descriptions added |
| Discordance 2×2 | Have the correct-side cells (486; 18 at ≤1; 87 at ≤2) but not the incorrect-side breakdown (114 ratings). Worth a table if the pipeline gives the full 2×2 — it's the direct answer to R2-2 |
| Adjudicator validation | Not run |
| Case mix | Not computed |
