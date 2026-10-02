# Complete analysis record — AI-26-00237

Every analysis run between the NEJM AI rejection and the pre-submission inquiry.
Self-contained: rationale, mathematics, code, results, and the errors made along the way.

**Nothing here required new data collection.** All findings come from files that
existed at the time of the original submission.

---

## Contents

| § | Analysis | Status |
|---|---|---|
| 1 | Data structure and loading | Verified |
| 2 | Discordance 2×2 (Reviewer 2's request) | Verified — result too thin to use |
| 3 | Intraclass correlation | Verified — found a transposition in the original R |
| 4 | Accuracy/reasoning dissociation (primary finding) | Verified |
| 5 | Reliability-attenuation sensitivity check | Verified |
| 6 | Adjudication reliability | Verified |
| 7 | Trace-length effect | Verified after one correction |
| 8 | Automated proxies for reasoning quality | Verified (descriptive, n=4) |
| 9 | Run-to-run output comparison | Verified |
| 10 | Rescoring through the study's adjudicator | Verified |
| 11 | Fuzzy-tier audit | Verified |
| 12 | Diagnosis-relation coding | v1 failed; v2 pending clinician |
| 13 | Statistical principles carried forward | — |
| 14 | Errors made and corrected | — |

---

## 1. Data structure and loading

### Rationale
Establish what is actually in the rating workbook before computing anything from it.

### What the data is
- `Clinical_Annotation__Task_2B_-_Judge_LLM_Reasoning__Blinded_.xlsx`
- One sheet per rater; 30 cases × 4 models = 120 traces per rater
- Two rating dimensions, 0–4: **Extraction Score**, **Diagnosis Score**
  (0 Poor / 1 Fair / 2 Adequate / 3 Good / 4 Excellent)
- Per-rating correctness call: **Diagnosis Match?** (Yes/No)
- Models: Claude Opus 4.5, GPT-5.2, Gemini 3 Pro, DeepSeek-V3.2
- Cases: 19 literature (IDs < 1000) + 11 fictitious (IDs ≥ 1000)

### Row accounting
```
765 rows read from six sheets
 -45  demographics questionnaire appended below the ratings (no Diagnostician)
= 720 rating rows  (6 raters × 120)
-120  Manu Sharma, did not begin
= 600 analyzable ratings from 5 raters
```

Raters were blinded to model identity — the `Diagnostician` column was hidden in the
live sheet. This matters throughout: it removes brand expectation as an explanation
for any model-level result.

### Code
```python
import pandas as pd, numpy as np

F = 'Clinical_Annotation__Task_2B_-_Judge_LLM_Reasoning__Blinded_.xlsx'
RATERS = ['Carolyn Rodriguez','Salih Selek','Pooja Chaudhary',
          'Caesa Nagpal','Stan Mathis','Manu Sharma']

frames = []
for rater in RATERS:
    d = pd.read_excel(F, sheet_name=rater)
    d = d.loc[:, ~d.columns.astype(str).str.startswith('Unnamed')]
    d['rater'] = rater
    frames.append(d)
d = pd.concat(frames, ignore_index=True)

d = d[d['Diagnostician'].notna()].copy()        # drop questionnaire rows

def sc(x):                                       # "3 - Good" -> 3
    return np.nan if pd.isna(x) else int(str(x).split(' - ')[0])

d['ext']     = d['Extraction Score (0-4)'].apply(sc)
d['dx']      = d['Diagnosis Score (0-4)'].apply(sc)
d['correct'] = d['Diagnosis Match?'].map({'Yes': 1, 'No': 0})

r = d[d['dx'].notna() & d['correct'].notna()].copy()   # 600 rows
r.to_pickle('r.pkl')
```

---

## 2. Discordance 2×2 — Reviewer 2's request

### Rationale
Reviewer 2 argued that an association between reasoning quality and correctness is
close to expected by construction, and asked for the reverse direction: how often does
a model reach a **correct** answer with **poor** reasoning? Those cases are what an
accuracy benchmark scores as successes while a clinician would not.

### Mathematics
Conditional probability from a 2×2 contingency table. With reasoning dichotomized at
threshold *t*:

```
P(correct | low reasoning) = n(correct ∧ dx ≤ t) / n(dx ≤ t)
P(low reasoning | correct) = n(correct ∧ dx ≤ t) / n(correct)
```

Computed at both rating level (n = 600) and trace level (n = 120, averaging the five
raters), because the second is more conservative — a single harsh rater cannot create
a discordant cell.

### Code
```python
r = pd.read_pickle('r.pkl')
corr = r[r.correct == 1]

for thr in (1, 2):
    n = (corr.dx <= thr).sum()
    print(f"correct with reasoning <= {thr}: {n}/{len(corr)} = {n/len(corr):.1%}")

tr = (r.groupby(['Case ID','Diagnostician'])
        .agg(dx=('dx','mean'), prop_correct=('correct','mean'))
        .reset_index())
tr['majority_correct'] = (tr.prop_correct > 0.5).astype(int)
for thr in (1.5, 2.0):
    cell = ((tr.dx <= thr) & (tr.majority_correct == 1)).sum()
    print(f"trace-level, mean dx <= {thr}: {cell} traces in the discordant cell")
```

### Results
| Threshold | Correct ratings with poor reasoning |
|---|---|
| dx ≤ 1 (Poor/Fair) | 18 / 486 = **3.7%** |
| dx ≤ 2 (Adequate or worse) | 87 / 486 = **17.9%** |

Trace level: **0 traces** at mean dx ≤ 1.5; 6 traces at ≤ 2.0.

### Interpretation
The analysis runs, but the cell is nearly empty at any strict threshold. *Nearly one
in five correct diagnoses was reached with reasoning rated no better than adequate*
is defensible; a reframed manuscript cannot rest on it. Reported honestly rather than
dropped — publishing the weak cell is what separates this from analysis shopping.

---

## 3. Intraclass correlation

### Rationale
Verify the manuscript's reported ICC of 0.84 and establish reliability separately for
each rating dimension.

### Mathematics
Two-way random effects, absolute agreement — matching `irr::icc(model="twoway",
type="agreement", unit="average")`. For *n* subjects × *k* raters:

```
SSR = k · Σᵢ (x̄ᵢ· − x̄)²          rows (subjects),  df = n−1
SSC = n · Σⱼ (x̄·ⱼ − x̄)²          columns (raters), df = k−1
SSE = SST − SSR − SSC             df = (n−1)(k−1)

MSR = SSR/(n−1)   MSC = SSC/(k−1)   MSE = SSE/((n−1)(k−1))

                        MSR − MSE
ICC(A,1) = ───────────────────────────────────────
           MSR + (k−1)·MSE + (k/n)·(MSC − MSE)

                    MSR − MSE
ICC(A,k) = ─────────────────────────
           MSR + (MSC − MSE)/n
```

ICC(A,1) is the reliability of a single rater; ICC(A,k) that of the mean of *k* raters.
Report both — the difference is what justifies using a panel.

### Code
```python
def icc_twoway(p):
    """p: rows = subjects, cols = raters. Returns ICC(A,1), ICC(A,k)."""
    X = p.values.astype(float); n, k = X.shape
    gm = X.mean(); rm = X.mean(axis=1); cm = X.mean(axis=0)
    SSR = k * ((rm - gm) ** 2).sum()
    SSC = n * ((cm - gm) ** 2).sum()
    SSE = ((X - gm) ** 2).sum() - SSR - SSC
    MSR, MSC, MSE = SSR/(n-1), SSC/(k-1), SSE/((n-1)*(k-1))
    a1 = (MSR - MSE) / (MSR + (k-1)*MSE + (k/n)*(MSC - MSE))
    ak = (MSR - MSE) / (MSR + (MSC - MSE)/n)
    return a1, ak

for col, name in [('dx','Diagnostic reasoning'), ('ext','Data extraction')]:
    p = r.pivot_table(index=['Case ID','Diagnostician'], columns='rater',
                      values=col).dropna()
    print(name, icc_twoway(p))
```

R equivalent:
```r
library(irr); library(tidyr); library(dplyr)
wide <- ratings %>%
  select(case_id, model, rater, diagnosis_score) %>%
  pivot_wider(names_from = rater, values_from = diagnosis_score) %>%
  select(-case_id, -model) %>% as.data.frame()
icc(wide, model = "twoway", type = "agreement", unit = "average")
icc(wide, model = "twoway", type = "agreement", unit = "single")
```

### Results
| Dimension | ICC(A,1) | ICC(A,k), k=5 |
|---|---|---|
| Data extraction | 0.512 | **0.840** |
| Diagnostic reasoning | 0.405 | **0.773** |
| Mean of both | 0.496 | 0.831 |

### Interpretation
The manuscript's 0.84 is the **extraction** dimension. The reasoning dimension — the
one carrying the headline finding — is 0.77. Cause: transposed column assignment in
the original R script. Audit confirmed the error was confined to the ICC calculation.
Both values are acceptable; the manuscript must label which is which, and report
ICC(A,1) alongside, since single-rater reliability is only 0.41–0.51.

---

## 4. Accuracy/reasoning dissociation — the primary finding

### Rationale
Reviewer 2's objection was that reasoning quality correlating with correctness is
uninformative. The stronger test is whether the two measures ever **come apart**: can
clinician-rated reasoning distinguish models that accuracy cannot?

### Mathematics
The unit of analysis is the **case**, not the rating. Each case yields one score per
model (mean of five raters), so models are compared **paired within case**, which
removes case difficulty as a source of variance.

For models A and B across *n* = 30 cases, with dᵢ = x̄ᵢᴬ − x̄ᵢᴮ:

```
d̄ = (1/n) Σ dᵢ          SE = s_d / √n          t = d̄ / SE,  df = n−1
95% CI = d̄ ± t₀.₉₇₅,ₙ₋₁ · SE
```

Six pairwise comparisons, so **Holm–Bonferroni** correction. Sort p-values ascending
p₍₁₎ ≤ … ≤ p₍₆₎; the adjusted value is

```
p̃₍ᵢ₎ = max_{j ≤ i} [ (m − j + 1) · p₍ⱼ₎ ]   capped at 1
```

Holm is uniformly more powerful than Bonferroni and controls family-wise error.

Cross-checked with a crossed random-effects model on rating-level data:
```
dx_ijk = μ + β_model + u_case(i) + v_rater(j) + ε_ijk
u ~ N(0, σ²_case),  v ~ N(0, σ²_rater),  ε ~ N(0, σ²)
```

### Code
```python
import itertools
from scipy import stats

tr = (r.groupby(['Case ID','Diagnostician'])
        .agg(dx=('dx','mean'), acc=('correct','mean')).reset_index())
w_dx  = tr.pivot(index='Case ID', columns='Diagnostician', values='dx')
w_acc = tr.pivot(index='Case ID', columns='Diagnostician', values='acc')

def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj

for w, label in [(w_dx, 'reasoning'), (w_acc, 'accuracy')]:
    out = []
    for a, b in itertools.combinations(w.columns, 2):
        d = (w[a] - w[b]).dropna()
        t, p = stats.ttest_rel(w[a].dropna(), w[b].dropna())
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = stats.t.interval(0.95, len(d)-1, loc=d.mean(), scale=se)
        out.append((f"{a} vs {b}", d.mean(), lo, hi, p))
    ps = holm([o[4] for o in out])
    for o, ph in zip(out, ps):
        print(f"{label:9s} {o[0]:50s} {o[1]:+.2f} [{o[2]:+.2f},{o[3]:+.2f}] p_holm={ph:.4f}")
```

R equivalent:
```r
library(lme4); library(lmerTest); library(emmeans)
m <- lmer(diagnosis_score ~ model + (1|case_id) + (1|rater), data = ratings)
pairs(emmeans(m, ~ model), adjust = "holm")
confint(pairs(emmeans(m, ~ model), adjust = "holm"))
```

### Results

Per model (5 raters, 30 cases):

| Model | Accuracy | Reasoning (0–4) | Extraction (0–4) |
|---|---|---|---|
| Claude Opus 4.5 | 0.827 | 3.60 | 3.67 |
| DeepSeek-V3.2 | 0.827 | 2.90 | 2.97 |
| Gemini 3 Pro | 0.820 | 3.12 | 3.15 |
| GPT-5.2 | 0.767 | 2.37 | 2.21 |

Pairwise, Holm-corrected:

| Comparison | Δ reasoning [95% CI] | Holm p | Δ accuracy [95% CI] |
|---|---|---|---|
| Claude vs GPT-5.2 | +1.23 [1.04, 1.43] | <0.001 | +0.060 [−0.079, +0.199] |
| Gemini vs GPT-5.2 | +0.75 [0.52, 0.99] | <0.001 | +0.053 [−0.052, +0.159] |
| Claude vs DeepSeek | +0.70 [0.50, 0.90] | <0.001 | −0.000 [−0.071, +0.071] |
| DeepSeek vs GPT-5.2 | +0.53 [0.27, 0.80] | 0.0006 | +0.060 [−0.090, +0.210] |
| Claude vs Gemini | +0.48 [0.29, 0.67] | 0.0001 | +0.007 [−0.090, +0.104] |
| DeepSeek vs Gemini | −0.22 [−0.52, +0.08] | 0.138 | +0.007 [−0.117, +0.130] |

**Accuracy separated 0 of 6 pairs. Reasoning separated 5 of 6.**

Mixed model cross-check: identical point estimates (−0.70, −0.48, −1.23 vs Claude).
Variance components: case 0.129, rater **0.002**, residual 0.625. Rater variance is
near zero, which is why averaging over raters loses almost nothing and the
paired-by-case approach is defensible.

All five raters produced the **identical rank ordering**, blinded to model identity.

### Why this resists the halo critique
The editors noted that raters adjudicated correctness before rating reasoning. Two
structural defences:

1. **Equal accuracy.** Claude and DeepSeek are at 0.827 each — the correctness signal
   available to a rater is identical, so a halo cannot manufacture a 0.70-point gap.
   More generally a halo pulls equal-accuracy models toward equal ratings, biasing
   *against* this finding.
2. **Identity blinding.** No rater knew which system produced any trace, so brand
   expectation is unavailable as an explanation by design.

---

## 5. Reliability-attenuation sensitivity check

### Rationale
A reviewer can argue the dissociation is an artifact: correctness is measured less
reliably (κ = 0.62) than reasoning (ICC 0.77–0.84), and a noisier measure has less
power to detect true differences. On that account accuracy fails to separate the
models because it was measured badly, not because they don't differ.

### Method
Restrict to cases where **all five raters agreed unanimously on all four models** —
the cleanest possible adjudication — and re-run the paired comparisons.

### Code
```python
g = r.groupby(['Case ID','Diagnostician'])['correct'].agg(['sum','size']).reset_index()
g['unanimous'] = (g['sum'] == 0) | (g['sum'] == g['size'])
g['acc'] = (g['sum'] == g['size']).astype(int)

clean = g.groupby('Case ID')['unanimous'].all()
clean = clean[clean].index                      # 18 of 30 cases
w = g[g['Case ID'].isin(clean)].pivot(index='Case ID',
                                      columns='Diagnostician', values='acc')
for a, b in itertools.combinations(w.columns, 2):
    if (w[a]-w[b]).std() == 0:
        print(a, b, "no variance"); continue
    print(a, b, stats.ttest_rel(w[a], w[b]))
```

### Results
On the 18 cleanly adjudicated cases: Claude 1.000, DeepSeek 1.000, Gemini 0.944,
GPT-5.2 0.889. No pairwise comparison approaches significance.

### Interpretation
The accuracy null is not produced by adjudication noise. Small subset — report as a
sensitivity analysis rather than lean on it — but it closes the objection.

---

## 6. Adjudication reliability

### Rationale
Correctness is treated across this literature as the objective anchor and reasoning
assessment as the subjective addition. Test whether that is true in our data.

### Mathematics
Cohen's κ for each rater pair, then averaged:

```
        p_o − p_e
κ = ─────────────
         1 − p_e
```
where p_o is observed agreement and, for binary judgments with marginal rates p_A, p_B:
```
p_e = p_A·p_B + (1−p_A)(1−p_B)
```

Unanimity rate = proportion of traces on which all five raters gave the same verdict.

### Code
```python
g = r.groupby(['Case ID','Diagnostician'])['correct'].agg(['sum','size'])
g['unanimous'] = (g['sum'] == 0) | (g['sum'] == g['size'])
print("split rate:", (~g.unanimous).mean())

piv = r.pivot_table(index=['Case ID','Diagnostician'], columns='rater', values='correct')
ks = []
for a, b in itertools.combinations(piv.columns, 2):
    m = piv[[a, b]].dropna()
    po = (m[a] == m[b]).mean()
    pa, pb = m[a].mean(), m[b].mean()
    pe = pa*pb + (1-pa)*(1-pb)
    ks.append((po - pe) / (1 - pe))
print("mean pairwise kappa:", np.nanmean(ks))
```

### Results
- Correctness calls **not unanimous for 23.3%** of traces (28/120)
- Mean pairwise Cohen's **κ = 0.626** (range 0.46–0.83)
- Reasoning quality over the same traces: **ICC(A,k) = 0.77 / 0.84**

### Interpretation
Clinicians agreed about reasoning quality at least as reliably as about whether the
answer was correct. The supposedly objective measure is the less reproducible one.

---

## 7. Trace-length effect

### Rationale
The obvious deflation of the dissociation finding: raters rewarded longer traces, and
the better-rated models simply wrote more.

### Mathematics — and a correction

**What was done first (wrong).** Centre length and score within each model, then run
an ordinary Pearson correlation on the residuals:
```
r = +0.295, p = 0.001, n = 120
```
This treats 120 observations as independent. They are 30 cases × 4 models; each case
appears four times, so the p-value is far too small. "Pooled r" is not a standard
statistic — it was an ad hoc construction.

**What is correct.** A mixed model with a random intercept for case:
```
dx_im = β₀ + β₁·(words_im / 100) + β_model + u_case(i) + ε_im
```
β₁ is the within-case, within-model association between length and rating.

### Code
```python
import statsmodels.formula.api as smf
j['w100'] = j['words_run2'] / 100
fit = smf.mixedlm("dx ~ w100 + C(model)", data=j, groups=j['case_id']).fit(reml=True)
b, se = fit.params['w100'], fit.bse['w100']
print(f"{b:+.4f} per 100 words, 95% CI [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}]")
```

### Results
**+0.041 points per 100 words** (95% CI 0.021 to 0.060).
Within-case decomposition across the four models gives the same answer (slope +0.039).

### Interpretation
Length matters, but cannot account for the finding. The between-model gaps are 0.48
to 1.23 points; explaining them by length alone would require 1,200 to 3,000 extra
words. The observed Claude–GPT-5.2 difference is **188 words**, predicting +0.08
points against an observed +1.23.

---

## 8. Automated proxies for reasoning quality

### Rationale
If a computable feature of a trace tracked clinician judgment, the clinician panel
would be unnecessary and the paper's method would be hard to justify.

### Mathematics
Spearman rank correlation across the four models:
```
ρ = 1 − 6·Σdᵢ² / (n(n²−1)),  n = 4
```
With n = 4 this is descriptive only — it cannot support an inferential claim.

### Provenance of the inputs
```
raw model output JSON  (model_thoughts field, run 2 = the traces clinicians rated)
        │
        ├─ run_comparison.py :: compare_traces()   ← computes per-trace features
        │
        ▼
trace_comparison_by_case.csv  (120 rows: one per case × model)
        │
        ├─ groupby('model').mean()
        │
        ▼                            ┌── r.pkl (rating workbook) ──┐
per-model feature means  ────────────┴── groupby('model')['dx'] ───┴──► Spearman
```

### Feature definitions
```python
import re

ICD_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b")

SELF_CORRECTION = ["wait", "hmm", "actually", "let me reconsider",
                   "on second thought", "but then", "however", "although",
                   "revisit", "double-check", "double check", "rule out",
                   "alternatively", "need to check"]          # 14 phrases, hand-built

def trace_features(trace_text):
    t = (trace_text or "")
    return {
        "words":      len(t.split()),
        "icd_codes":  len(set(ICD_RE.findall(t))),            # distinct codes named
        "self_corr":  sum(t.lower().count(m) for m in SELF_CORRECTION),
    }
```

### Code
```python
import pandas as pd
from scipy import stats

# Trace features: aggregate the per-case CSV that run_comparison.py wrote
t = pd.read_csv('trace_comparison_by_case.csv')
feat = t.groupby('model')[['words_run2', 'codes_in_trace_run2', 'delib_run2']].mean()

# Clinician ratings: aggregate the rating workbook
r = pd.read_pickle('r.pkl')
name_map = {'Anthropic Claude Opus 4.5': 'Claude Opus 4.5',
            'OpenAI GPT-5.2': 'GPT-5.2',
            'Google Gemini 3 Pro': 'Gemini 3 Pro',
            'DeepSeek-V3.2': 'DeepSeek-V3.2'}
r['model'] = r['Diagnostician'].map(name_map)
clin = r.groupby('model')['dx'].mean().rename('clin_dx')

j = pd.concat([clin, feat], axis=1).sort_values('clin_dx', ascending=False)

for c in ['words_run2', 'codes_in_trace_run2', 'delib_run2']:
    rho, p = stats.spearmanr(j['clin_dx'], j[c])
    print(f"rho(clin_dx, {c}) = {rho:+.2f}  p={p:.3f}  n={len(j)}")
```

### Derived values
| Model | Clinician rating | Trace words | ICD codes named | Self-correction phrases |
|---|---|---|---|---|
| Claude Opus 4.5 | 3.60 | 1087.8 | 2.83 | 0.73 |
| Gemini 3 Pro | 3.12 | 431.0 | 3.97 | 1.40 |
| DeepSeek-V3.2 | 2.90 | 619.6 | 3.50 | 3.77 |
| GPT-5.2 | 2.37 | 900.2 | 8.77 | 3.07 |

All figures use run-2 traces — the generation the clinicians actually rated.

### Results
| Feature | Spearman ρ vs clinician rating | p |
|---|---|---|
| ICD codes named in trace | **−0.80** | 0.200 |
| Self-correction phrases | **−0.80** | 0.200 |
| Trace length | +0.20 | 0.800 |

None reaches significance, and none can: with n = 4 the smallest attainable two-sided
p for Spearman is 0.083. These are descriptive rank orderings, and the direction is
the finding — not the coefficient.

The lowest-rated model named more than three times as many candidate diagnoses as the
highest-rated model (8.8 vs 2.8) and used more than four times as many self-correction
phrases.

### Caveat
The self-correction measure is the hand-built 14-phrase list shown above — not a
validated instrument, and the phrase set was chosen before seeing how it correlated,
but not preregistered. Report as exploratory. The ICD-code count is objectively
countable from a regex and is the stronger of the two measures.

A note on how these numbers were first obtained: the per-model means were originally
transcribed from printed terminal output rather than recomputed from
`trace_comparison_by_case.csv`. They were subsequently verified against the CSV and
match to rounding, but the derivation code above is what should be re-run for the
manuscript.

---

## 9. Run-to-run output comparison

### Rationale
The clinician-rated subset was generated in a **separate model run** from the
196-case accuracy analysis, with identical prompts, parameters and aliases, ≤4 days
apart. This is an unplanned test–retest of the models themselves.

### Mathematics
Per case–model pair, comparing the two runs' ranked five-item differentials:

```
top-1 agreement  = 1 if first ICD code identical
Jaccard(A,B)     = |A ∩ B| / |A ∪ B|        on the sets of five codes
positional match = # ranks at which the codes coincide
```

### Code
```python
import re
CODE_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b")

def parse_differential(raw):
    out = []
    for line in str(raw).split("\n"):
        m = re.match(r"^\s*(\d+)\s*[\.\)\:]\s*(.+?)\s*$", line)
        if not m: continue
        body = m.group(2)
        code = CODE_RE.search(body)
        out.append({'rank': int(m.group(1)),
                    'code': code.group(1) if code else None})
    return sorted(out, key=lambda d: d['rank'])

def compare(a, b):
    ca = [d['code'] for d in a if d['code']]
    cb = [d['code'] for d in b if d['code']]
    return {
        'top1':      ca[0] == cb[0] if ca and cb else None,
        'jaccard':   len(set(ca) & set(cb)) / len(set(ca) | set(cb)),
        'shared':    len(set(ca) & set(cb)),
        'positional': sum(1 for i in range(min(len(ca), len(cb))) if ca[i] == cb[i]),
    }
```

### Results
| Model | Top-1 same dx | Mean shared dx (of 5) | Identical lists |
|---|---|---|---|
| Claude Opus 4.5 | 93.3% | 3.80 | 2/30 |
| Gemini 3 Pro | 90.0% | 3.60 | 2/30 |
| GPT-5.2 | 72.4% | 3.59 | 6/29 |
| **DeepSeek-V3.2** | **53.3%** | **2.27** | **0/30** |
| Overall | 77.3% | 3.31 | 10/119 (8.4%) |

DeepSeek shared a mean of **0.3** ICD codes *within* its two reasoning traces for the
same case — the diagnoses it considered were nearly disjoint across identical queries.
GPT-5.2 shared 5.0.

### Interpretation
DeepSeek was the only model set to `temperature = 0`, and was by far the least
reproducible. DeepSeek's documentation confirms the parameter is ignored in thinking
mode. This supports Reviewer 1's comment that temperature 0 does not produce
deterministic output — demonstrated from the study's own data.

Drift is excluded: gaps ≤4 days, and Claude and GPT-5.2 used pinned dated snapshots.

---

## 10. Rescoring through the study's own adjudicator

### Rationale
§9 measured *diagnosis* agreement. What matters for the manuscript is *correctness*
agreement, in the study's own currency: rapidfuzz `token_set_ratio ≥ 90`, then
gpt-5-mini with the original adjudication prompt.

Two passes: threshold 90 (the study's setting), and threshold 100 as a sensitivity
check on the fuzzy tier.

### Mathematics
Metrics as defined in `evaluate_accuracy.py`:
```
top-1     = 1 if the rank-1 prediction matches any reference diagnosis
hit_rate  = 1 if any prediction matches any reference          ("top-5")
recall@5  = |matched references| / |references|
MRR       = 1 / rank of first matching prediction
flip      = 1 if the metric differs between run 1 and run 2
```

### Results — correctness flips between generations
| | Top-1 flipped | Top-5 flipped |
|---|---|---|
| Threshold 90 | 17/119 = **14.3%** | 15/119 = **12.6%** |
| Threshold 100 | 18/119 = **15.1%** | 19/119 = **16.0%** |

(119 not 120: GPT-5.2 case 94 run 1 was withheld by a provider content filter.)

### Results — effect on the accuracy table
Top-5 accuracy, threshold 90:

| Model | Run 1 | Run 2 | Shift |
|---|---|---|---|
| GPT-5.2 | 0.793 | 0.897 | **+10.4** |
| Gemini 3 Pro | 0.733 | 0.800 | +6.7 |
| Claude Opus 4.5 | 0.833 | 0.867 | +3.4 |
| DeepSeek-V3.2 | 0.900 | 0.867 | −3.3 |

**The manuscript's between-model spread is 7.1 points (0.730–0.801). The largest
shift between generations of the same model is 10.4 points.**

Ranking:
- Run 1: DeepSeek 0.900 > Claude 0.833 > GPT-5.2 0.793 > Gemini 0.733
- Run 2: GPT-5.2 0.897 > Claude 0.867 > DeepSeek 0.867 > Gemini 0.800

The best-performing model depends on which generation was scored.

### Results — the adjudicator is itself non-deterministic

**The proof is logical, not statistical.** Raising the threshold from 90 to 100 sends
pairs scoring 90–99 to the LLM instead of auto-matching them as correct. The LLM can
only agree or disagree, so

```
{matches at threshold 100} ⊆ {matches at threshold 90}
⟹ accuracy(100) ≤ accuracy(90)
```

| Model / run | Thr 90 | Thr 100 | Change | |
|---|---|---|---|---|
| Claude run 1 | 0.833 | 0.867 | **+3.4** | violates the constraint |
| GPT-5.2 run 1 | 0.793 | 0.828 | **+3.5** | violates the constraint |
| Claude run 2 | 0.867 | 0.800 | −6.7 | |
| DeepSeek run 1 | 0.900 | 0.833 | −6.7 | |
| DeepSeek run 2 | 0.867 | 0.733 | −13.4 | |
| GPT-5.2 run 2 | 0.897 | 0.828 | −6.9 | |
| Gemini both | — | — | 0.0 | |

Two estimates rose. The only available explanation is that gpt-5-mini returned
different verdicts on the second pass. Neither `evaluate_accuracy.py` nor the rescore
script sets a sampling temperature for the adjudicator.

Mean absolute change across the eight cells: **5.1 points**; maximum 13.4.

### Consequences
1. Every accuracy figure in the manuscript would move if the adjudication were re-run
   over identical model outputs.
2. The fuzzy-tier contribution **cannot be isolated** from these two passes, because
   adjudicator variance is of comparable magnitude. Isolating it requires 3–5
   adjudication passes per threshold, reporting mean and spread.
3. 9–10 LLM calls failed per pass and were scored as non-matches — the behaviour of
   the original pipeline. These should be retried; the bias is downward.

---

## 11. Fuzzy-tier audit

### Rationale
`token_set_ratio ≥ 90` auto-accepts a match without consulting the LLM. Establish how
much of the reported accuracy passes through that shortcut, and whether it accepts
clinically decisive distinctions.

### Mathematics
`token_set_ratio` scores on the **intersection** of token sets, so a short
discriminating token barely moves the score:
```
"Bipolar I disorder"  vs  "Bipolar II Disorder — F31.81"   → 94.1   (auto-matched)
"Major depressive disorder" vs "MDD, recurrent, severe with psychotic features" → 78 (escalated)
```
The gate is backwards: it auto-passes short decisive differences and escalates long
benign ones.

### Code
```python
from rapidfuzz import fuzz
score = fuzz.token_set_ratio(reference.lower(), prediction.lower())
auto_matched = score >= 90        # never reaches the LLM
```

### Results
- 1,950 total reference × prediction comparisons; **926 unique** after caching
- **22 auto-matched** by the fuzzy tier = **2.4%** of unique comparisons
- LLM calls: 904 at threshold 90, 926 at threshold 100

Of the 22, two are clinically wrong:
```
97.0  Mild neurocognitive disorder due to TBI == Major Neurocognitive Disorder Due to TBI
```
Three accepted MDD moderate against MDD severe (defensible as the same diagnosis per
clinical review); one accepted "in remission" against "in sustained remission".

**No Bipolar I/II pair actually occurs in the data** — that was a property of the
matcher, not an error that was made. The letter should say *can auto-accept*.

---

## 12. Diagnosis-relation coding

### Rationale
Characterize *where* clinicians disagree about correctness. If disagreement
concentrates in a describable class of comparison, the reference standard's ambiguity
becomes a measurement rather than an assertion.

### v1 — hypothesis failed
**Hypothesis:** disagreement tracks verbatim vs non-verbatim matching.

| | non-exact | total | % |
|---|---|---|---|
| Disputed traces | 14 | 28 | 50.0% |
| Unanimous traces | 40 | 83 | 48.2% |

Fisher exact OR = 0.93, p = 1. **No association.** The hypothesis was wrong.

**Post-hoc regrouping** by relation type revealed a gradient:

| Relation | n | % disputed |
|---|---|---|
| ALTERNATE | 11 | 81.8% |
| PARTIAL | 6 | 66.7% |
| ETIOLOGY | 7 | 42.9% |
| WRONG | 10 | 40.0% |
| SPECIFIER | 44 | 15.9% |
| EXACT | 42 | 2.4% |

Regrouped: same disorder (EXACT+SPECIFIER) 8/86 = **9.3%** disputed;
different disorder 20/34 = **58.8%**.

The v1 grouping had failed because `SPECIFIER` (n=44, 16% disputed) and `ALTERNATE`
(n=11, 82%) — opposite behaviours — were in the same bucket.

**Two disqualifying problems:**
1. The regrouping was chosen after the first failed. Post-hoc.
2. Coding was done by a non-clinician. The `SPECIFIER`/`ALTERNATIVE` boundary is
   load-bearing and requires DSM knowledge.

### v2 — organized around clinical management
Clinical review (Manu) identified that `SPECIFIER` conflates distinctions of very
different weight:
- MDD severe **without** psychotic features vs MDD moderate → same management
- MDD severe **with** psychotic features vs **without** → different management
- Primary psychosis vs substance-induced → different management

The data confirms the concern. Against reference *"Bipolar disorder, current episode
manic (F31.2)"*, raters unanimously accepted **both** a model answer of "with
psychotic features" **and** one of "without psychotic features" — mutually exclusive
specifiers, both scored correct. (F31.2 denotes the manic episode *with* psychotic
symptoms, so the information was present in the code but not used.)

**v2 coding, five judgments per trace:**

| | Judgment |
|---|---|
| C1 | Which reference diagnosis is the model answering? (1, 2, 3… or 0) |
| C2 | Relation: EXACT / SYNONYM / SEVERITY / SPECIFIER_CAT / SUBTYPE / ETIOLOGY / DIFFERENT_DX / NO_MATCH |
| C3 | **Does the difference change clinical management?** SAME / DIVERGENT / UNCERTAIN |
| C4 | If raters disagreed here, what kind? CLINICAL / RUBRIC / NEITHER |
| C5 | Coder confidence: HIGH / MEDIUM / LOW |

**C3 is the analysis variable.** It is clinically motivated and specified in advance,
which fixes the post-hoc problem. It must be pre-registered for the replication study.

### Analysis code
```python
df = pd.read_csv('coding_sheet_v2.csv').merge(pd.read_csv('key.csv'), on='row_key')
df['split'] = df.status == 'SPLIT'

print(pd.crosstab(df.C3_management, df.status))
for cat in df.C3_management.unique():
    s = df[df.C3_management == cat]
    print(f"{cat}: {s.split.sum()}/{len(s)} disputed = {100*s.split.mean():.1f}%")

# Inference must cluster on case:
#   glmer(split ~ C3_management + (1|case_id), family = binomial)
```

### Status
Pending clinician coding. Report the descriptive contrast, not a p-value, and label
exploratory until the replication tests it prospectively.

---

## 13. Statistical principles carried forward

**1. Cluster by case.** The 120 traces are 30 cases × 4 models. Any statistic computed
across them needs `(1|case_id)`. Two estimates in this record were corrected for
exactly this — a bootstrap that found 6/6 pairs separated (correct answer: 5/6) and
the trace-length correlation.

**2. Report both ICC forms.** Single-rater agreement is 0.41–0.51; the reported
0.77–0.84 comes from averaging five raters. Stating both also motivates why
panel-based reasoning assessment is a methodological contribution rather than an
inconvenience.

**3. Non-separation is not equivalence.** At n = 30 the accuracy CIs are roughly
±0.09. *Accuracy did not separate the models at this sample size* is defensible;
*the models are equivalent in accuracy* is not.

**4. Label post-hoc analyses as post-hoc.** The same/different-disorder grouping and
the entire run-to-run comparison were unplanned.

**5. Claim what was measured.** Provider-supplied reasoning artifacts rated by
clinicians — not the models' internal reasoning. Write *"clinicians rated model X's
reasoning output lowest"*, not *"model X reasons less well."*

**6. Descriptive contrasts beat p-values on small n.** With four models or 28 disputed
traces, report the numbers and let the reader judge.

---

## 14. Errors made and corrected

| Error | Detected by | Correction |
|---|---|---|
| ICC columns transposed in the original R script | Recomputation from raw ratings | 0.84 is extraction, 0.77 is reasoning. Audit confirmed no other analysis affected. |
| Bootstrap treated 600 ratings as independent → 6/6 pairs separated | Clustering review | Paired-by-case + Holm → **5/6**. Reported in the letter. |
| "Pooled r = +0.295, p = 0.001" — an ad hoc statistic ignoring repeated cases | Reviewer question | Mixed model → **+0.041 per 100 words** [0.021, 0.060] |
| Claimed the lowest-rated model wrote the longest traces and hedged most | Data check | False. Claude wrote the longest (1,088 words) and ranked first. Corrected to candidate-diagnosis and self-correction counts. |
| §8 feature means transcribed from terminal output rather than recomputed, and presented as if the hardcoded dataframe were the derivation | Co-author question | Values verified correct against `trace_comparison_by_case.csv`, but §8 now shows the actual derivation chain from raw JSON onward. |
| Claimed post-hoc elicitation differed between generation runs | Full-sample detector | 1 of 120 traces, not systematic. Withdrawn. |
| Claimed the content filter fires non-deterministically | Code review | Unsupported — run 2's function has no filter branch, and the retry loop would have masked filter events. Withdrawn. |
| Verbatim-match hypothesis for adjudication disagreement | Blinded coding | Failed (50.0% vs 48.2%). Replaced, then the replacement failed clinical review. |
| Built a clinical coding task for a non-clinician | Author correction | Re-assigned to a psychiatrist co-author; v1 retained only for inter-coder comparison. |

**Pattern worth noting.** Every one of these was caught by returning to raw data or
raw code rather than trusting a derived output. The ICC transposition in particular
would have reached print had the paper been accepted in July.
