import pandas as pd, numpy as np

F = '/Users/kevin/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/2_evaluate_diagnostic_reasoning/clinician_annotated_reasoning_traces/original_google_sheet.xlsx'
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