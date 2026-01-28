#!/usr/bin/env python3
"""
Fill missing/NULL ground-truth diagnosis fields in a model-predictions JSON.

- Reads ground truth from: fictitious_only.json
- Reads predictions from: predicted_diagnoses_..._fictitious_only_....json
- For each item in predictions, if diagnosis fields are missing/null/blank,
  it copies them from the ground-truth item with the same case_id.

Writes: predicted_diagnoses_..._fictitious_only_....json.filled.json

Usage (from the directory containing the files):
  python fill_missing_diagnoses.py \
    --ground_truth /mnt/data/fictitious_only.json \
    --predictions /mnt/data/predicted_diagnoses_claude-opus-4-5-20251101_fictitious_only_20260126_201438.json \
    --out /mnt/data/predicted_diagnoses_claude-opus-4-5-20251101_fictitious_only_20260126_201438.filled.json
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple


DIAG_FIELDS = [
    "diagnosis",
    "diagnosis_dsm",
    "diagnosis_icd",
    # Add more ground-truth fields here if you want to backfill them too:
    # "difficulty",
    # "source",
]


def is_missing(val: Any) -> bool:
    """True if value should be treated as missing for our fill purposes."""
    if val is None:
        return True
    if isinstance(val, float):
        # NaN check without importing numpy
        return val != val
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data)}")
    return data


def build_truth_map(
    truth_items: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Map case_id -> dict of fields we can copy.
    case_id normalized to string for safe matching (handles int vs str).
    """
    m: Dict[str, Dict[str, Any]] = {}
    for it in truth_items:
        if "case_id" not in it:
            continue
        cid = str(it["case_id"])
        m[cid] = it
    return m


def fill_predictions(
    preds: List[Dict[str, Any]],
    truth_map: Dict[str, Dict[str, Any]],
    fields: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "total_pred_items": 0,
        "matched_case_id": 0,
        "unmatched_case_id": 0,
        "items_filled_any": 0,
        "fields_filled_total": 0,
    }
    per_field_filled = {f: 0 for f in fields}

    out: List[Dict[str, Any]] = []
    for it in preds:
        stats["total_pred_items"] += 1
        cid_val = it.get("case_id", None)
        cid = None if cid_val is None else str(cid_val)

        if cid is None or cid not in truth_map:
            stats["unmatched_case_id"] += 1
            out.append(it)
            continue

        stats["matched_case_id"] += 1
        truth = truth_map[cid]
        filled_this_item = False

        it2 = dict(it)  # shallow copy
        for f in fields:
            if is_missing(it2.get(f, None)) and not is_missing(truth.get(f, None)):
                it2[f] = truth.get(f)
                filled_this_item = True
                stats["fields_filled_total"] += 1
                per_field_filled[f] += 1

        if filled_this_item:
            stats["items_filled_any"] += 1

        out.append(it2)

    # Merge per-field counts into stats for convenient printing
    for f, n in per_field_filled.items():
        stats[f"filled_{f}"] = n

    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth", required=True, help="Path to fictitious_only.json")
    ap.add_argument("--predictions", required=True, help="Path to predicted_diagnoses_...json")
    ap.add_argument("--out", required=True, help="Output path for filled JSON")
    ap.add_argument(
        "--fields",
        nargs="*",
        default=DIAG_FIELDS,
        help=f"Fields to backfill (default: {DIAG_FIELDS})",
    )
    args = ap.parse_args()

    truth_items = load_json(args.ground_truth)
    pred_items = load_json(args.predictions)

    truth_map = build_truth_map(truth_items)
    filled, stats = fill_predictions(pred_items, truth_map, args.fields)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(filled, f, ensure_ascii=False, indent=2)

    print("Wrote:", args.out)
    print("---- Stats ----")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")


if __name__ == "__main__":
    main()