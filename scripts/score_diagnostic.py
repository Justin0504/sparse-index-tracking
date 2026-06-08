#!/usr/bin/env python3
"""Score blind diagnostic predictions against experimental results.

Loads blind_predictions.json and all experiment result files,
matches predictions to outcomes, computes accuracy and confusion matrix.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_predictions() -> dict:
    path = Path("results/diagnostic/blind_predictions.json")
    if not path.exists():
        raise FileNotFoundError(f"Blind predictions not found at {path}")
    with open(path) as f:
        return json.load(f)


def load_all_results() -> dict:
    """Load all experiment results and index by (market, K, model_type)."""
    results = {}
    result_dirs = [
        ("results/sp1500", "SP1500"),
        ("results/multi_market/nikkei225", "Nikkei225"),
        ("results/multi_market/eurostoxx50", "EuroStoxx50"),
        ("results/multi_market/asx200", "ASX200"),
        ("results/multi_market/hangseng", "HangSeng"),
        ("results/weather", "Weather"),
        ("results/epa", "EPA"),
    ]

    for dirpath, market_prefix in result_dirs:
        p = Path(dirpath)
        if not p.exists():
            continue
        for f in p.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            records = data.get("results", data if isinstance(data, list) else [])
            for r in records:
                K = r.get("K")
                model = r.get("model", "")
                loss = r.get("loss", "")
                N = r.get("N")
                key = (market_prefix, K, model, loss, N)
                results.setdefault(key, []).append(r)

    return results


def compute_dfl_gain(results: dict, case: dict) -> float | None:
    """Compute DFL gain % for a prediction case.

    Returns (MSE_TE - DFL_TE) / MSE_TE * 100, positive = DFL better.
    """
    market_aliases = {
        "EPA_AQS": "EPA",
        "NOAA_ASOS": "Weather",
        "nikkei225": "Nikkei225",
        "eurostoxx50": "EuroStoxx50",
        "asx200": "ASX200",
        "hangseng": "HangSeng",
    }
    market = market_aliases.get(case["market"], case["market"])
    K = case["K"]
    model_name = case.get("model", "shrinkage")
    N = case.get("N")

    model_map = {
        "shrinkage": "shrinkage",
        "structured": "structured",
        "spatial_structured": "spatial_structured",
    }
    model_key = model_map.get(model_name, model_name)

    mse_key = (market, K, model_key, "mse", N)
    dfl_key = (market, K, model_key, "task", N)

    mse_results = results.get(mse_key, [])
    dfl_results = results.get(dfl_key, [])

    if not mse_results or not dfl_results:
        mse_candidates = []
        dfl_candidates = []
        for key, vals in results.items():
            mk, kk, mdl, loss, nn = key
            if mk == market and kk == K and model_key in mdl.lower():
                if N and nn and max(N, nn) / max(min(N, nn), 1) > 2.5:
                    continue
                if loss == "mse":
                    mse_candidates.append((nn, vals))
                elif loss == "task":
                    dfl_candidates.append((nn, vals))
        if mse_candidates:
            mse_candidates.sort(key=lambda x: abs((x[0] or 0) - (N or 0)))
            mse_results = mse_candidates[0][1]
        if dfl_candidates:
            dfl_candidates.sort(key=lambda x: abs((x[0] or 0) - (N or 0)))
            dfl_results = dfl_candidates[0][1]

    if not mse_results or not dfl_results:
        return None

    mse_by_fold = {r.get("fold", i): r["tracking_error"] for i, r in enumerate(mse_results)}
    dfl_by_fold = {r.get("fold", i): r["tracking_error"] for i, r in enumerate(dfl_results)}
    common_folds = sorted(set(mse_by_fold) & set(dfl_by_fold))
    if not common_folds:
        return None

    mse_te = np.mean([mse_by_fold[f] for f in common_folds])
    dfl_te = np.mean([dfl_by_fold[f] for f in common_folds])

    if mse_te == 0:
        return 0.0
    return (mse_te - dfl_te) / mse_te * 100


DOMAIN_HET = {"finance": 1.0, "weather": 0.144, "air_quality": 0.606}


def refined_v2_prediction(case: dict) -> str:
    """Apply refined v2 diagnostic using effective_dim = model_dim × het_factor."""
    het = DOMAIN_HET.get(case["domain"], 1.0)
    eff_dim = case["model_dim"] * het
    alignment = case["alignment"]

    if eff_dim <= 1.5 and alignment > 0.4:
        return "mse_sufficient"
    if eff_dim <= 1.5 and alignment <= 0.4:
        return "equivalent"
    if eff_dim >= 4 and alignment < 0.40:
        return "dfl_wins"
    return "ambiguous"


def score_prediction(predicted: str, dfl_gain_pct: float, threshold: float = 1.0) -> bool:
    """Score a single prediction against actual DFL gain.

    threshold: minimum % gain to declare DFL "wins" (default 1%).
    """
    if predicted == "dfl_wins":
        return dfl_gain_pct > threshold
    elif predicted == "mse_sufficient":
        return dfl_gain_pct < threshold
    elif predicted == "equivalent":
        return abs(dfl_gain_pct) < threshold
    elif predicted == "ambiguous":
        return True  # ambiguous is always "correct" (non-committal)
    return False


def main():
    preds = load_predictions()
    results = load_all_results()
    threshold = preds.get("scoring_threshold_pct", 1.0)

    print(f"Loaded {len(preds['cases'])} predictions, {len(results)} result groups")
    print(f"Scoring threshold: {threshold}%\n")

    scored = []
    unmatched = []

    for case in preds["cases"]:
        gain = compute_dfl_gain(results, case)
        if gain is None:
            unmatched.append(case["id"])
            continue

        correct = score_prediction(case["predict"], gain, threshold)
        scored.append({
            "id": case["id"],
            "predicted": case["predict"],
            "alignment": case["alignment"],
            "model_dim": case["model_dim"],
            "dfl_gain_pct": round(gain, 3),
            "correct": correct,
            "domain": case["domain"],
        })

    if not scored:
        print("No predictions could be matched to results.")
        print(f"Unmatched: {len(unmatched)}")
        for u in unmatched[:10]:
            print(f"  {u}")
        return

    # Overall accuracy
    n_correct = sum(s["correct"] for s in scored)
    n_total = len(scored)
    print(f"{'='*60}")
    print(f"OVERALL: {n_correct}/{n_total} = {n_correct/n_total:.1%}")
    print(f"Unmatched (no results): {len(unmatched)}")

    # Per-category accuracy
    print(f"\n{'='*60}")
    print("PER CATEGORY:")
    categories = defaultdict(list)
    for s in scored:
        categories[s["predicted"]].append(s)

    for cat in ["dfl_wins", "mse_sufficient", "equivalent", "ambiguous"]:
        items = categories.get(cat, [])
        if not items:
            continue
        correct = sum(s["correct"] for s in items)
        print(f"  {cat}: {correct}/{len(items)} = {correct/len(items):.1%}")
        for s in items:
            mark = "✓" if s["correct"] else "✗"
            print(f"    {mark} {s['id']}: gain={s['dfl_gain_pct']:+.2f}%")

    # Per-domain accuracy
    print(f"\n{'='*60}")
    print("PER DOMAIN:")
    domains = defaultdict(list)
    for s in scored:
        domains[s["domain"]].append(s)
    for domain, items in sorted(domains.items()):
        correct = sum(s["correct"] for s in items)
        print(f"  {domain}: {correct}/{len(items)} = {correct/len(items):.1%}")

    # Confusion matrix
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX (predicted vs actual):")
    actual_labels = []
    for s in scored:
        if s["dfl_gain_pct"] > threshold:
            actual_labels.append("dfl_wins")
        elif s["dfl_gain_pct"] < -threshold:
            actual_labels.append("mse_wins")
        else:
            actual_labels.append("equivalent")

    pred_cats = sorted(set(s["predicted"] for s in scored))
    actual_cats = sorted(set(actual_labels))
    print(f"{'':>18s}", end="")
    for ac in actual_cats:
        print(f" {ac:>12s}", end="")
    print()
    for pc in pred_cats:
        print(f"  {pc:>16s}", end="")
        for ac in actual_cats:
            count = sum(1 for s, al in zip(scored, actual_labels)
                        if s["predicted"] == pc and al == ac)
            print(f" {count:>12d}", end="")
        print()

    # Refined v2 scoring (effective_dim = model_dim × het_factor)
    print(f"\n{'='*60}")
    print("REFINED V2 DIAGNOSTIC (effective_dim = dim × het):")
    v2_scored = []
    for case in preds["cases"]:
        gain = compute_dfl_gain(results, case)
        if gain is None:
            continue
        v2_pred = refined_v2_prediction(case)
        v2_correct = score_prediction(v2_pred, gain, threshold)
        v2_scored.append({
            "id": case["id"],
            "original": case["predict"],
            "v2": v2_pred,
            "changed": case["predict"] != v2_pred,
            "dfl_gain_pct": round(gain, 3),
            "v2_correct": v2_correct,
        })

    v2_correct_count = sum(s["v2_correct"] for s in v2_scored)
    print(f"  V2 accuracy: {v2_correct_count}/{len(v2_scored)} = {v2_correct_count/len(v2_scored):.1%}")
    print(f"  Original:    {n_correct}/{n_total} = {n_correct/n_total:.1%}")
    changed = [s for s in v2_scored if s["changed"]]
    if changed:
        print(f"\n  Predictions changed by v2 ({len(changed)}):")
        for s in changed:
            mark = "✓" if s["v2_correct"] else "✗"
            print(f"    {mark} {s['id']}: {s['original']} → {s['v2']}, gain={s['dfl_gain_pct']:+.2f}%")

    # Save scoring results
    output = {
        "overall_accuracy": n_correct / n_total,
        "v2_accuracy": v2_correct_count / len(v2_scored),
        "n_scored": n_total,
        "n_unmatched": len(unmatched),
        "threshold_pct": threshold,
        "scored": scored,
        "v2_scored": v2_scored,
        "unmatched": unmatched,
        "per_category": {
            cat: {
                "accuracy": sum(s["correct"] for s in items) / len(items),
                "n": len(items),
            }
            for cat, items in categories.items()
        },
    }
    out_path = Path("results/diagnostic/scoring.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    print(f"\nSaved scoring to {out_path}")


if __name__ == "__main__":
    main()
