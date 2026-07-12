"""Evaluate strict M2/M3/PCA models with AP, lateral, and composite studies.

The evaluation endpoint is always an exclusive, pure M2/M3/PCA label.  The two
available composite-label patients are never scored.  Sensitivity analyses may
add them to training as down-weighted examples of a *present* target component:
M2+A3 -> M2 present and M3+M4 -> M3 present.  Those analyses therefore do not
redefine the strict headline endpoint.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import metrics as M
from anatomy_task_experiments import anatomy_codes, load_cache
from experiments import make_folds


LABELS = ("m2", "m3", "pca")


@dataclass
class Dataset:
    name: str
    family: str
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    identities: list[str]
    views: np.ndarray | None = None
    weak_x: np.ndarray | None = None
    weak_y: np.ndarray | None = None
    weak_groups: np.ndarray | None = None
    weak_weight: float = 0.0
    definition: str = ""


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ap-uniform", type=Path, required=True)
    p.add_argument("--ap-temporal", type=Path, required=True)
    p.add_argument("--lat-uniform", type=Path, required=True)
    p.add_argument("--lat-temporal", type=Path, required=True)
    p.add_argument("--seeds", default="0:20")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("runs/multiview_anatomy/results.json"))
    return p.parse_args()


def parse_seeds(text: str) -> list[int]:
    if ":" in text:
        start, stop = map(int, text.split(":"))
        return list(range(start, stop))
    return [int(x) for x in text.split(",")]


def run_index(meta: dict) -> int:
    match = re.search(r"_(\d+)$", str(meta["run_column"]))
    if not match:
        raise ValueError(f"Cannot parse run index from {meta['run_column']}")
    return int(match.group(1))


def pair_key(meta: dict) -> tuple[str, int]:
    return str(meta["study_key"]), run_index(meta)


def strict_label(text: str) -> int | None:
    codes = anatomy_codes(text)
    if codes == {"M2"}:
        return 0
    if codes == {"M3"}:
        return 1
    if codes and all(code.startswith("P") for code in codes):
        return 2
    return None


def weak_label(text: str) -> int | None:
    codes = anatomy_codes(text)
    if codes == {"M2", "A3"}:
        return 0
    if codes == {"M3", "M4"}:
        return 1
    return None


def identity(meta: dict) -> str:
    return f"{meta['study_key']}:{run_index(meta)}:{meta['view']}"


def validate_aligned(a: dict, b: dict, name: str) -> None:
    if tuple(a["metadata_identity"]) != tuple(b["metadata_identity"]):
        raise ValueError(f"Uniform and temporal {name} caches are not aligned")


def rows(cache: dict, label_fn) -> np.ndarray:
    return np.asarray([i for i, m in enumerate(cache["meta"]) if label_fn(m["label_text"]) is not None])


def labels(cache: dict, indices: np.ndarray, label_fn) -> np.ndarray:
    return np.asarray([label_fn(cache["meta"][i]["label_text"]) for i in indices], dtype=int)


def append_view(x: np.ndarray, view_value: float) -> np.ndarray:
    return np.column_stack((x, np.full(len(x), view_value, dtype=x.dtype)))


def make_single(name: str, cache: dict, feature: np.ndarray, view: str) -> Dataset:
    ix = rows(cache, strict_label)
    return Dataset(
        name=name, family="single_view", x=feature[ix], y=labels(cache, ix, strict_label),
        groups=np.asarray(cache["groups"])[ix], identities=[identity(cache["meta"][i]) for i in ix],
        views=np.asarray([view] * len(ix)), definition=f"Strict pure M2/M3/PCA, {view} runs only.",
    )


def make_pooled(name: str, ap: dict, lat: dict, ap_x: np.ndarray, lat_x: np.ndarray,
                weak_weight: float = 0.0) -> Dataset:
    ai, li = rows(ap, strict_label), rows(lat, strict_label)
    x = np.concatenate((append_view(ap_x[ai], 0.0), append_view(lat_x[li], 1.0)))
    y = np.concatenate((labels(ap, ai, strict_label), labels(lat, li, strict_label)))
    groups = np.concatenate((np.asarray(ap["groups"])[ai], np.asarray(lat["groups"])[li]))
    identities = [identity(ap["meta"][i]) for i in ai] + [identity(lat["meta"][i]) for i in li]
    views = np.asarray(["AP"] * len(ai) + ["Lateral"] * len(li))
    result = Dataset(name, "pooled", x, y, groups, identities, views=views,
                     weak_weight=weak_weight,
                     definition="AP and lateral strict runs pooled with a view indicator; patient grouping prevents cross-view leakage.")
    if weak_weight:
        aw, lw = rows(ap, weak_label), rows(lat, weak_label)
        result.weak_x = np.concatenate((append_view(ap_x[aw], 0.0), append_view(lat_x[lw], 1.0)))
        result.weak_y = np.concatenate((labels(ap, aw, weak_label), labels(lat, lw, weak_label)))
        result.weak_groups = np.concatenate((np.asarray(ap["groups"])[aw], np.asarray(lat["groups"])[lw]))
    return result


def make_paired(name: str, ap: dict, lat: dict, ap_x: np.ndarray, lat_x: np.ndarray,
                weak_weight: float = 0.0) -> tuple[Dataset, dict]:
    amap = {pair_key(m): i for i, m in enumerate(ap["meta"])}
    lmap = {pair_key(m): i for i, m in enumerate(lat["meta"])}
    common = sorted(set(amap).intersection(lmap))
    strict, discordant, weak = [], [], []
    for key in common:
        ai, li = amap[key], lmap[key]
        ay, ly = strict_label(ap["meta"][ai]["label_text"]), strict_label(lat["meta"][li]["label_text"])
        if ay is not None and ly is not None:
            (strict if ay == ly else discordant).append((key, ai, li, ay, ly))
        aw, lw = weak_label(ap["meta"][ai]["label_text"]), weak_label(lat["meta"][li]["label_text"])
        if aw is not None and aw == lw:
            weak.append((key, ai, li, aw))
    x = np.asarray([np.concatenate((ap_x[ai], lat_x[li])) for _, ai, li, _, _ in strict])
    y = np.asarray([ay for _, _, _, ay, _ in strict], dtype=int)
    groups = np.asarray([ap["groups"][ai] for _, ai, _, _, _ in strict])
    ids = [f"{key[0]}:{key[1]}:paired" for key, *_ in strict]
    result = Dataset(name, "paired", x, y, groups, ids, weak_weight=weak_weight,
                     definition="Concatenated AP+lateral features for matched runs with concordant strict labels; discordant pairs excluded.")
    if weak_weight:
        result.weak_x = np.asarray([np.concatenate((ap_x[ai], lat_x[li])) for _, ai, li, _ in weak])
        result.weak_y = np.asarray([wy for _, _, _, wy in weak], dtype=int)
        result.weak_groups = np.asarray([ap["groups"][ai] for _, ai, _, _ in weak])
    audit = {
        "matched_pairs": len(common), "concordant_strict_pairs": len(strict),
        "discordant_strict_pairs": len(discordant), "weak_composite_pairs": len(weak),
        "discordant": [{"study_key": k[0], "run": k[1], "ap": LABELS[ay], "lateral": LABELS[ly]}
                       for k, _, _, ay, ly in discordant],
    }
    return result, audit


def make_matched_single(name: str, view: str, ap: dict, lat: dict,
                        ap_x: np.ndarray, lat_x: np.ndarray) -> Dataset:
    """Single-view control restricted to the exact concordant paired cohort."""
    amap = {pair_key(m): i for i, m in enumerate(ap["meta"])}
    lmap = {pair_key(m): i for i, m in enumerate(lat["meta"])}
    kept = []
    for key in sorted(set(amap).intersection(lmap)):
        ai, li = amap[key], lmap[key]
        ay, ly = strict_label(ap["meta"][ai]["label_text"]), strict_label(lat["meta"][li]["label_text"])
        if ay is not None and ay == ly:
            kept.append((key, ai, li, ay))
    chosen_x, chosen_cache, pos = (ap_x, ap, 1) if view == "AP" else (lat_x, lat, 2)
    return Dataset(
        name=name, family="paired_control", x=np.asarray([chosen_x[row[pos]] for row in kept]),
        y=np.asarray([row[3] for row in kept], dtype=int),
        groups=np.asarray([chosen_cache["groups"][row[pos]] for row in kept]),
        identities=[f"{row[0][0]}:{row[0][1]}:{view}" for row in kept],
        definition=f"{view}-only control on the exact concordant cohort used for paired fusion.",
    )


def fit_model(x: np.ndarray, y: np.ndarray, c: float, sample_weight: np.ndarray | None = None):
    scaler = StandardScaler()
    scaler.fit(x, sample_weight=sample_weight)
    xs = scaler.transform(x)
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
    model.fit(xs, y, sample_weight=sample_weight)
    return scaler, model


def score_dataset(ds: Dataset, seed: int, folds: int, c: float) -> dict:
    split_count = min(folds, *(len(set(ds.groups[ds.y == k])) for k in range(3)))
    splits = make_folds(ds.y, ds.groups, split_count, seed)
    pred = np.full(len(ds.y), -1, dtype=int)
    prob = np.zeros((len(ds.y), 3), dtype=float)
    weak_used = []
    for train, valid in splits:
        x, y = ds.x[train], ds.y[train]
        sw = np.ones(len(train), dtype=float)
        count = 0
        if ds.weak_x is not None:
            # A weak patient may be used only when it is not a validation patient.
            keep = ~np.isin(ds.weak_groups, np.unique(ds.groups[valid]))
            wx, wy = ds.weak_x[keep], ds.weak_y[keep]
            count = len(wy)
            x, y = np.concatenate((x, wx)), np.concatenate((y, wy))
            sw = np.concatenate((sw, np.full(count, ds.weak_weight)))
        scaler, model = fit_model(x, y, c, sw)
        pred[valid] = model.predict(scaler.transform(ds.x[valid]))
        prob[valid] = model.predict_proba(scaler.transform(ds.x[valid]))
        weak_used.append(count)
    metric = M.compute_metrics(ds.y, pred, 3)
    result = {
        "seed": seed, "folds": split_count, "macro_f1": metric["macro_f1"],
        "balanced_accuracy": metric["balanced_accuracy"], "accuracy": metric["accuracy"],
        "macro_auprc": float(np.mean([average_precision_score(ds.y == k, prob[:, k]) for k in range(3)])),
        "per_class_precision": metric["per_class"]["precision"],
        "per_class_recall": metric["per_class"]["recall"],
        "per_class_f1": metric["per_class"]["f1"], "confusion": metric["confusion"],
        "weak_rows_per_fold": weak_used,
    }
    if ds.views is not None and len(set(ds.views)) > 1:
        result["view_metrics"] = {}
        for view in sorted(set(ds.views)):
            keep = ds.views == view
            vm = M.compute_metrics(ds.y[keep], pred[keep], 3)
            result["view_metrics"][view] = {"macro_f1": vm["macro_f1"], "accuracy": vm["accuracy"],
                                                   "per_class_f1": vm["per_class"]["f1"]}
    return result


def aggregate(rows: list[dict]) -> dict:
    keys = ("macro_f1", "balanced_accuracy", "accuracy", "macro_auprc",
            "per_class_precision", "per_class_recall", "per_class_f1")
    out = {}
    for key in keys:
        values = np.asarray([r[key] for r in rows], dtype=float)
        out[f"{key}_mean"] = values.mean(axis=0).tolist() if values.ndim > 1 else float(values.mean())
        out[f"{key}_std"] = values.std(axis=0, ddof=1).tolist() if values.ndim > 1 else float(values.std(ddof=1))
    if "view_metrics" in rows[0]:
        out["view_metrics"] = {}
        for view in rows[0]["view_metrics"]:
            out["view_metrics"][view] = {}
            for key in ("macro_f1", "accuracy", "per_class_f1"):
                values = np.asarray([r["view_metrics"][view][key] for r in rows], dtype=float)
                out["view_metrics"][view][f"{key}_mean"] = values.mean(axis=0).tolist() if values.ndim > 1 else float(values.mean())
                out["view_metrics"][view][f"{key}_std"] = values.std(axis=0, ddof=1).tolist() if values.ndim > 1 else float(values.std(ddof=1))
    return out


def class_counts(ds: Dataset) -> dict:
    return {LABELS[k]: int((ds.y == k).sum()) for k in range(3)}


def main() -> int:
    a = args()
    ap, apt, lat, latt = map(load_cache, (a.ap_uniform, a.ap_temporal, a.lat_uniform, a.lat_temporal))
    validate_aligned(ap, apt, "AP"); validate_aligned(lat, latt, "lateral")
    datasets = [
        make_single("ap_uniform", ap, ap["mean"], "AP"),
        make_single("lateral_uniform", lat, lat["mean"], "Lateral"),
        make_single("lateral_temporal", lat, latt["mean"], "Lateral"),
        make_pooled("pooled_uniform", ap, lat, ap["mean"], lat["mean"]),
        make_pooled("pooled_temporal", ap, lat, apt["mean"], latt["mean"]),
        make_pooled("pooled_uniform_weak05", ap, lat, ap["mean"], lat["mean"], 0.5),
        make_pooled("pooled_uniform_weak10", ap, lat, ap["mean"], lat["mean"], 1.0),
    ]
    paired, pair_audit = make_paired("paired_uniform", ap, lat, ap["mean"], lat["mean"])
    paired_t, _ = make_paired("paired_temporal", ap, lat, apt["mean"], latt["mean"])
    paired_w, _ = make_paired("paired_uniform_weak05", ap, lat, ap["mean"], lat["mean"], 0.5)
    paired_w10, _ = make_paired("paired_uniform_weak10", ap, lat, ap["mean"], lat["mean"], 1.0)
    datasets += [
        make_matched_single("matched_ap_uniform", "AP", ap, lat, ap["mean"], lat["mean"]),
        make_matched_single("matched_lateral_uniform", "Lateral", ap, lat, ap["mean"], lat["mean"]),
        make_matched_single("matched_ap_temporal", "AP", ap, lat, apt["mean"], latt["mean"]),
        make_matched_single("matched_lateral_temporal", "Lateral", ap, lat, apt["mean"], latt["mean"]),
        paired, paired_t, paired_w, paired_w10,
    ]
    seeds = parse_seeds(a.seeds)
    per_seed, summaries = {}, []
    for ds in datasets:
        per_seed[ds.name] = {}
        for c in (1.0, 3.0):
            key = f"std_logreg_c{int(c)}"
            print(f"Evaluating {ds.name} / {key} ({len(ds.y)} strict rows, {len(set(ds.groups))} groups)", flush=True)
            result_rows = [score_dataset(ds, seed, a.folds, c) for seed in seeds]
            per_seed[ds.name][key] = result_rows
            summary = aggregate(result_rows)
            summary.update({"dataset": ds.name, "family": ds.family, "recipe": key,
                            "n_strict_rows": len(ds.y), "n_groups": len(set(ds.groups)),
                            "class_counts": class_counts(ds), "weak_weight": ds.weak_weight,
                            "n_weak_rows": 0 if ds.weak_y is None else len(ds.weak_y),
                            "n_weak_groups": 0 if ds.weak_groups is None else len(set(ds.weak_groups)),
                            "definition": ds.definition})
            summaries.append(summary)
    summaries.sort(key=lambda r: r["macro_f1_mean"], reverse=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    model_dir = a.out.parent / "models"; model_dir.mkdir(exist_ok=True)
    artifacts = {}
    by_name = {ds.name: ds for ds in datasets}
    for family in ("single_view", "pooled", "paired"):
        best = next(r for r in summaries if r["family"] == family)
        ds = by_name[best["dataset"]]; c = float(best["recipe"].removeprefix("std_logreg_c"))
        x, y, sw = ds.x, ds.y, np.ones(len(ds.y))
        if ds.weak_x is not None:
            x, y = np.concatenate((x, ds.weak_x)), np.concatenate((y, ds.weak_y))
            sw = np.concatenate((sw, np.full(len(ds.weak_y), ds.weak_weight)))
        scaler, model = fit_model(x, y, c, sw)
        path = model_dir / f"{family}__{ds.name}__{best['recipe']}.joblib"
        joblib.dump({"scaler": scaler, "model": model, "labels": LABELS, "dataset": ds.name,
                     "definition": ds.definition, "weak_weight": ds.weak_weight,
                     "selection_note": "Selected by repeated patient-grouped CV; full-data fit is not held-out evidence."}, path)
        artifacts[family] = str(path)

    # Preserve the defensible strict paired model separately even if the weak-label
    # sensitivity wins the model-selection table by a few thousandths.
    strict_ds = by_name["paired_uniform"]
    strict_scaler, strict_model = fit_model(strict_ds.x, strict_ds.y, 1.0, np.ones(len(strict_ds.y)))
    strict_path = model_dir / "paired_strict__paired_uniform__std_logreg_c1.joblib"
    joblib.dump({"scaler": strict_scaler, "model": strict_model, "labels": LABELS,
                 "dataset": strict_ds.name, "definition": strict_ds.definition,
                 "input_schema": "2048 values: 1024D mean AP DINOv2-L embedding followed by 1024D mean lateral embedding",
                 "weak_weight": 0.0,
                 "selection_note": "Strict full-data fit after repeated grouped CV; not held-out evidence."}, strict_path)
    artifacts["paired_strict"] = str(strict_path)
    temporal_ds = by_name["paired_temporal"]
    temporal_scaler, temporal_model = fit_model(
        temporal_ds.x, temporal_ds.y, 1.0, np.ones(len(temporal_ds.y))
    )
    temporal_path = model_dir / "paired_strict__paired_temporal__std_logreg_c1.joblib"
    joblib.dump({"scaler": temporal_scaler, "model": temporal_model, "labels": LABELS,
                 "dataset": temporal_ds.name, "definition": temporal_ds.definition,
                 "input_schema": "2048 values: 1024D mean AP temporal-change DINOv2-L embedding followed by the matched 1024D lateral embedding",
                 "weak_weight": 0.0,
                 "selection_note": "Best strict paired mean macro-F1 in repeated grouped CV; full-data fit is not held-out evidence."}, temporal_path)
    artifacts["paired_strict_temporal"] = str(temporal_path)

    composite_audit = []
    for cache in (ap, lat):
        for m, g in zip(cache["meta"], cache["groups"]):
            wy = weak_label(m["label_text"])
            if wy is not None:
                composite_audit.append({"study_key": m["study_key"], "accession": m["accession"],
                                        "run": run_index(m), "view": m["view"], "original_label": m["label_text"],
                                        "codes": sorted(anatomy_codes(m["label_text"])),
                                        "weak_target_component": LABELS[wy], "group": str(g)})
    output = {
        "protocol": {"labels": LABELS, "seeds": seeds, "folds": a.folds,
                     "selection_metric": "mean macro-F1", "validation": "strict pure labels only",
                     "grouping": "Study_Key/patient; all views and runs from a patient stay in one fold",
                     "composite_policy": "training-only component-presence sensitivity at weights 0.5 and 1.0; never validation"},
        "feature_sources": {
            "ap_uniform": {"path": str(a.ap_uniform), "sha256": ap["sha256"], "signature": ap["signature"]},
            "ap_temporal": {"path": str(a.ap_temporal), "sha256": apt["sha256"], "signature": apt["signature"]},
            "lateral_uniform": {"path": str(a.lat_uniform), "sha256": lat["sha256"], "signature": lat["signature"]},
            "lateral_temporal": {"path": str(a.lat_temporal), "sha256": latt["sha256"], "signature": latt["signature"]},
        },
        "pair_audit": pair_audit, "composite_audit": composite_audit,
        "summaries": summaries, "per_seed": per_seed, "artifacts": artifacts,
    }
    with a.out.open("w") as f: json.dump(output, f, indent=2)
    print(f"Wrote {a.out}")
    for r in summaries:
        print(f"{r['dataset']:28s} {r['recipe']:17s} {r['macro_f1_mean']:.3f} +/- {r['macro_f1_std']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
