"""Ensemble the best neural probe with a classical SVM over radiomics features.

Motivation
----------
Every prior experiment in this repo probes a *single* representation (a frozen
neural backbone) with a linear head.  The deep features have repeatedly stalled
on the m3 minority class: an attentive-pooling probe over the full token
sequence could not separate m2 from m3, which means that distinction is only
weakly present in the learned features.  Hand-crafted radiomics texture
statistics are a genuinely different view of the same runs, so a heterogeneous
ensemble — the classical route (SVM on radiomics) fused with the deep route — is
a principled thing to try, and the medical-imaging literature repeatedly finds
that radiomics + deep-feature stacking beats either member alone.

Rigor
-----
This mirrors the discipline already established in
``three_class_augmentation_experiments.py``:

* the same patient-grouped ``StratifiedGroupKFold`` folds, repeated over many
  split seeds, with mean ± std reported;
* every learned combiner (fusion weight, stacking meta-learner) is fit with an
  *inner* grouped cross-validation on the outer training patients only, so no
  validation patient ever influences the combiner — the ensemble numbers are
  leak-free, not a best-of selection;
* the neural-only probe is the reference, so the table directly answers "does
  adding radiomics help, and by how much?".

Feature caches are aligned by canonical run metadata (never by row position),
exactly like the existing comparison scripts.

Examples:
    # Real run: best deep fusion (V-JEPA ViT-g/384 + DINOv2-L/252) + radiomics.
    python ensemble_experiments.py \
      --nn vjepa=runs/ap_exp_vitg384_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm.npz \
      --nn dino=runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
      --radiomics runs/ap_radiomics/cache/radiomics_otsu_bc32_16_AP_positive_subtype.npz \
      --seeds 0:20 --out runs/ensemble/ap_ensemble_20seeds.json

    # Machinery check with no data / no caches (fabricated grouped features):
    python ensemble_experiments.py --synthetic-demo --seeds 0:10
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC, LinearSVC

import metrics as M
from compare_feature_caches import canonical_metadata, file_sha256
from experiments import make_folds

NN_CLASSIFIERS = ("logreg_c3", "logreg_c1", "linsvm_cal")
RADIOMICS_CLASSIFIERS = ("svm_rbf", "linsvm_cal", "logreg_c1")
ALPHA_GRID = tuple(round(0.1 * step, 2) for step in range(11))


# ----------------------------------------------------------------------------------------
# Cache loading + alignment (shared conventions with compare_feature_caches.py).
# ----------------------------------------------------------------------------------------
def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    return name, Path(path)


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = value.split(":", 1)
        return list(range(int(start), int(stop)))
    return [int(item) for item in value.split(",") if item.strip()]


def load_cache(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    required = ("mean", "labels", "groups", "meta")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
    loaded = {key: data[key] for key in required}
    if not np.isfinite(loaded["mean"]).all():
        raise ValueError(f"{path} contains non-finite mean features.")
    loaded["metadata_identity"] = canonical_metadata(loaded["meta"])
    loaded["sha256"] = file_sha256(path)
    loaded["path"] = str(path)
    return loaded


def aligned_to_reference(cache: dict[str, object], reference_identity) -> dict[str, object]:
    identities = cache["metadata_identity"]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate canonical metadata identities in {cache['path']}.")
    index_by_identity = {identity: index for index, identity in enumerate(identities)}
    missing = [identity for identity in reference_identity if identity not in index_by_identity]
    if missing:
        raise ValueError(f"{cache['path']} is missing {len(missing)} runs present in the reference cache.")
    order = np.asarray([index_by_identity[identity] for identity in reference_identity], dtype=int)
    result = dict(cache)
    for key in ("mean", "labels", "groups"):
        result[key] = np.asarray(cache[key])[order]
    result["metadata_identity"] = tuple(cache["metadata_identity"][index] for index in order)
    return result


# ----------------------------------------------------------------------------------------
# Base learners.  Each returns full-width class probabilities.
# ----------------------------------------------------------------------------------------
def build_learner(kind: str):
    scaler = StandardScaler()
    if kind == "logreg_c3":
        estimator = LogisticRegression(max_iter=5000, C=3.0, class_weight="balanced")
    elif kind == "logreg_c1":
        estimator = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    elif kind == "svm_rbf":
        # Calibrated decision function gives leak-free predict_proba without the
        # deprecated SVC(probability=True) internal CV.
        estimator = CalibratedClassifierCV(
            SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", random_state=0),
            cv=3,
        )
    elif kind == "linsvm_cal":
        estimator = CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", max_iter=20000, random_state=0),
            cv=3,
        )
    else:
        raise ValueError(f"Unknown learner kind: {kind}")
    return make_pipeline(scaler, estimator)


def predict_full_proba(pipeline, X: np.ndarray, num_classes: int) -> np.ndarray:
    """predict_proba with columns placed into the full 0..num_classes-1 layout."""
    raw = pipeline.predict_proba(X)
    classes = pipeline.classes_.astype(int)
    full = np.zeros((X.shape[0], num_classes), dtype=float)
    full[:, classes] = raw
    return full


def inner_oof_proba(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    kind: str,
    num_classes: int,
    seed: int,
    outer_fold_index: int,
) -> np.ndarray:
    """Grouped out-of-fold probabilities on the outer-training patients only."""
    inner_y = y[train]
    inner_groups = groups[train]
    n_splits = min(3, len(set(inner_groups)), int(np.min(np.bincount(inner_y, minlength=num_classes))))
    oof = np.zeros((len(train), num_classes), dtype=float)
    if n_splits < 2:
        pipeline = build_learner(kind).fit(X[train], inner_y)
        return np.tile(predict_full_proba(pipeline, X[train], num_classes).mean(axis=0), (len(train), 1))
    folds = make_folds(inner_y, inner_groups, n_splits, seed + 1009 * (outer_fold_index + 1))
    for local_train, local_valid in folds:
        pipeline = build_learner(kind).fit(X[train][local_train], inner_y[local_train])
        oof[local_valid] = predict_full_proba(pipeline, X[train][local_valid], num_classes)
    return oof


# ----------------------------------------------------------------------------------------
# One outer fold: compute every ensemble's validation probabilities together.
# ----------------------------------------------------------------------------------------
def evaluate_fold(
    nn_features: np.ndarray,
    radiomics_features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    num_classes: int,
    nn_kind: str,
    radiomics_kind: str,
    seed: int,
    outer_fold_index: int,
) -> tuple[dict[str, np.ndarray], float]:
    nn_pipeline = build_learner(nn_kind).fit(nn_features[train], y[train])
    radiomics_pipeline = build_learner(radiomics_kind).fit(radiomics_features[train], y[train])
    nn_valid = predict_full_proba(nn_pipeline, nn_features[valid], num_classes)
    radiomics_valid = predict_full_proba(radiomics_pipeline, radiomics_features[valid], num_classes)

    # Inner OOF on training patients only — reused by the weighted fusion and the stack.
    nn_oof = inner_oof_proba(nn_features, y, groups, train, nn_kind, num_classes, seed, outer_fold_index)
    radiomics_oof = inner_oof_proba(
        radiomics_features, y, groups, train, radiomics_kind, num_classes, seed, outer_fold_index
    )
    inner_y = y[train]

    # Weighted late fusion: pick alpha on inner OOF, apply to the held-out fold.
    best_alpha, best_score = 1.0, -1.0
    for alpha in ALPHA_GRID:
        blended = alpha * nn_oof + (1.0 - alpha) * radiomics_oof
        score = M.compute_metrics(inner_y, blended.argmax(axis=1), num_classes)["macro_f1"]
        if score > best_score + 1e-9:  # ties keep the lower alpha, i.e. more radiomics never on a tie
            best_score, best_alpha = score, alpha
    weighted = best_alpha * nn_valid + (1.0 - best_alpha) * radiomics_valid

    # Stacking: meta-logistic-regression over the two members' inner OOF probabilities.
    meta_features_train = np.concatenate([nn_oof, radiomics_oof], axis=1)
    meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"),
    ).fit(meta_features_train, inner_y)
    meta_valid = predict_full_proba(meta, np.concatenate([nn_valid, radiomics_valid], axis=1), num_classes)

    # Early fusion: one learner over standardized concatenation of both raw feature sets.
    early = build_learner(nn_kind).fit(
        np.concatenate([nn_features[train], radiomics_features[train]], axis=1), y[train]
    )
    early_valid = predict_full_proba(
        early, np.concatenate([nn_features[valid], radiomics_features[valid]], axis=1), num_classes
    )

    probabilities = {
        "nn_only": nn_valid,
        "radiomics_only": radiomics_valid,
        "late_fusion_equal": 0.5 * (nn_valid + radiomics_valid),
        "late_fusion_weighted": weighted,
        "stacking_logreg": meta_valid,
        "early_fusion": early_valid,
    }
    return probabilities, best_alpha


def macro_auprc(y: np.ndarray, probabilities: np.ndarray, num_classes: int) -> float:
    binary = label_binarize(y, classes=np.arange(num_classes))
    if num_classes == 2:
        binary = np.column_stack([1 - binary, binary])
    return float(average_precision_score(binary, probabilities, average="macro"))


def evaluate_seed(
    seed: int,
    nn_features: np.ndarray,
    radiomics_features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
    n_splits: int,
    nn_kind: str,
    radiomics_kind: str,
    method_names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    folds = make_folds(y, groups, n_splits, seed)
    pooled = {name: np.zeros((len(y), num_classes), dtype=float) for name in method_names}
    predictions = {name: np.full(len(y), -1, dtype=int) for name in method_names}
    alphas = []
    for outer_fold_index, (train, valid) in enumerate(folds):
        fold_probabilities, alpha = evaluate_fold(
            nn_features, radiomics_features, y, groups, train, valid,
            num_classes, nn_kind, radiomics_kind, seed, outer_fold_index,
        )
        alphas.append(alpha)
        for name in method_names:
            pooled[name][valid] = fold_probabilities[name]
            predictions[name][valid] = fold_probabilities[name].argmax(axis=1)

    rows: dict[str, dict[str, object]] = {}
    for name in method_names:
        computed = M.compute_metrics(y, predictions[name], num_classes)
        rows[name] = {
            "seed": seed,
            "macro_f1": computed["macro_f1"],
            "balanced_accuracy": computed["balanced_accuracy"],
            "accuracy": computed["accuracy"],
            "macro_auprc": macro_auprc(y, pooled[name], num_classes),
            "per_class_f1": computed["per_class"]["f1"],
        }
        if name == "late_fusion_weighted":
            rows[name]["mean_nn_weight"] = float(np.mean(alphas))
    return rows


# ----------------------------------------------------------------------------------------
# Summary + printing.
# ----------------------------------------------------------------------------------------
def summarize(per_method: dict[str, list[dict[str, object]]], reference: str, num_classes: int):
    reference_scores = np.asarray([row["macro_f1"] for row in per_method[reference]], dtype=float)
    summaries = []
    for name, rows in per_method.items():
        scores = np.asarray([row["macro_f1"] for row in rows], dtype=float)
        class_scores = np.asarray([row["per_class_f1"] for row in rows], dtype=float)
        delta = scores - reference_scores
        summary = {
            "name": name,
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "macro_auprc_mean": float(np.mean([row["macro_auprc"] for row in rows])),
            "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in rows])),
            "per_class_f1_mean": class_scores.mean(axis=0).tolist(),
            "delta_vs_reference_mean": float(delta.mean()),
            "delta_vs_reference_min": float(delta.min()),
            "delta_vs_reference_max": float(delta.max()),
            "wins_vs_reference": int((delta > 0).sum()),
            "ties_vs_reference": int((delta == 0).sum()),
            "n_seeds": len(scores),
        }
        if name == "late_fusion_weighted":
            summary["mean_nn_weight"] = float(np.mean([row["mean_nn_weight"] for row in rows]))
        summaries.append(summary)
    summaries.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    return summaries


def print_table(summaries, n_seeds: int) -> None:
    widths = (22, 14, 22, 12, 8)
    header = ("method", "macroF1", "class F1", "delta ref", "wins")
    print("  ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for row in summaries:
        values = (
            row["name"][: widths[0]],
            f"{row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f}",
            "/".join(f"{value:.2f}" for value in row["per_class_f1_mean"]),
            f"{row['delta_vs_reference_mean']:+.3f}",
            f"{row['wins_vs_reference']}/{n_seeds}",
        )
        print("  ".join(str(value).ljust(width) for value, width in zip(values, widths)))


# ----------------------------------------------------------------------------------------
# Synthetic demo: fabricate grouped features to validate the machinery end-to-end.
# ----------------------------------------------------------------------------------------
def synthetic_dataset(seed: int = 0):
    """Grouped, imbalanced 3-class data where radiomics carries independent m3 signal.

    NOT a performance claim.  The deep view is engineered to separate m2/other
    but to be weak on m3; the radiomics view is engineered to carry partial,
    independent m3 signal.  This lets the demo show the *mechanism* by which a
    heterogeneous ensemble can lift the minority class, and proves the fold and
    leakage discipline runs correctly.
    """
    rng = np.random.default_rng(seed)
    class_group_counts = {0: 150, 1: 45, 2: 40}  # roughly the real m2 / m3 / other ratio
    nn_dim, radiomics_dim = 48, 32
    nn_rows, radiomics_rows, labels, groups, identities = [], [], [], [], []
    patient = 0
    for class_index, n_groups in class_group_counts.items():
        for _ in range(n_groups):
            patient += 1
            runs = int(rng.integers(1, 3))
            for run in range(runs):
                # Deep features: strong class-0/2 separation, weak class-1 separation.
                nn_center = np.zeros(nn_dim)
                nn_center[0] = {0: -2.0, 1: 0.3, 2: 2.0}[class_index]
                nn_center[1] = {0: 0.0, 1: 0.4, 2: 1.2}[class_index]
                nn_rows.append(nn_center + rng.normal(0, 1.0, nn_dim))
                # Radiomics: independent partial signal that is actually informative for class 1.
                radiomics_center = np.zeros(radiomics_dim)
                radiomics_center[0] = {0: -0.6, 1: 1.4, 2: -0.2}[class_index]
                radiomics_center[1] = {0: -0.3, 1: 0.9, 2: 0.5}[class_index]
                radiomics_rows.append(radiomics_center + rng.normal(0, 1.1, radiomics_dim))
                labels.append(class_index)
                groups.append(f"P{patient:04d}")
                identities.append((f"P{patient:04d}", f"A{patient:04d}", "AP", f"AP_{run + 1}", f"class{class_index}"))
    return (
        np.asarray(nn_rows, dtype=np.float64),
        np.asarray(radiomics_rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=object),
        tuple(identities),
    )


# ----------------------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--nn", action="append", type=parse_named_path, default=[], metavar="NAME=PATH",
        help="Neural feature cache(s). Multiple are early-fused (concatenated mean features).",
    )
    parser.add_argument("--radiomics", type=Path, help="Radiomics feature cache from radiomics_probe.py.")
    parser.add_argument("--nn-classifier", choices=NN_CLASSIFIERS, default="logreg_c3")
    parser.add_argument("--radiomics-classifier", choices=RADIOMICS_CLASSIFIERS, default="svm_rbf")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0:20"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--synthetic-demo", action="store_true", help="Run on fabricated grouped features (no data needed).")
    parser.add_argument("--label-names", default="", help="Optional comma-separated class names for the printout.")
    parser.add_argument("--out", type=Path, default=Path("runs/ensemble/results.json"))
    return parser.parse_args()


def load_real_features(args: argparse.Namespace):
    if not args.nn or args.radiomics is None:
        raise SystemExit("Provide at least one --nn NAME=PATH and one --radiomics PATH (or use --synthetic-demo).")
    nn_caches = [(name, load_cache(path)) for name, path in args.nn]
    reference_identity = nn_caches[0][1]["metadata_identity"]
    nn_caches = [(name, aligned_to_reference(cache, reference_identity)) for name, cache in nn_caches]
    radiomics_cache = aligned_to_reference(load_cache(args.radiomics), reference_identity)

    y = np.asarray(nn_caches[0][1]["labels"], dtype=int)
    groups = np.asarray(nn_caches[0][1]["groups"], dtype=object)
    for name, cache in nn_caches:
        if not np.array_equal(np.asarray(cache["labels"], dtype=int), y):
            raise ValueError(f"Labels in nn cache '{name}' do not match after alignment.")
    if not np.array_equal(np.asarray(radiomics_cache["labels"], dtype=int), y):
        raise ValueError("Labels in the radiomics cache do not match after alignment.")

    nn_features = np.concatenate([np.asarray(cache["mean"], dtype=np.float64) for _, cache in nn_caches], axis=1)
    radiomics_features = np.asarray(radiomics_cache["mean"], dtype=np.float64)
    sources = {
        "nn": {name: {"path": cache["path"], "sha256": cache["sha256"]} for name, cache in nn_caches},
        "radiomics": {"path": radiomics_cache["path"], "sha256": radiomics_cache["sha256"]},
    }
    return nn_features, radiomics_features, y, groups, reference_identity, sources


def main() -> int:
    args = parse_args()
    if args.synthetic_demo:
        nn_features, radiomics_features, y, groups, reference_identity = synthetic_dataset()[:5]
        sources = {"synthetic": True}
        print("SYNTHETIC DEMO — fabricated grouped features, not a performance claim.\n")
    else:
        nn_features, radiomics_features, y, groups, reference_identity, sources = load_real_features(args)

    num_classes = int(y.max()) + 1
    if args.label_names:
        label_names = args.label_names.split(",")
    elif num_classes == 3:
        label_names = ["m2", "m3", "other_positive"]
    else:
        label_names = [f"class{index}" for index in range(num_classes)]

    n_splits = min(
        args.folds, len(set(groups)), int(np.min(np.bincount(y, minlength=num_classes)))
    )
    method_names = (
        "nn_only", "radiomics_only", "late_fusion_equal",
        "late_fusion_weighted", "stacking_logreg", "early_fusion",
    )
    print(
        f"Ensemble over {len(args.seeds)} seeds × {n_splits} folds — "
        f"{len(y)} runs / {len(set(groups))} patients, classes {label_names}\n"
        f"NN dim={nn_features.shape[1]} ({args.nn_classifier}), "
        f"radiomics dim={radiomics_features.shape[1]} ({args.radiomics_classifier})\n"
    )

    evaluated = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(evaluate_seed)(
            seed, nn_features, radiomics_features, y, groups,
            num_classes, n_splits, args.nn_classifier, args.radiomics_classifier, method_names,
        )
        for seed in args.seeds
    )
    per_method: dict[str, list[dict[str, object]]] = {name: [] for name in method_names}
    for seed_rows in evaluated:
        for name, row in seed_rows.items():
            per_method[name].append(row)
    for rows in per_method.values():
        rows.sort(key=lambda row: row["seed"])

    summaries = summarize(per_method, "nn_only", num_classes)
    print(f"Class F1 order: {'/'.join(label_names)}\n")
    print_table(summaries, len(args.seeds))

    protocol_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = {
        "protocol": {
            "label_names": label_names,
            "seeds": args.seeds,
            "folds": n_splits,
            "group": "Study_Key",
            "nn_classifier": args.nn_classifier,
            "radiomics_classifier": args.radiomics_classifier,
            "reference": "nn_only",
            "combiners": {
                "late_fusion_weighted": "alpha chosen on inner grouped OOF (leak-free)",
                "stacking_logreg": "balanced logistic meta-learner on inner grouped OOF probabilities",
                "early_fusion": "single learner on standardized concat(nn, radiomics)",
            },
            "n_samples": int(len(y)),
            "n_groups": int(len(set(groups))),
            "class_counts": {label_names[index]: int((y == index).sum()) for index in range(num_classes)},
            "script_sha256": protocol_hash,
            "feature_sources": sources,
            "synthetic_demo": bool(args.synthetic_demo),
        },
        "summaries": summaries,
        "per_seed": per_method,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(output, handle, indent=2, default=float)
    print(f"\nWrote ensemble comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
