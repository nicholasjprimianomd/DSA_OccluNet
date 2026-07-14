"""Compare cached backbone features on identical repeated patient-grouped folds.

The comparison deliberately fixes one representation and one classifier recipe
for every backbone.  This avoids declaring a winner merely because a different
recipe happened to win each model's sweep on the same validation folds.

Example:
    python compare_feature_caches.py \
      --feature vjepa_l256=runs/ap_exp/cache/rich_AP_positive_subtype.npz \
      --feature raddino_518=runs/ap_raddino518/cache/image_microsoft-rad-dino_518_16_AP_positive_subtype.npz \
      --seeds 0:20 --out runs/backbone_comparison.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from joblib import Parallel, delayed

from experiments import (
    LINEAR_SVM_DUAL,
    LINEAR_SVM_MAX_ITER,
    LINEAR_SVM_TOL,
    evaluate_recipe,
    grouped_fold_count,
    make_folds,
    representation,
)


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = value.split(":", 1)
        seeds = list(range(int(start), int(stop)))
    else:
        seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one split seed is required.")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Split seeds must be unique.")
    return sorted(seeds)


def parse_feature(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Feature must be NAME=PATH.")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Feature must be NAME=PATH.")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature", action="append", type=parse_feature, required=True, metavar="NAME=PATH")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0:20"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--representation", choices=("mean", "meanmaxstd"), default="mean")
    parser.add_argument("--preprocessing", choices=("raw", "std", "l2"), default="std")
    parser.add_argument(
        "--classifier",
        choices=("logreg", "logreg_C0.3", "logreg_C3", "logreg_noW", "svm_rbf", "linsvm", "mlp"),
        default="linsvm",
    )
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel workers (-1 uses all CPUs).")
    parser.add_argument(
        "--allow-legacy-metadata",
        action="store_true",
        help="Explicitly allow a cache whose metadata has no label_text (for the historical "
        "RadImageNet cache). Labels, groups, and sample identities are still validated.",
    )
    parser.add_argument("--out", default="runs/backbone_comparison.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_metadata(meta) -> tuple[tuple[str, ...], ...]:
    keys = ("study_key", "accession", "view", "run_column", "label_text")
    return tuple(tuple(str(row.get(key, "")) for key in keys) for row in meta)


def canonical_sample_identity(meta) -> tuple[tuple[str, ...], ...]:
    """Identity shared by modern caches and the older RadImageNet cache."""
    keys = ("study_key", "accession", "view", "run_column")
    return tuple(tuple(str(row.get(key, "")) for key in keys) for row in meta)


def load_cache(path: Path, representation_name: str = "mean") -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    feature_arrays = ("mean",) if representation_name == "mean" else ("mean", "max", "std")
    required = (*feature_arrays, "labels", "groups", "meta")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
    loaded = {key: data[key] for key in required}
    n_rows = len(loaded["labels"])
    for key in (*feature_arrays, "groups", "meta"):
        if len(loaded[key]) != n_rows:
            raise ValueError(f"{path} has {len(loaded[key])} {key} rows but {n_rows} labels.")
    for key in feature_arrays:
        if not np.isfinite(loaded[key]).all():
            raise ValueError(f"{path} contains non-finite {key} features.")
    loaded["metadata_identity"] = canonical_metadata(loaded["meta"])
    loaded["sample_identity"] = canonical_sample_identity(loaded["meta"])
    if len(set(loaded["sample_identity"])) != len(loaded["sample_identity"]):
        raise ValueError(f"{path} contains duplicate sample identities.")
    label_text_key_presence = ["label_text" in row for row in loaded["meta"]]
    if any(label_text_key_presence) and not all(label_text_key_presence):
        raise ValueError(f"{path} has partial label_text metadata; expected all rows or none.")
    loaded["has_label_text_metadata"] = all(label_text_key_presence)
    loaded["cache_sha256"] = file_sha256(path)
    loaded["signature"] = (
        json.loads(str(data["signature_json"].item())) if "signature_json" in data else None
    )
    return loaded


def evaluate_one(
    name: str,
    seed: int,
    features: np.ndarray,
    labels: np.ndarray,
    folds,
    num_classes: int,
    preprocessing: str,
    classifier: str,
) -> tuple[str, dict[str, object]]:
    try:
        _, metrics = evaluate_recipe(
            features,
            labels,
            folds,
            num_classes,
            preprocessing,
            classifier,
        )
    except RuntimeError as error:
        raise RuntimeError(f"model={name}, seed={seed}: {error}") from error
    return name, {
        "seed": seed,
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "per_class_f1": metrics["per_class"]["f1"],
    }


def score_for_seed(rows: list[dict[str, object]], seed: int) -> float | None:
    return next((float(row["macro_f1"]) for row in rows if row["seed"] == seed), None)


def main() -> int:
    args = parse_args()
    if len(args.feature) < 2:
        raise SystemExit("Provide at least two --feature NAME=PATH entries.")
    names = [name for name, _ in args.feature]
    if len(set(names)) != len(names):
        raise SystemExit("Every --feature name must be unique.")

    caches = [
        (name, path, load_cache(path, args.representation)) for name, path in args.feature
    ]
    legacy_metadata_caches = [
        name for name, _, data in caches if not data["has_label_text_metadata"]
    ]
    if legacy_metadata_caches and not args.allow_legacy_metadata:
        raise ValueError(
            "Caches without label_text metadata require --allow-legacy-metadata: "
            + ", ".join(legacy_metadata_caches)
        )
    reference_y = caches[0][2]["labels"]
    reference_groups = caches[0][2]["groups"]
    reference_sample_identity = caches[0][2]["sample_identity"]
    modern_metadata_reference = next(
        (
            data["metadata_identity"]
            for _, _, data in caches
            if data["has_label_text_metadata"]
        ),
        None,
    )
    for name, path, data in caches[1:]:
        if not np.array_equal(data["labels"], reference_y):
            raise ValueError(f"Labels in {path} ({name}) do not match the reference cache.")
        if not np.array_equal(data["groups"], reference_groups):
            raise ValueError(f"Groups in {path} ({name}) do not match the reference cache.")
        if data["sample_identity"] != reference_sample_identity:
            raise ValueError(f"Sample metadata/order in {path} ({name}) does not match the reference cache.")
        if (
            data["has_label_text_metadata"]
            and modern_metadata_reference is not None
            and data["metadata_identity"] != modern_metadata_reference
        ):
            raise ValueError(f"Label metadata in {path} ({name}) does not match the reference cache.")

    num_classes = int(reference_y.max()) + 1
    n_splits = grouped_fold_count(
        reference_y,
        reference_groups,
        args.folds,
        num_classes,
    )
    per_model: dict[str, list[dict[str, object]]] = {name: [] for name, _, _ in caches}
    feature_matrices = {
        name: representation(data, args.representation) for name, _, data in caches
    }
    tasks = []
    for seed in args.seeds:
        folds = make_folds(reference_y, reference_groups, n_splits, seed)
        for name, _, _ in caches:
            tasks.append(
                delayed(evaluate_one)(
                    name,
                    seed,
                    feature_matrices[name],
                    reference_y,
                    folds,
                    num_classes,
                    args.preprocessing,
                    args.classifier,
                )
            )
    evaluated = Parallel(n_jobs=args.jobs, verbose=5)(tasks)
    for name, row in evaluated:
        per_model[name].append(row)
    for rows in per_model.values():
        rows.sort(key=lambda row: row["seed"])

    summaries = []
    reference_scores = np.asarray([row["macro_f1"] for row in per_model[caches[0][0]]], dtype=float)
    for name, path, _ in caches:
        rows = per_model[name]
        scores = np.asarray([row["macro_f1"] for row in rows], dtype=float)
        class_scores = np.asarray([row["per_class_f1"] for row in rows], dtype=float)
        delta = scores - reference_scores
        summaries.append(
            {
                "name": name,
                "path": str(path),
                "seed0_macro_f1": score_for_seed(rows, 0),
                "macro_f1_mean": float(scores.mean()),
                "macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "macro_f1_median": float(np.median(scores)),
                "macro_f1_min": float(scores.min()),
                "macro_f1_max": float(scores.max()),
                "delta_vs_reference_mean": float(delta.mean()),
                "delta_vs_reference_std": (
                    float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
                ),
                "delta_vs_reference_median": float(np.median(delta)),
                "delta_vs_reference_min": float(delta.min()),
                "delta_vs_reference_max": float(delta.max()),
                "wins_vs_reference": int((delta > 0).sum()),
                "ties_vs_reference": int((delta == 0).sum()),
                "losses_vs_reference": int((delta < 0).sum()),
                "per_class_f1_mean": class_scores.mean(axis=0).tolist(),
                "per_class_f1_std": class_scores.std(axis=0, ddof=1).tolist() if len(scores) > 1 else [0.0] * num_classes,
            }
        )
    summaries.sort(key=lambda row: row["macro_f1_mean"], reverse=True)

    widths = (18, 8, 12, 9, 9, 9, 12, 9)
    header = ("model", "seed0", "mean±sd", "median", "min", "max", "delta_ref", "wins")
    print("  ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for row in summaries:
        values = (
            row["name"][: widths[0]],
            f"{row['seed0_macro_f1']:.3f}" if row["seed0_macro_f1"] is not None else "n/a",
            f"{row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f}",
            f"{row['macro_f1_median']:.3f}",
            f"{row['macro_f1_min']:.3f}",
            f"{row['macro_f1_max']:.3f}",
            f"{row['delta_vs_reference_mean']:+.3f}",
            f"{row['wins_vs_reference']}/{len(args.seeds)}",
        )
        print("  ".join(str(value).ljust(width) for value, width in zip(values, widths)))

    output = {
        "protocol": {
            "seeds": args.seeds,
            "folds": n_splits,
            "representation": args.representation,
            "preprocessing": args.preprocessing,
            "classifier": args.classifier,
            "linear_svm_max_iterations": (
                LINEAR_SVM_MAX_ITER if args.classifier == "linsvm" else None
            ),
            "linear_svm_dual": (
                LINEAR_SVM_DUAL if args.classifier == "linsvm" else None
            ),
            "linear_svm_tolerance": (
                LINEAR_SVM_TOL if args.classifier == "linsvm" else None
            ),
            "convergence_policy": (
                "fail the run on any scikit-learn ConvergenceWarning"
                if args.classifier == "linsvm"
                else None
            ),
            "reference": caches[0][0],
            "n_samples": len(reference_y),
            "n_groups": len(set(reference_groups)),
            "legacy_metadata_opt_in": args.allow_legacy_metadata,
            "legacy_metadata_caches": legacy_metadata_caches,
            "script_sha256": file_sha256(Path(__file__)),
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "feature_sources": {
                name: {
                    "path": str(path),
                    "cache_sha256": data["cache_sha256"],
                    "signature": data["signature"],
                    "metadata_validation": (
                        "sample identity, labels, groups, and label text"
                        if data["has_label_text_metadata"]
                        else "sample identity, labels, and groups (legacy cache has no label_text)"
                    ),
                }
                for name, path, data in caches
            },
        },
        "summaries": summaries,
        "per_seed": per_model,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        json.dump(output, handle, indent=2, default=float)
    print(f"\nWrote repeated-fold comparison to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
