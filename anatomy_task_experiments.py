"""Train and compare anatomically coherent DSA classification probes.

The historical ``other_positive`` target mixes M1/M4, ACA, and PCA.  This
script replaces it with explicit tasks that have a defensible interpretation:

* ``territory_strict``: MCA vs ACA vs PCA, excluding cross-territory labels.
* ``clean_m2_m3``: pure M2 vs pure M3 only.
* ``m2_m3_pca_strict``: pure M2 vs pure M3 vs PCA.
* ``supported_4class``: M2 vs M3 vs ACA vs PCA, excluding M1/M4 and composites.
* ``mca_coarse_strict``: M2 vs M3 vs other-MCA (M1/M4), as a data-sufficiency test.
* ``mca_4class_strict``: M1 vs M2 vs M3 vs M4, as a sparse-class stress test.
* ``anatomy_6class_strict``: M1/M2/M3/M4/ACA/PCA, excluding composite labels.

Every result is repeated patient-grouped out-of-fold evaluation.  Feature
standardization and the balanced logistic probe are fit inside each fold.
Optionally, the script then fits full-data probe artifacts for later inference;
those artifacts are not additional held-out evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

import metrics as M
from compare_feature_caches import canonical_metadata, file_sha256, parse_feature, parse_seeds
from experiments import make_folds


RECIPE_NAMES = ("std_logreg_c3", "std_logreg_c1", "std_linsvm", "l2_logreg")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_names: tuple[str, ...]
    indices: np.ndarray
    labels: np.ndarray
    definition: str


def parse_fusion(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Fusion must be NAME=SOURCE_A+SOURCE_B[+...].")
    name, sources_text = value.split("=", 1)
    sources = tuple(part for part in sources_text.split("+") if part)
    if not name or len(sources) < 2:
        raise argparse.ArgumentTypeError("Fusion must be NAME=SOURCE_A+SOURCE_B[+...].")
    return name, sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature", action="append", type=parse_feature, required=True, metavar="NAME=PATH")
    parser.add_argument("--fusion", action="append", type=parse_fusion, default=[], metavar="NAME=A+B")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0:20"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--recipes",
        default=",".join(RECIPE_NAMES),
        help=f"Comma-separated fixed probe recipes. Choices: {', '.join(RECIPE_NAMES)}.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/anatomy_tasks/results.json"))
    parser.add_argument(
        "--train-final",
        action="store_true",
        help="Fit one full-data probe per task using that task's best repeated-CV feature source.",
    )
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, object]:
    archive = np.load(path, allow_pickle=True)
    required = ("mean", "labels", "groups", "meta")
    missing = [name for name in required if name not in archive]
    if missing:
        raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
    result = {name: archive[name] for name in required}
    row_count = len(result["labels"])
    for name in required:
        if len(result[name]) != row_count:
            raise ValueError(f"{path} has inconsistent row counts.")
    if not np.isfinite(result["mean"]).all():
        raise ValueError(f"{path} contains non-finite mean features.")
    result["metadata_identity"] = canonical_metadata(result["meta"])
    result["signature"] = (
        json.loads(str(archive["signature_json"].item())) if "signature_json" in archive else None
    )
    result["sha256"] = file_sha256(path)
    result["path"] = str(path)
    return result


def align_cache(cache: dict[str, object], reference_identity) -> dict[str, object]:
    identities = cache["metadata_identity"]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate canonical sample identities in {cache['path']}.")
    index_by_identity = {identity: index for index, identity in enumerate(identities)}
    missing = [identity for identity in reference_identity if identity not in index_by_identity]
    extras = set(identities).difference(reference_identity)
    if missing or extras:
        raise ValueError(
            f"Metadata mismatch for {cache['path']}: {len(missing)} missing and {len(extras)} extra."
        )
    order = np.asarray([index_by_identity[identity] for identity in reference_identity], dtype=int)
    aligned = dict(cache)
    for name in ("mean", "labels", "groups", "meta"):
        aligned[name] = cache[name][order]
    aligned["metadata_identity"] = tuple(identities[index] for index in order)
    return aligned


def anatomy_codes(label_text: str) -> frozenset[str]:
    compact = re.sub(r"\s+", "", str(label_text).upper())
    return frozenset(re.findall(r"[MAP][1-4]", compact))


def make_task(
    name: str,
    label_names: tuple[str, ...],
    assigned: list[str | None],
    definition: str,
) -> TaskSpec:
    label_to_index = {label: index for index, label in enumerate(label_names)}
    indices = np.asarray([index for index, label in enumerate(assigned) if label is not None], dtype=int)
    labels = np.asarray([label_to_index[assigned[index]] for index in indices], dtype=int)
    if set(np.unique(labels)) != set(range(len(label_names))):
        raise ValueError(f"Task {name} does not contain every declared class.")
    return TaskSpec(name, label_names, indices, labels, definition)


def build_tasks(label_texts: list[str]) -> list[TaskSpec]:
    codes = [anatomy_codes(text) for text in label_texts]

    territory = []
    clean_m2_m3 = []
    m2_m3_pca = []
    supported_4class = []
    mca_coarse = []
    mca_4class = []
    anatomy_6class = []
    for sample_codes in codes:
        territories = {
            "mca" if code.startswith("M") else "aca" if code.startswith("A") else "pca"
            for code in sample_codes
        }
        territory.append(next(iter(territories)) if len(territories) == 1 else None)

        clean_m2_m3.append(
            "m2" if sample_codes == {"M2"} else "m3" if sample_codes == {"M3"} else None
        )

        if sample_codes == {"M2"}:
            m2_m3_pca.append("m2")
        elif sample_codes == {"M3"}:
            m2_m3_pca.append("m3")
        elif sample_codes and all(code.startswith("P") for code in sample_codes):
            m2_m3_pca.append("pca")
        else:
            m2_m3_pca.append(None)

        if sample_codes == {"M2"}:
            supported_4class.append("m2")
        elif sample_codes == {"M3"}:
            supported_4class.append("m3")
        elif sample_codes and all(code.startswith("A") for code in sample_codes):
            supported_4class.append("aca")
        elif sample_codes and all(code.startswith("P") for code in sample_codes):
            supported_4class.append("pca")
        else:
            supported_4class.append(None)

        if sample_codes == {"M2"}:
            mca_coarse.append("m2")
        elif sample_codes == {"M3"}:
            mca_coarse.append("m3")
        elif sample_codes in ({"M1"}, {"M4"}):
            mca_coarse.append("other_mca")
        else:
            mca_coarse.append(None)

        if len(sample_codes) == 1 and next(iter(sample_codes)).startswith("M"):
            mca_4class.append(next(iter(sample_codes)).lower())
        else:
            mca_4class.append(None)

        if len(sample_codes) == 1:
            code = next(iter(sample_codes))
            if code.startswith("M"):
                anatomy_6class.append(code.lower())
            elif code.startswith("A"):
                anatomy_6class.append("aca")
            else:
                anatomy_6class.append("pca")
        else:
            anatomy_6class.append(None)

    return [
        make_task(
            "territory_strict",
            ("mca", "aca", "pca"),
            territory,
            "MCA vs ACA vs PCA; exclude labels spanning more than one territory.",
        ),
        make_task(
            "clean_m2_m3",
            ("m2", "m3"),
            clean_m2_m3,
            "Pure M2 vs pure M3; exclude every composite and non-M2/M3 label.",
        ),
        make_task(
            "m2_m3_pca_strict",
            ("m2", "m3", "pca"),
            m2_m3_pca,
            "Pure M2 vs pure M3 vs PCA; exclude ACA, M1/M4, and every composite label.",
        ),
        make_task(
            "supported_4class",
            ("m2", "m3", "aca", "pca"),
            supported_4class,
            "M2 vs M3 vs ACA vs PCA; exclude M1/M4 and cross/within-MCA composites.",
        ),
        make_task(
            "mca_coarse_strict",
            ("m2", "m3", "other_mca"),
            mca_coarse,
            "Pure M2 vs pure M3 vs pure M1/M4 grouped as other-MCA; diagnostic only.",
        ),
        make_task(
            "mca_4class_strict",
            ("m1", "m2", "m3", "m4"),
            mca_4class,
            "Pure single-segment MCA labels M1/M2/M3/M4; sparse-class stress test.",
        ),
        make_task(
            "anatomy_6class_strict",
            ("m1", "m2", "m3", "m4", "aca", "pca"),
            anatomy_6class,
            "Pure M1/M2/M3/M4 plus single-territory ACA/PCA; exclude composites.",
        ),
    ]


def parse_recipes(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names).difference(RECIPE_NAMES))
    if unknown:
        raise ValueError(f"Unknown recipes: {unknown}. Choices: {RECIPE_NAMES}")
    if not names:
        raise ValueError("At least one recipe is required.")
    return names


def new_probe(recipe: str):
    if recipe == "std_logreg_c3":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=3.0, class_weight="balanced"),
        )
    if recipe == "std_logreg_c1":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"),
        )
    if recipe == "std_linsvm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.5, class_weight="balanced", max_iter=20000, random_state=0),
        )
    if recipe == "l2_logreg":
        return make_pipeline(
            Normalizer(),
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"),
        )
    raise ValueError(recipe)


def prediction_scores(probe, features: np.ndarray, num_classes: int) -> np.ndarray:
    if hasattr(probe, "predict_proba"):
        return probe.predict_proba(features)
    decision = np.asarray(probe.decision_function(features), dtype=float)
    if decision.ndim == 1:
        decision = np.column_stack((-decision, decision))
    if decision.shape[1] != num_classes:
        raise ValueError(f"Expected {num_classes} score columns, got {decision.shape}.")
    return decision


def macro_auprc(y_true: np.ndarray, probabilities: np.ndarray, num_classes: int) -> float:
    return float(
        np.mean(
            [
                average_precision_score((y_true == class_index).astype(int), probabilities[:, class_index])
                for class_index in range(num_classes)
            ]
        )
    )


def task_fold_count(task: TaskSpec, groups: np.ndarray, requested: int) -> int:
    task_groups = groups[task.indices]
    groups_per_class = [
        len(set(task_groups[task.labels == class_index]))
        for class_index in range(len(task.label_names))
    ]
    return min(requested, *groups_per_class)


def evaluate_one(
    task: TaskSpec,
    feature_name: str,
    features: np.ndarray,
    recipe: str,
    groups: np.ndarray,
    requested_folds: int,
    seed: int,
) -> tuple[str, str, str, dict[str, object]]:
    x = features[task.indices]
    y = task.labels
    task_groups = groups[task.indices]
    fold_count = task_fold_count(task, groups, requested_folds)
    folds = make_folds(y, task_groups, fold_count, seed)
    predictions = np.full(len(y), -1, dtype=int)
    probabilities = np.zeros((len(y), len(task.label_names)), dtype=float)
    for train, valid in folds:
        probe = new_probe(recipe)
        probe.fit(x[train], y[train])
        predictions[valid] = probe.predict(x[valid])
        probabilities[valid] = prediction_scores(probe, x[valid], len(task.label_names))
    metrics = M.compute_metrics(y, predictions, len(task.label_names))
    return task.name, feature_name, recipe, {
        "seed": seed,
        "folds": fold_count,
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "macro_auprc": macro_auprc(y, probabilities, len(task.label_names)),
        "per_class_precision": metrics["per_class"]["precision"],
        "per_class_recall": metrics["per_class"]["recall"],
        "per_class_f1": metrics["per_class"]["f1"],
        "confusion": metrics["confusion"],
    }


def mean_std(rows: list[dict[str, object]], key: str) -> tuple[object, object]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    return (
        mean.tolist() if mean.ndim else float(mean),
        std.tolist() if std.ndim else float(std),
    )


def summarize(tasks, feature_names, recipes, per_seed):
    summaries: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        task_rows = []
        for feature_name in feature_names:
            for recipe in recipes:
                rows = per_seed[task.name][feature_name][recipe]
                macro_mean, macro_std = mean_std(rows, "macro_f1")
                bal_mean, bal_std = mean_std(rows, "balanced_accuracy")
                auprc_mean, auprc_std = mean_std(rows, "macro_auprc")
                precision_mean, precision_std = mean_std(rows, "per_class_precision")
                recall_mean, recall_std = mean_std(rows, "per_class_recall")
                f1_mean, f1_std = mean_std(rows, "per_class_f1")
                task_rows.append(
                    {
                        "feature": feature_name,
                        "recipe": recipe,
                        "seed0_macro_f1": rows[0]["macro_f1"] if rows[0]["seed"] == 0 else None,
                        "macro_f1_mean": macro_mean,
                        "macro_f1_std": macro_std,
                        "balanced_accuracy_mean": bal_mean,
                        "balanced_accuracy_std": bal_std,
                        "macro_auprc_mean": auprc_mean,
                        "macro_auprc_std": auprc_std,
                        "per_class_precision_mean": precision_mean,
                        "per_class_precision_std": precision_std,
                        "per_class_recall_mean": recall_mean,
                        "per_class_recall_std": recall_std,
                        "per_class_f1_mean": f1_mean,
                        "per_class_f1_std": f1_std,
                    }
                )
        summaries[task.name] = sorted(task_rows, key=lambda row: row["macro_f1_mean"], reverse=True)
    return summaries


def train_final_models(
    out_dir: Path,
    tasks: list[TaskSpec],
    features: dict[str, np.ndarray],
    summaries: dict[str, list[dict[str, object]]],
    groups: np.ndarray,
    metadata_identity,
) -> dict[str, object]:
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    task_by_name = {task.name: task for task in tasks}
    for task_name, task_summaries in summaries.items():
        task = task_by_name[task_name]
        feature_name = task_summaries[0]["feature"]
        recipe = task_summaries[0]["recipe"]
        probe = new_probe(recipe)
        probe.fit(features[feature_name][task.indices], task.labels)
        path = model_dir / f"{task_name}__{feature_name}__{recipe}.joblib"
        artifact = {
            "pipeline": probe,
            "task": task.name,
            "definition": task.definition,
            "label_names": list(task.label_names),
            "feature_source": feature_name,
            "recipe": recipe,
            "feature_dimension": int(features[feature_name].shape[1]),
            "training_indices": task.indices,
            "training_groups": groups[task.indices],
            "training_metadata_identity": [metadata_identity[index] for index in task.indices],
            "selection_note": "Feature source and fixed probe recipe selected by mean macro-F1 across repeated grouped CV; full-data fit is not held-out evidence.",
        }
        joblib.dump(artifact, path)
        artifacts[task_name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "feature_source": feature_name,
            "recipe": recipe,
            "labels": list(task.label_names),
            "n_samples": len(task.labels),
            "n_groups": len(set(groups[task.indices])),
        }

    cascade = {
        "territory_model": artifacts["territory_strict"],
        "mca_segment_model": artifacts["clean_m2_m3"],
        "policy": {
            "aca": "return ACA",
            "pca": "return PCA",
            "mca": "run clean M2-vs-M3 probe only when the case is known to be within its supported M2/M3 scope",
            "m1_m4": "unsupported/abstain; the available data do not justify an M1/M4 detector",
        },
        "warning": "Component scores are not calibrated for clinical decision-making.",
    }
    cascade_path = model_dir / "territory_then_m2_m3_cascade.json"
    with cascade_path.open("w") as handle:
        json.dump(cascade, handle, indent=2)
    artifacts["territory_then_m2_m3_cascade"] = {
        "path": str(cascade_path),
        "sha256": file_sha256(cascade_path),
    }
    return artifacts


def main() -> int:
    args = parse_args()
    recipes = parse_recipes(args.recipes)
    feature_paths = dict(args.feature)
    if len(feature_paths) != len(args.feature):
        raise ValueError("Every --feature name must be unique.")
    loaded = {name: load_cache(path) for name, path in feature_paths.items()}
    first_name = next(iter(loaded))
    reference_identity = loaded[first_name]["metadata_identity"]
    loaded = {name: align_cache(cache, reference_identity) for name, cache in loaded.items()}
    reference_groups = np.asarray(loaded[first_name]["groups"], dtype=object)
    for name, cache in loaded.items():
        if cache["metadata_identity"] != reference_identity:
            raise ValueError(f"Metadata alignment failed for {name}.")
        if not np.array_equal(cache["groups"], reference_groups):
            raise ValueError(f"Patient groups do not match for {name}.")

    features = {name: np.asarray(cache["mean"], dtype=np.float32) for name, cache in loaded.items()}
    for fusion_name, source_names in args.fusion:
        if fusion_name in features:
            raise ValueError(f"Duplicate feature/fusion name: {fusion_name}")
        missing = [name for name in source_names if name not in features]
        if missing:
            raise ValueError(f"Fusion {fusion_name} references unknown sources: {missing}")
        features[fusion_name] = np.concatenate([features[name] for name in source_names], axis=1)

    label_texts = [identity[4] for identity in reference_identity]
    tasks = build_tasks(label_texts)
    per_seed = {
        task.name: {
            feature_name: {recipe: [] for recipe in recipes}
            for feature_name in features
        }
        for task in tasks
    }
    jobs = [
        delayed(evaluate_one)(
            task, feature_name, matrix, recipe, reference_groups, args.folds, seed
        )
        for task in tasks
        for feature_name, matrix in features.items()
        for recipe in recipes
        for seed in args.seeds
    ]
    evaluated = Parallel(n_jobs=args.jobs, verbose=5)(jobs)
    for task_name, feature_name, recipe, row in evaluated:
        per_seed[task_name][feature_name][recipe].append(row)
    for task_results in per_seed.values():
        for feature_results in task_results.values():
            for rows in feature_results.values():
                rows.sort(key=lambda row: row["seed"])

    summaries = summarize(tasks, list(features), recipes, per_seed)
    task_metadata = {}
    for task in tasks:
        counts = {
            label: int(np.sum(task.labels == index))
            for index, label in enumerate(task.label_names)
        }
        task_metadata[task.name] = {
            "definition": task.definition,
            "labels": list(task.label_names),
            "class_counts": counts,
            "n_samples": len(task.labels),
            "n_groups": len(set(reference_groups[task.indices])),
            "folds": task_fold_count(task, reference_groups, args.folds),
            "excluded_samples": len(reference_groups) - len(task.labels),
        }
        print(f"\n{task.name}: {task.definition}")
        print("classes: " + "  ".join(f"{name}={counts[name]}" for name in task.label_names))
        print(f"{'feature':22} {'recipe':16} {'macroF1':12} {'class F1'}")
        print("-" * 86)
        for row in summaries[task.name]:
            class_f1 = "/".join(f"{value:.3f}" for value in row["per_class_f1_mean"])
            print(
                f"{row['feature'][:22]:22} {row['recipe'][:16]:16} "
                f"{row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f}  {class_f1}"
            )

    output_dir = args.out.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (
        train_final_models(
            output_dir,
            tasks,
            features,
            summaries,
            reference_groups,
            reference_identity,
        )
        if args.train_final
        else {}
    )
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = {
        "protocol": {
            "seeds": args.seeds,
            "requested_folds": args.folds,
            "group": "Study_Key",
            "classifier": "predeclared fold-local probe recipes; see recipes field",
            "recipes": recipes,
            "representation": "mean pooled frozen features",
            "n_source_runs": len(reference_groups),
            "n_source_groups": len(set(reference_groups)),
            "script_sha256": script_hash,
            "feature_sources": {
                name: {
                    "path": cache["path"],
                    "sha256": cache["sha256"],
                    "signature": cache["signature"],
                }
                for name, cache in loaded.items()
            },
            "fusions": {name: list(sources) for name, sources in args.fusion},
        },
        "tasks": task_metadata,
        "summaries": summaries,
        "per_seed": per_seed,
        "final_model_artifacts": artifacts,
    }
    with args.out.open("w") as handle:
        json.dump(output, handle, indent=2, default=float)
    print(f"\nWrote anatomy-task results to {args.out}")
    if artifacts:
        print(f"Wrote {len(artifacts) - 1} trained probes and one cascade manifest to {output_dir / 'models'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
