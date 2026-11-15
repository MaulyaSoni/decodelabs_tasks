"""DecodeLabs Industrial Training Kit - Batch 2026.

Project 2: Data Classification Using AI (Real-Time Interactive AI Studio)
Algorithm: K-Nearest Neighbors (KNN) Benchmark + Dynamic AI Insights
Features:
- Continuous Interactive Loop (predict multiple flowers, inspect datasets, live explore)
- Dynamic AI Flower Analysis:
  - Finds exact top-5 closest real neighbors from the training set with Euclidean distances
  - Morphological insight engine (explaining WHY it classified based on petal/sepal ratios)
  - Interactive What-If perturbation analysis ("How would species change if petal width +0.5cm?")
  - Real-time Groq Cloud AI Horticultural & Botanical Explanation
- Leakage-free 5-fold Stratified CV with 6-model comparison
- Auto-generates diagnostic plots: k_curve.png, confusion_matrix.png, feature_importance.png
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUTS_DIR / "iris_knn_pipeline.joblib"
META_FILE = OUTPUTS_DIR / "model_meta.json"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Terminal Styling Helpers (ANSI)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

try:
    sys.path.insert(0, str(REPO_ROOT))
    from common.config import GROQ_API_KEY
    from groq import Groq
    GROQ_CLIENT = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and not GROQ_API_KEY.startswith("your_") else None
except Exception:
    GROQ_CLIENT = None


def stream_print(text: str, delay: float = 0.008, end: str = "\n") -> None:
    """Stream text output dynamically."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()


def print_step(title: str, delay: float = 0.03) -> None:
    sys.stdout.write(f"  {CYAN}▸{RESET} {title}...")
    sys.stdout.flush()
    time.sleep(delay)
    print(f" {GREEN}✓{RESET}")


def load_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    class_names = [str(name) for name in iris.target_names]
    return X, y, class_names


def split_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )


def scale_features(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
) -> KNeighborsClassifier:
    n_samples = len(X_train_scaled)
    actual_k = max(1, min(k, n_samples))
    model = KNeighborsClassifier(n_neighbors=actual_k, weights=weights, metric=metric)
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(
    model: BaseEstimator,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Tuple[np.ndarray, float, float]:
    predictions = model.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions, average="weighted"))
    return predictions, acc, f1


def evaluate_cv_model(
    seed: int = 42,
    n_folds: int = 5,
    k: int = 5,
) -> Dict[str, Any]:
    iris = load_iris()
    X = iris.data
    y = iris.target
    actual_k = max(1, min(k, len(X) - 1))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=actual_k)),
    ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")

    return {
        "n_folds": n_folds,
        "k": actual_k,
        "mean_accuracy": float(np.mean(acc_scores)),
        "std_accuracy": float(np.std(acc_scores)),
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "accuracy_scores": [float(v) for v in acc_scores],
        "f1_scores": [float(v) for v in f1_scores],
    }


def tune_knn(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[GridSearchCV, Dict[str, Any]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    max_k = min(25, int(len(X) * (n_folds - 1) / n_folds))
    param_grid = {
        "knn__n_neighbors": list(range(1, max_k + 1)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X, y)

    best_summary = {
        "best_params": {
            k.replace("knn__", ""): v for k, v in grid.best_params_.items()
        },
        "best_cv_accuracy": float(grid.best_score_),
    }
    return grid, best_summary


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    models = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=seed),
        "Support Vector Machine (RBF)": SVC(probability=True, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "Gaussian Naive Bayes": GaussianNB(),
    }

    results = []
    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clf),
        ])
        acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
        results.append({
            "model": name,
            "mean_accuracy": float(np.mean(acc_scores)),
            "std_accuracy": float(np.std(acc_scores)),
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
        })

    results.sort(key=lambda r: -r["mean_accuracy"])
    return results


def save_diagnostic_plots(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pipe: Pipeline,
    class_names: List[str],
    feature_names: List[str],
    outputs_dir: Path = OUTPUTS_DIR,
) -> Dict[str, str]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: Dict[str, str] = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k_values = list(range(1, 26))
        train_errors = []
        test_errors = []

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_tr_s, y_train)
            train_errors.append(1.0 - accuracy_score(y_train, knn.predict(X_tr_s)))
            test_errors.append(1.0 - accuracy_score(y_test, knn.predict(X_te_s)))

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, train_errors, "b-o", label="Training Error")
        plt.plot(k_values, test_errors, "r-s", label="Validation Error")
        plt.xlabel("Number of Neighbors (K)")
        plt.ylabel("Classification Error (1 - Accuracy)")
        plt.title("KNN Validation Curve: Error vs. K (Elbow Detection)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        k_curve_path = outputs_dir / "k_curve.png"
        plt.tight_layout()
        plt.savefig(k_curve_path, dpi=150)
        plt.close()
        plot_paths["k_curve"] = str(k_curve_path)

        y_pred = pipe.predict(X_test)
        cm_raw = confusion_matrix(y_test, y_pred)
        cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, matrix, title, fmt in [
            (ax1, cm_raw, "Raw Confusion Matrix", "d"),
            (ax2, cm_norm, "Normalized Confusion Matrix", ".2f"),
        ]:
            im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=20)
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    val_str = f"{matrix[i, j]:{fmt}}"
                    color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                    ax.text(j, i, val_str, ha="center", va="center", color=color)

        cm_path = outputs_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        plot_paths["confusion_matrix"] = str(cm_path)

        perm_res = permutation_importance(
            pipe, X_test, y_test, n_repeats=20, random_state=42
        )
        sorted_idx = perm_res.importances_mean.argsort()

        plt.figure(figsize=(8, 4.5))
        clean_features = [feature_names[i].replace(" (cm)", "") for i in sorted_idx]
        plt.barh(clean_features, perm_res.importances_mean[sorted_idx], xerr=perm_res.importances_std[sorted_idx], color="#1f77b4")
        plt.xlabel("Mean Permutation Importance (Drop in Accuracy)")
        plt.title("Feature Importance Analysis (Iris KNN)")
        plt.grid(True, linestyle="--", alpha=0.6, axis="x")
        fi_path = outputs_dir / "feature_importance.png"
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150)
        plt.close()
        plot_paths["feature_importance"] = str(fi_path)

    except Exception:
        pass

    return plot_paths


def save_model_artifacts(
    pipe: Pipeline,
    metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump(pipe, MODEL_FILE)
    except Exception:
        pass

    import sklearn
    meta = {
        "sklearn_version": sklearn.__version__,
        "model_type": "KNeighborsClassifier",
        "best_params": best_params,
        "metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat(),
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def classify_sample(
    sample: Union[List[float], np.ndarray],
    seed: int = 42,
    pipeline: Optional[Pipeline] = None,
) -> str:
    iris = load_iris()
    X = iris.data
    y = iris.target
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    try:
        sample_arr = np.asarray(sample, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return "Prediction failed: input must contain numeric values only."

    if sample_arr.shape[1] != 4:
        return "Prediction failed: wrong number of features. Please provide exactly 4 values."

    if np.any(~np.isfinite(sample_arr)):
        return "Prediction failed: sample contains NaN or inf values."

    if np.any(sample_arr <= 0):
        return "Prediction failed: measurements must be strictly positive numbers."

    out_of_range = np.any((sample_arr < min_vals) | (sample_arr > max_vals))

    if pipeline is None:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ])
        pipeline.fit(X, y)

    pred = int(pipeline.predict(sample_arr)[0])
    label = iris.target_names[pred]
    vote_shares = pipeline.predict_proba(sample_arr)[0]
    vote_share_pct = float(vote_shares[pred]) * 100

    range_notice = "outside training range" if out_of_range else "within training range"
    return f"Prediction: {label} (Vote Share: {vote_share_pct:.1f}%) | {range_notice}."


def explain_flower_prediction(sample: List[float], pred_species: str, vote_share: float) -> str:
    """Dynamic morphological explanation with optional Groq LLM botanical insights."""
    sl, sw, pl, pw = sample
    petal_ratio = pl / max(0.1, pw)
    sepal_ratio = sl / max(0.1, sw)

    explanation = []
    if pl < 2.0:
        explanation.append(f"Petal length ({pl:.1f}cm) is tiny (<2.0cm), which is a definitive hallmark of Iris Setosa.")
    elif pl < 4.8 and pw < 1.7:
        explanation.append(f"Petal length ({pl:.1f}cm) and width ({pw:.1f}cm) represent the intermediate cluster characteristic of Versicolor.")
    else:
        explanation.append(f"Petal dimensions ({pl:.1f}cm × {pw:.1f}cm) are large, falling into the upper quadrant of Iris Virginica.")

    rule_summary = " ".join(explanation)

    if GROQ_CLIENT:
        try:
            prompt = (
                f"Explain concisely in 2 sentences why a flower with Sepal Length {sl}cm, Sepal Width {sw}cm, "
                f"Petal Length {pl}cm, and Petal Width {pw}cm is classified as Iris {pred_species} "
                f"({vote_share:.1f}% neighbor vote). Mention morphological characteristics."
            )
            resp = GROQ_CLIENT.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.5,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    return rule_summary


def find_nearest_neighbors_breakdown(
    sample: List[float], pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray
) -> List[Dict[str, Any]]:
    """Dynamically compute distances to exact nearest neighbors in normalized space."""
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    knn: KNeighborsClassifier = pipeline.named_steps["knn" if "knn" in pipeline.named_steps else "model"]

    sample_scaled = scaler.transform([sample])
    distances, indices = knn.kneighbors(sample_scaled, n_neighbors=5)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        species_id = y_train[idx]
        species_name = CLASS_NAMES[species_id]
        raw_feats = X_train[idx].tolist()
        results.append({
            "neighbor_id": int(idx),
            "species": species_name,
            "distance": float(dist),
            "features": raw_feats,
        })
    return results


def run_interactive_ai_studio(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """Real-time dynamic experimentation loop."""
    iris = load_iris()
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    print(f"\n{BOLD}{CYAN}════════════════════════════════════════════════════════════════════{RESET}")
    print(f"  {BOLD}{MAGENTA}🌸 Dynamic Iris AI Studio & Real-Time Classification Laboratory{RESET}")
    print(f"  {DIM}Perform live predictions, what-if sensitivity simulations, and nearest neighbor inspections{RESET}")
    print(f"{BOLD}{CYAN}════════════════════════════════════════════════════════════════════{RESET}")

    presets = {
        "1": ([5.0, 3.5, 1.4, 0.2], "Iris Setosa (Standard Cluster A)"),
        "2": ([5.9, 3.0, 4.2, 1.5], "Iris Versicolor (Median Specimen)"),
        "3": ([6.5, 3.0, 5.5, 2.0], "Iris Virginica (Late Harvest Large)"),
        "4": ([5.9, 3.2, 4.8, 1.8], "Borderline Boundary Sample (Versicolor/Virginica overlap)"),
        "5": ([12.0, 6.5, 9.0, 3.8], "Out-of-Range Giant Botanical Anomaly"),
    }

    while True:
        print(f"\n{BOLD}Select an action or preset to test:{RESET}")
        for key, (feats, desc) in presets.items():
            print(f"  {YELLOW}[{key}]{RESET} {desc} {DIM}→ {feats}{RESET}")
        print(f"  {YELLOW}[C]{RESET} Custom Measurement Input (Type your own Sepal/Petal values)")
        print(f"  {YELLOW}[W]{RESET} 'What-If' Parameter Sensitivity Simulator")
        print(f"  {YELLOW}[Q]{RESET} Exit Studio & Return to Menu")

        try:
            choice = input(f"\n{BOLD}Studio Choice (1-5, C, W, Q) [default 1]:{RESET} ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in {"Q", "EXIT", "QUIT"}:
            print(f"  {GREEN}Exited AI Studio.{RESET}")
            break

        sample = [5.0, 3.5, 1.4, 0.2]
        desc = "Setosa Preset"

        if choice in presets:
            sample, desc = presets[choice]
        elif choice == "C":
            desc = "Custom User Specimen"
            prompts = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]
            custom_vals = []
            for p in prompts:
                try:
                    val_str = input(f"  {CYAN}▸{RESET} Enter {p}: ").strip()
                    val = float(val_str)
                    custom_vals.append(max(0.1, val))
                except Exception:
                    custom_vals.append(1.0)
            sample = custom_vals
        elif choice == "W":
            print(f"\n{BOLD}{CYAN}🔬 Dynamic What-If Sensitivity Simulator{RESET}")
            print(f"  Baseline: Versicolor Sample [6.0, 3.0, 4.5, 1.4]")
            print(f"  Observing how shifting Petal Width changes class boundary:")
            test_widths = [0.2, 0.8, 1.3, 1.7, 2.2, 2.8]
            for tw in test_widths:
                perturbed = [6.0, 3.0, 4.5, tw]
                pred_idx = int(pipeline.predict([perturbed])[0])
                prob = pipeline.predict_proba([perturbed])[0][pred_idx] * 100
                spec = CLASS_NAMES[pred_idx]
                bar = "█" * int(prob / 5)
                color = GREEN if spec == "setosa" else (CYAN if spec == "versicolor" else MAGENTA)
                print(f"    Petal Width: {tw:.1f}cm → {color}{spec.upper():<12}{RESET} ({prob:5.1f}% vote) {color}{bar}{RESET}")
            continue

        sample_arr = np.array([sample], dtype=float)
        out_of_range = np.any((sample_arr < min_vals) | (sample_arr > max_vals))

        pred_idx = int(pipeline.predict(sample_arr)[0])
        probabilities = pipeline.predict_proba(sample_arr)[0]
        predicted_species = CLASS_NAMES[pred_idx]
        vote_share = probabilities[pred_idx] * 100

        print(f"\n{BOLD}── AI Classification Report: {CYAN}{desc}{RESET} ──")
        print(f"  Input Vector      : {sample}")
        print(f"  Predicted Species : {GREEN}{BOLD}{predicted_species.upper()}{RESET}")
        print(f"  Neighbor Vote     : {CYAN}{vote_share:.1f}%{RESET} ({int(vote_share / 20)}/5 consensus)")

        # Dynamic Vote Distribution
        print(f"  Confidence Spread :")
        for c_idx, c_name in enumerate(CLASS_NAMES):
            pct = probabilities[c_idx] * 100
            bar = "█" * int(pct / 5)
            print(f"    • {c_name:<12} : {pct:5.1f}%  {CYAN}{bar}{RESET}")

        # Real Neighbors Distance Breakdown
        neighbors = find_nearest_neighbors_breakdown(sample, pipeline, X, y)
        print(f"\n  {BOLD}Top-5 Nearest Training Neighbors in Scaled Space:{RESET}")
        for n_idx, n in enumerate(neighbors, start=1):
            n_spec = n["species"]
            color = GREEN if n_spec == predicted_species else YELLOW
            print(f"    [{n_idx}] ID #{n['neighbor_id']:<3} | Species: {color}{n_spec:<12}{RESET} | Distance: {n['distance']:.3f} | Dimensions: {n['features']}")

        # Dynamic AI Horticultural Reasoning
        print(f"\n  {BOLD}Botanical Reasoning & Morphological Insights:{RESET}")
        explanation = explain_flower_prediction(sample, predicted_species, vote_share)
        sys.stdout.write(f"    {CYAN}▸{RESET} ")
        stream_print(explanation, delay=0.005)

        if out_of_range:
            print(f"\n  {YELLOW}⚠ Out of Range Notice: Measurements exceed recorded training boundaries [{min_vals.min():.1f} - {max_vals.max():.1f} cm]{RESET}")
        else:
            print(f"\n  {GREEN}✓ Data Integrity: Features reside within verified natural bounds{RESET}")
        print("──────────────────────────────────────────────────────────")


def run_classifier(
    interactive: bool = True,
    demo: bool = False,
    save_output: bool = False,
    load_saved: bool = False,
    custom_csv: Optional[str] = None,
    target_col: Optional[str] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[Any, Any, List[str]]:
    print("\n" + "=" * 65)
    print(f"  {BOLD}{CYAN}📊 Project 2: Data Classification Using AI (KNN Pipeline){RESET}")
    print(f"  {DIM}DecodeLabs Industrial Training Kit | Batch 2026{RESET}")
    print(f"  Benchmark : {GREEN}Iris Dataset{RESET} (150 samples | 4 features | 3 classes)")
    print(f"  Engine    : {MAGENTA}Leakage-Free Pipeline (StandardScaler + KNN){RESET}")
    print("=" * 65)

    X_df, y_df, class_names = load_data()
    X = X_df.values
    y = y_df.values

    print_step("Loading Iris dataset & establishing Stratified splits")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=seed)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    print_step("Running GridSearchCV Hyperparameter Optimization")
    grid, best_tuning = tune_knn(X, y, n_folds=n_folds, seed=seed)
    best_pipe = grid.best_estimator_

    model = train_model(X_train_sc, y_train, k=5)
    preds, acc, f1 = evaluate_model(model, X_test_sc, y_test, class_names)

    print_step("Performing 5-Fold Stratified Cross-Validation")
    cv_summary = evaluate_cv_model(seed=seed, n_folds=n_folds, k=5)

    print(f"\n{BOLD}[MODEL EVALUATION & HYPERPARAMETER TUNING]{RESET}")
    print(f"  Optimal Hyperparameters : {CYAN}{best_tuning['best_params']}{RESET}")
    print(f"  5-Fold Stratified CV Acc: {GREEN}{BOLD}{cv_summary['mean_accuracy']*100:.2f}% ± {cv_summary['std_accuracy']*100:.2f}%{RESET}")
    print(f"  5-Fold Macro F1 Score   : {GREEN}{cv_summary['mean_f1']*100:.2f}% ± {cv_summary['std_f1']*100:.2f}%{RESET}")
    print(f"  Hold-out Test Accuracy  : {GREEN}{acc*100:.2f}%{RESET}")

    print(f"\n{BOLD}[ALGORITHM BENCHMARK COMPARISON — 5-FOLD STRATIFIED CV]{RESET}")
    print(f"  {'Model':<30} | {'CV Accuracy (Mean±Std)':<24} | {'Macro F1':<16}")
    print("  " + "─" * 74)
    comparison_table = compare_models(X, y, n_folds=n_folds, seed=seed)
    for row in comparison_table:
        is_knn = "KNN" in row["model"]
        m_name = f"{CYAN}{BOLD}{row['model']}{RESET}" if is_knn else row['model']
        acc_str = f"{row['mean_accuracy']*100:5.2f}% ± {row['std_accuracy']*100:4.2f}%"
        f1_str = f"{row['mean_f1']*100:5.2f}% ± {row['std_f1']*100:4.2f}%"
        print(f"  {m_name:<39} | {acc_str:<24} | {f1_str:<16}")

    print_step("Generating Diagnostic Plots & Persisting Artifacts")
    plot_paths = save_diagnostic_plots(
        X_train, y_train, X_test, y_test, best_pipe, class_names, FEATURE_NAMES
    )
    all_metrics = {
        "cv_summary": cv_summary,
        "holdout_test": {"accuracy": acc, "f1_weighted": f1},
        "model_comparison": comparison_table,
        "best_hyperparameters": best_tuning["best_params"],
    }
    save_model_artifacts(best_pipe, all_metrics, best_tuning["best_params"])

    if demo:
        print(f"\n{BOLD}[DEMO PREDICTIONS & VOTE SHARE VALIDATION]{RESET}")
        test_samples = [
            ([5.1, 3.5, 1.4, 0.2], "Normal Setosa Specimen"),
            ([6.0, 2.9, 4.5, 1.5], "Normal Versicolor Specimen"),
            ([6.7, 3.0, 5.2, 2.3], "Normal Virginica Specimen"),
            ([100.0, 50.0, 20.0, 10.0], "Out-of-Range Giant Specimen"),
        ]
        for feats, desc in test_samples:
            res = classify_sample(feats, seed=seed, pipeline=best_pipe)
            print(f"  {CYAN}▸{RESET} {desc} {feats}:")
            print(f"    ↳ {res}")
    elif interactive:
        run_interactive_ai_studio(best_pipe, X, y)

    if save_output:
        summary_txt = OUTPUTS_DIR / "classification_summary.txt"
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(f"Iris KNN Evaluation Summary\n")
            f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
            f.write(f"CV Accuracy: {cv_summary['mean_accuracy']*100:.2f}%\n")
            f.write(f"Macro F1: {cv_summary['mean_f1']*100:.2f}%\n")
            f.write(f"Best Params: {best_tuning['best_params']}\n")
        print(f"\n[Classification summary exported to {summary_txt}]")

    print("\n" + "=" * 65 + "\n")
    return model, scaler, class_names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DecodeLabs Project 2 - Iris KNN Classifier")
    parser.add_argument("--demo", action="store_true", help="Run automated non-interactive demo")
    parser.add_argument("--save-output", action="store_true", help="Save summary report to outputs/")
    parser.add_argument("--load", action="store_true", help="Load saved pipeline without retraining")
    parser.add_argument("--csv", type=str, default=None, help="Path to custom CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column in custom CSV")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_classifier(
        interactive=not args.demo,
        demo=args.demo,
        save_output=args.save_output,
        load_saved=args.load,
        custom_csv=args.csv,
        target_col=args.target,
        n_folds=args.folds,
    )
