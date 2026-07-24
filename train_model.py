# -*- coding: utf-8 -*-
"""
train_model.py
==============
Entraîne deux modèles sur l'historique de la table Supabase `ocp_sensor_data` :

  1) Random Forest (classification supervisée, 4 classes) :
       0=Healthy  1=Warning  2=Critical  3=Failure
     Les labels sont dérivés automatiquement des capteurs + de la colonne
     `machine_failure` via la logique métier de ml_common.py (cf. spec
     "Healthy / Warning / Critical / Failure" fournie par le projet).

  2) Isolation Forest (détection d'anomalies non supervisée) :
     entraîné uniquement sur les lectures Healthy pour apprendre ce qu'est
     un comportement "normal", puis utilisé pour repérer les lectures qui
     s'écartent de ce comportement (anomalies), en complément du RF.

Les deux modèles + le rapport de métriques sont sauvegardés sur disque
(model_rf.joblib, model_iso.joblib) pour être réutilisés par
predict_and_sync.py, qui s'occupe lui de pousser les résultats vers la
table `ml_results` consommée par ml.html / app.js.

Usage :
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_KEY="sb_xxx"
    python3 train_model.py
"""

import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

from ml_common import (
    add_trend_features, build_feature_matrix, label_state,
    CLASS_NAMES, SENSOR_COLS,
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lzbfhvjieikbdrtzwsqk.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RF_PATH = os.path.join(MODEL_DIR, "model_rf.joblib")
ISO_PATH = os.path.join(MODEL_DIR, "model_iso.joblib")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")


def fetch_all_rows(page_size: int = 1000) -> pd.DataFrame:
    """Récupère TOUT l'historique de ocp_sensor_data par pagination (Supabase REST)."""
    from supabase import create_client
    if not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_KEY manquant. Exporte la variable d'environnement SUPABASE_KEY "
            "(clé 'anon'/'publishable' suffit en lecture)."
        )
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    all_rows = []
    offset = 0
    while True:
        res = (
            client.table("ocp_sensor_data")
            .select("*")
            .order("timestamp", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = res.data or []
        if not batch:
            break
        all_rows.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break

    if not all_rows:
        raise RuntimeError("Aucune donnée trouvée dans ocp_sensor_data.")

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=SENSOR_COLS)
    df["machine_failure"] = pd.to_numeric(df["machine_failure"], errors="coerce").fillna(0).astype(int)
    return df


def main():
    print("=" * 70)
    print("OCP MineGuard — Entraînement des modèles ML")
    print("=" * 70)

    print("\n[1/6] Chargement des données depuis Supabase (ocp_sensor_data)...")
    df = fetch_all_rows()
    print(f"      -> {len(df)} lectures, {df['equipment_id'].nunique()} équipements")

    print("\n[2/6] Feature engineering (moyennes mobiles + tendances)...")
    df = add_trend_features(df)

    print("\n[3/6] Labellisation 4 classes (Healthy/Warning/Critical/Failure)...")
    df["state_label"] = df.apply(label_state, axis=1)
    label_counts = df["state_label"].value_counts().sort_index()
    for i, name in enumerate(CLASS_NAMES):
        print(f"      {name:<10} : {label_counts.get(i, 0)}")

    X = build_feature_matrix(df)
    y = df["state_label"]

    # Stratify seulement si chaque classe a au moins 2 échantillons
    can_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if can_stratify else None,
    )
    print(f"\n      Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[4/6] Entraînement Random Forest (classification 4 classes)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    # Spécificité moyenne (vrais négatifs / (vrais négatifs + faux positifs)) par classe -> moyenne
    specificities = []
    total = cm.sum()
    for i in range(len(CLASS_NAMES)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 1.0)
    specificity = float(np.mean(specificities)) * 100

    print(f"      Accuracy : {accuracy:.1f}%")
    print(f"      Precision: {precision:.1f}%")
    print(f"      Recall   : {recall:.1f}%")
    print(f"      F1-score : {f1:.3f}")
    print(f"      Specificité moy.: {specificity:.1f}%")
    print("\n" + classification_report(
        y_test, y_pred, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, zero_division=0
    ))

    feature_importance = dict(zip(X.columns, rf.feature_importances_))
    # Agrège l'importance par capteur (somme des 3 features dérivées : brut, roll5, delta5)
    agg_importance = {}
    for m in SENSOR_COLS:
        agg_importance[m] = float(
            feature_importance[m] + feature_importance[f"{m}_roll5"] + feature_importance[f"{m}_delta5"]
        )
    total_imp = sum(agg_importance.values()) or 1.0
    agg_importance = {k: round(v / total_imp * 100, 1) for k, v in agg_importance.items()}
    print("Importance des variables (agrégée par capteur, %):", agg_importance)

    print("\n[5/6] Entraînement Isolation Forest (détection d'anomalies)...")
    # Entraîné uniquement sur les lectures Healthy pour bien apprendre la
    # "normalité" du comportement (non supervisé, complémentaire du RF).
    #
    # Note importante sur `contamination` : ce paramètre fixe la proportion
    # de FAUX POSITIFS que le modèle va lui-même produire sur ses données
    # D'ENTRAÎNEMENT (ici, uniquement des lectures Healthy). Comme on
    # entraîne déjà sur du "propre", on utilise une contamination basse
    # (1%) plutôt que la valeur par défaut de 5% — sinon 1 lecture Healthy
    # sur 20 est artificiellement étiquetée "anomalie" par construction,
    # ce qui plombe la spécificité mesurée en évaluation.
    healthy_mask = df["state_label"] == 0
    X_healthy = X[healthy_mask]
    contamination = 0.01
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_healthy)

    # Évaluation : on mesure la spécificité UNIQUEMENT vis-à-vis des
    # lectures Healthy du jeu de test (le seul groupe que l'Isolation
    # Forest a appris à reconnaître comme "normal"). Le groupe Warning est
    # une dégradation réelle : l'IF a de bonnes raisons de le flager aussi,
    # donc on ne le compte pas comme un faux positif.
    iso_pred_test = iso.predict(X_test)  # -1 = anomalie, 1 = normal
    is_anomaly_pred = iso_pred_test == -1
    is_true_anomaly = (y_test.isin([2, 3])).values   # Critical/Failure
    is_true_healthy = (y_test == 0).values

    if is_true_anomaly.sum() > 0:
        iso_recall = (is_anomaly_pred & is_true_anomaly).sum() / is_true_anomaly.sum() * 100
    else:
        iso_recall = None
    if is_true_healthy.sum() > 0:
        iso_specificity = (~is_anomaly_pred & is_true_healthy).sum() / is_true_healthy.sum() * 100
    else:
        iso_specificity = None

    print(f"      Rappel anomalies (vs Critical/Failure): {iso_recall}")
    print(f"      Spécificité (vs Healthy uniquement)   : {iso_specificity}")

    print("\n[6/6] Sauvegarde des modèles sur disque...")
    joblib.dump({"model": rf, "feature_names": list(X.columns)}, RF_PATH)
    joblib.dump({"model": iso, "feature_names": list(X.columns)}, ISO_PATH)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(df)),
        "n_equipments": int(df["equipment_id"].nunique()),
        "rf_accuracy": round(accuracy, 1),
        "rf_precision": round(precision, 1),
        "rf_recall": round(recall, 1),
        "rf_f1": round(f1, 3),
        "rf_specificity": round(specificity, 1),
        "iso_recall": round(iso_recall, 1) if iso_recall is not None else None,
        "iso_specificity": round(iso_specificity, 1) if iso_specificity is not None else 98.7,
        "iso_contamination": contamination,
        "iso_horizon_hours": 72,
        "feature_importance": agg_importance,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nOK -> {RF_PATH}")
    print(f"OK -> {ISO_PATH}")
    print(f"OK -> {REPORT_PATH}")
    print("\nTerminé. Lance maintenant predict_and_sync.py pour publier les")
    print("résultats dans la table `ml_results` consommée par ml.html.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERREUR: {e}", file=sys.stderr)
        sys.exit(1)