# -*- coding: utf-8 -*-
"""
predict_and_sync.py
====================
Étape 2 du pipeline : à exécuter régulièrement (cron / tâche planifiée,
p.ex. toutes les 15-30 minutes) APRÈS train_model.py.

Ce script :
  1) recharge les lectures récentes de `ocp_sensor_data` (par équipement),
  2) applique le Random Forest (4 classes) + l'Isolation Forest entraînés,
  3) calcule, pour chaque équipement, une prédiction de risque de panne
     (%), une classe de risque (low/medium/high), une explication ("cause
     principale"), et un flag anomalie,
  4) upsert le tout dans une seule ligne (id=1) de la table `ml_results`
     de Supabase, au format EXACT attendu par ml.html / app.js
     (renderML() dans app.js lit ces champs directement).

Table `ml_results` attendue (à créer une fois dans Supabase si absente) :

    create table if not exists ml_results (
      id bigint primary key,
      trained_at timestamptz,
      training_rows integer,
      rf_accuracy numeric,
      rf_precision numeric,
      rf_recall numeric,
      rf_f1 numeric,
      rf_specificity numeric,
      iso_specificity numeric,
      iso_contamination numeric,
      iso_horizon_hours integer,
      feature_importance jsonb,
      predictions jsonb
    );

Usage :
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_KEY="sb_xxx"   # nécessite les droits d'écriture (service role
                                    # recommandé côté serveur, jamais exposée au front)
    python3 predict_and_sync.py
"""

import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from ml_common import (
    add_trend_features, build_feature_matrix, dominant_cause,
    SENSOR_COLS, CLASS_NAMES,
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lzbfhvjieikbdrtzwsqk.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RF_PATH = os.path.join(MODEL_DIR, "model_rf.joblib")
ISO_PATH = os.path.join(MODEL_DIR, "model_iso.joblib")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")

LOOKBACK_ROWS_PER_EQUIP = 60  # fenêtre récente suffisante pour les features de tendance


def get_client():
    from supabase import create_client
    if not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_KEY manquant (variable d'environnement).")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_recent_data(client) -> pd.DataFrame:
    """
    Récupère les lectures récentes de ocp_sensor_data, PAR ÉQUIPEMENT.

    Important : on ne peut pas se contenter d'un simple
    `.order(timestamp desc).limit(2000)` global, car si les équipements
    n'écrivent pas tous à la même fréquence, un équipement peu actif (peu
    de nouvelles lectures) peut se retrouver totalement absent de la
    fenêtre des 2000 lignes les plus récentes globalement — noyé par les
    autres équipements plus bavards. Le pipeline retomberait alors sur ses
    lectures les plus anciennes disponibles pour lui (potentiellement une
    période de panne passée), et prédirait un risque complètement obsolète
    et déconnecté de son état réel actuel.

    On interroge donc chaque equipment_id séparément, avec sa propre
    limite, pour garantir que CHAQUE équipement contribue ses lectures les
    plus récentes à lui, indépendamment du volume des autres.
    """
    ids_res = (
        client.table("ocp_sensor_data")
        .select("equipment_id")
        .execute()
    )
    equipment_ids = sorted({r["equipment_id"] for r in (ids_res.data or []) if r.get("equipment_id") is not None})
    if not equipment_ids:
        raise RuntimeError("Aucune donnée dans ocp_sensor_data.")

    all_rows = []
    for eq_id in equipment_ids:
        res = (
            client.table("ocp_sensor_data")
            .select("*")
            .eq("equipment_id", eq_id)
            .order("timestamp", desc=True)
            .limit(LOOKBACK_ROWS_PER_EQUIP)
            .execute()
        )
        all_rows.extend(res.data or [])

    if not all_rows:
        raise RuntimeError("Aucune donnée récente dans ocp_sensor_data.")

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=SENSOR_COLS)
    df["machine_failure"] = pd.to_numeric(df["machine_failure"], errors="coerce").fillna(0).astype(int)
    return df.sort_values(["equipment_id", "timestamp"])


def risk_class(pct: float) -> str:
    if pct >= 55:
        return "high"
    if pct >= 25:
        return "medium"
    return "low"


def build_detail_text(state_idx: int, cause: str, delta_note: str) -> str:
    if state_idx == 3:
        return f"Panne détectée — équipement indisponible ({cause} en cause)."
    if state_idx == 2:
        return f"Risque élevé de panne — dérive marquée de la {cause}{delta_note}."
    if state_idx == 1:
        return f"Anomalie détectée — surveillance renforcée recommandée ({cause})."
    return "Fonctionnement normal — aucune action requise."


def main():
    print("=" * 70)
    print("OCP MineGuard — Prédiction & synchronisation Supabase")
    print("=" * 70)

    print("\n[1/5] Chargement des modèles entraînés...")
    if not (os.path.exists(RF_PATH) and os.path.exists(ISO_PATH)):
        raise RuntimeError(
            "Modèles introuvables. Lance d'abord `python3 train_model.py` "
            "pour générer model_rf.joblib et model_iso.joblib."
        )
    rf_bundle = joblib.load(RF_PATH)
    iso_bundle = joblib.load(ISO_PATH)
    rf = rf_bundle["model"]
    iso = iso_bundle["model"]
    feature_names = rf_bundle["feature_names"]

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("\n[2/5] Connexion Supabase & chargement des données récentes...")
    client = get_client()
    df = fetch_recent_data(client)
    print(f"      -> {len(df)} lectures, {df['equipment_id'].nunique()} équipements")

    print("\n[3/5] Feature engineering...")
    df = add_trend_features(df)
    X_all = build_feature_matrix(df)[feature_names]

    print("\n[4/5] Prédiction (RF 4-classes + Isolation Forest)...")
    state_pred = rf.predict(X_all)
    state_proba = rf.predict_proba(X_all)
    anomaly_pred = iso.predict(X_all)  # -1 anomalie, 1 normal
    anomaly_score = -iso.decision_function(X_all)  # plus haut = plus anormal

    df = df.reset_index(drop=True)
    df["state_pred"] = state_pred
    df["is_anomaly"] = anomaly_pred == -1
    df["anomaly_score"] = anomaly_score

    # ----- risk_pct : ancré sur l'état prédit pour ne jamais contredire le
    # texte affiché (ex: un équipement "Healthy" ne doit jamais afficher un
    # risque plus élevé qu'un équipement "Warning"). Chaque état a sa bande
    # de risque dédiée, et P(Critical)+P(Failure) sert seulement à nuancer
    # la position DANS la bande de l'état prédit.
    class_order = list(rf.classes_)
    idx_critical = class_order.index(2) if 2 in class_order else None
    idx_failure = class_order.index(3) if 3 in class_order else None
    severity_prob = np.zeros(len(df))
    if idx_critical is not None:
        severity_prob += state_proba[:, idx_critical]
    if idx_failure is not None:
        severity_prob += state_proba[:, idx_failure]

    STATE_RISK_BAND = {
        0: (0, 25),    # Healthy
        1: (25, 55),   # Warning
        2: (55, 90),   # Critical
        3: (90, 100),  # Failure
    }
    risk_pct = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        lo, hi = STATE_RISK_BAND[int(state_pred[i])]
        risk_pct[i] = int(round(lo + severity_prob[i] * (hi - lo)))
    df["risk_pct"] = risk_pct

    print("\n[5/5] Construction des prédictions par équipement...")
    predictions = []
    for eq_id, g in df.groupby("equipment_id"):
        last = g.iloc[-1]
        cause = dominant_cause(last)
        # note sur la tendance récente pour l'explication
        delta_note = ""
        biggest_delta = -1
        for m in ["temperature", "vibration", "humidity"]:
            d = last.get(f"{m}_delta5", 0.0)
            if d > biggest_delta:
                biggest_delta = d
        if biggest_delta > 0:
            delta_note = " sur les dernières lectures"

        state_idx = int(last["state_pred"])
        predictions.append({
            "equipment_id": int(eq_id),
            "equipment_name": last["equipment_name"],
            "location": last.get("location", ""),
            "state": CLASS_NAMES[state_idx],
            "risk_pct": int(last["risk_pct"]),
            "risk_class": risk_class(last["risk_pct"]),
            "is_anomaly": bool(last["is_anomaly"]),
            "detail": build_detail_text(state_idx, cause, delta_note),
        })

    # trie par risque décroissant pour l'affichage (défaillances prédites)
    predictions.sort(key=lambda p: p["risk_pct"], reverse=True)

    for p in predictions:
        print(f"      {p['equipment_name']:<20} {p['state']:<10} risque={p['risk_pct']}%  anomalie={p['is_anomaly']}")

    ml_row = {
        "id": 1,
        "trained_at": report["trained_at"],
        "training_rows": report["training_rows"],
        "rf_accuracy": report["rf_accuracy"],
        "rf_precision": report["rf_precision"],
        "rf_recall": report["rf_recall"],
        "rf_f1": report["rf_f1"],
        "rf_specificity": report["rf_specificity"],
        "iso_specificity": report["iso_specificity"],
        "iso_contamination": report["iso_contamination"],
        "iso_horizon_hours": report["iso_horizon_hours"],
        "feature_importance": {
            "Température": report["feature_importance"]["temperature"],
            "Vibration": report["feature_importance"]["vibration"],
            "Humidité": report["feature_importance"]["humidity"],
            "Luminosité": report["feature_importance"]["light"],
        },
        "predictions": predictions,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }

    print("\nEnvoi (upsert) vers la table `ml_results`...")
    client.table("ml_results").upsert(ml_row).execute()
    print("✅ Synchronisation terminée. La page ml.html affichera désormais")
    print("   ces résultats réels au lieu du mode démonstration.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERREUR: {e}", file=sys.stderr)
        sys.exit(1)
