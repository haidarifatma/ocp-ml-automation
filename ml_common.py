# -*- coding: utf-8 -*-
"""
ml_common.py
============
Logique métier partagée entre train_model.py et predict_and_sync.py.

Implémente la classification à 4 niveaux décrite dans la spec projet :

    0 = Healthy   (🟢 fonctionnement normal)
    1 = Warning   (🟠 dégradation / anomalie détectée, pas encore de panne)
    2 = Critical  (🔴 risque élevé de panne imminente)
    3 = Failure   (⚫ panne réelle, équipement indisponible)

Principe (cf. ISO 13374 - surveillance de l'état des machines) :
Une machine n'est PAS "en panne" dès qu'un seuil est dépassé. On combine :
  1) les valeurs instantanées des capteurs vs seuils (warn/crit),
  2) la TENDANCE dans le temps (dérive récente vs passé récent),
  3) l'état opérationnel réel (machine_failure = 0/1, fourni par les capteurs
     terrain / arrêt moteur / débit nul, etc.).

On ne déclare jamais une panne à partir d'un seul capteur isolé : le score de
dégradation est une combinaison pondérée des 4 capteurs + de leur évolution.
"""

import numpy as np
import pandas as pd

# ===== SEUILS (alignés sur THRESH dans app.js) =====
THRESH = {
    "temperature": {"warn": 50, "crit": 75, "max": 100},
    "humidity":    {"warn": 70, "crit": 95, "max": 100},
    "vibration":   {"warn": 400, "crit": 600, "max": 1000},
}

SENSOR_COLS = ["temperature", "humidity", "vibration"]

# Poids relatifs de chaque capteur dans le score de dégradation.
# La vibration et la température sont les indicateurs mécaniques les plus
# parlants pour de la maintenance prédictive ; l'humidité a un rôle
# secondaire (ambiance / environnement plutôt que défaillance directe).
SENSOR_WEIGHTS = {
    "temperature": 0.40,
    "vibration": 0.45,
    "humidity": 0.15,
}

CLASS_NAMES = ["Healthy", "Warning", "Critical", "Failure"]


def sensor_severity(value: float, metric: str) -> float:
    """
    Convertit une valeur brute de capteur en un score de sévérité 0-100+
    par rapport aux seuils warn/crit. 0 = parfaitement normal, 100 = au
    seuil critique, >100 = au-delà du seuil critique.
    """
    t = THRESH[metric]
    if value <= t["warn"]:
        # 0 -> 60 dans la zone normale (approche progressive du warn)
        return 60.0 * value / max(t["warn"], 1e-6)
    if value <= t["crit"]:
        # 60 -> 100 dans la zone warning
        span = max(t["crit"] - t["warn"], 1e-6)
        return 60.0 + 40.0 * (value - t["warn"]) / span
    # au-delà du seuil critique : on continue de monter au-delà de 100
    over = value - t["crit"]
    return 100.0 + 40.0 * (over / max(t["crit"], 1e-6))


def degradation_score(row: pd.Series) -> float:
    """Score composite pondéré de dégradation (0 = sain, >100 = critique+)."""
    score = 0.0
    for metric, w in SENSOR_WEIGHTS.items():
        score += w * sensor_severity(row[metric], metric)
    return score


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features de TENDANCE par équipement : moyenne mobile et delta
    récent (fenêtre de 5 lectures vs les 5 précédentes), pour capturer une
    dégradation progressive même quand la valeur instantanée est encore dans
    la zone normale/warning. Le dataframe doit être trié par equipment_id
    puis timestamp croissant.
    """
    df = df.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)
    out_frames = []
    for eq_id, g in df.groupby("equipment_id", sort=False):
        g = g.copy()
        for m in SENSOR_COLS:
            g[f"{m}_roll5"] = g[m].rolling(5, min_periods=1).mean()
            g[f"{m}_delta5"] = g[m].diff(5).fillna(0.0)
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def compute_trend_bonus(row: pd.Series) -> float:
    """
    Bonus de score ajouté quand plusieurs capteurs montrent une dérive
    positive récente (dégradation progressive), même si les valeurs
    instantanées ne sont pas encore critiques.
    """
    bonus = 0.0
    for m in ["temperature", "vibration", "humidity"]:
        delta = row.get(f"{m}_delta5", 0.0)
        if delta > 0:
            # normalisé grossièrement par le seuil critique du capteur
            bonus += min(delta / THRESH[m]["crit"] * 25.0, 8.0)
    return bonus


def label_state(row: pd.Series) -> int:
    """
    Applique la logique 4-niveaux décrite dans la spec :

        Healthy  : tous les paramètres dans leur plage normale
        Warning  : un ou plusieurs paramètres s'éloignent du normal,
                   dégradation potentielle, machine encore fonctionnelle
        Critical : plusieurs indicateurs montrent un risque élevé de
                   perte de fonctionnement (dérive forte + tendance)
        Failure  : l'équipement ne peut plus assurer sa fonction
                   (signal terrain machine_failure = 1)
    """
    # Priorité absolue : un arrêt réel constaté sur le terrain = Failure,
    # quelles que soient les valeurs instantanées des capteurs.
    if int(row.get("machine_failure", 0)) == 1:
        return 3

    score = degradation_score(row) + compute_trend_bonus(row)

    if score < 45:
        return 0  # Healthy
    if score < 65:
        return 1  # Warning
    return 2      # Critical (approche ou dépasse la zone de panne sans
                  # qu'un arrêt réel ait encore été constaté)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice de features utilisée par le Random Forest."""
    feature_cols = []
    for m in SENSOR_COLS:
        feature_cols += [m, f"{m}_roll5", f"{m}_delta5"]
    return df[feature_cols].fillna(0.0)


def dominant_cause(row: pd.Series) -> str:
    """Retourne le nom du capteur (FR) le plus responsable de la sévérité."""
    sev = {m: sensor_severity(row[m], m) * SENSOR_WEIGHTS[m] for m in SENSOR_COLS}
    top = max(sev, key=sev.get)
    fr = {
        "temperature": "température",
        "vibration": "vibrations",
        "humidity": "humidité",
    }
    return fr[top]
