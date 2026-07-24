# Automatiser train_model.py + predict_and_sync.py avec GitHub Actions

Ce guide met en place l'exécution **automatique toutes les 30 minutes** de
tes deux scripts ML, gratuitement, sans laisser un ordinateur allumé.

## Comment ça marche

GitHub met à disposition des machines gratuites ("runners") qui peuvent
exécuter du code sur un déclencheur planifié (cron). Le fichier
`.github/workflows/ml_pipeline.yml` fourni ici dit à GitHub : "toutes les
30 minutes, installe Python, installe les dépendances, lance
`train_model.py` puis `predict_and_sync.py`".

Tes clés Supabase ne sont jamais écrites dans le code : elles sont
stockées dans les **Secrets** du repo GitHub (chiffrés, jamais visibles,
même par toi une fois enregistrés).

## Étape 1 — Créer le compte et le dépôt GitHub

1. Va sur [github.com](https://github.com) et crée un compte si ce n'est
   pas déjà fait.
2. Clique sur **New repository** (bouton vert "New" en haut à gauche du
   tableau de bord).
3. Nom du dépôt, par exemple : `ocp-mineguard-ml`.
4. Choisis **Private** (recommandé — évite d'exposer publiquement la
   structure de ton projet, même si les secrets restent protégés en
   public aussi).
5. Ne coche aucune case d'initialisation (pas de README, pas de
   .gitignore — on les a déjà). Clique **Create repository**.

## Étape 2 — Envoyer les fichiers dans le dépôt

Sur ton ordinateur, dans le dossier contenant `ml_common.py`,
`train_model.py`, `predict_and_sync.py`, `requirements.txt`, le dossier
`.github/` et `.gitignore` :

```bash
cd chemin/vers/ce/dossier

git init
git add .
git commit -m "Pipeline ML OCP MineGuard"
git branch -M main
git remote add origin https://github.com/TON-NOM-UTILISATEUR/ocp-mineguard-ml.git
git push -u origin main
```

(Remplace `TON-NOM-UTILISATEUR` par ton pseudo GitHub — l'URL exacte est
affichée sur la page du dépôt vide juste après sa création.)

Si `git` n'est pas installé : télécharge-le sur
[git-scm.com](https://git-scm.com), ou utilise plus simplement le bouton
**"Add file" → "Upload files"** directement dans l'interface GitHub pour
glisser-déposer tous les fichiers (y compris le dossier `.github` en le
glissant tel quel).

## Étape 3 — Ajouter les secrets Supabase

Dans le dépôt GitHub : **Settings** (onglet en haut) → **Secrets and
variables** → **Actions** → **New repository secret**.

Ajoute ces 3 secrets un par un (nom exact à gauche, valeur à droite) :

| Nom du secret | Valeur |
|---|---|
| `SUPABASE_URL` | `https://lzbfhvjieikbdrtzwsqk.supabase.co` |
| `SUPABASE_KEY_READ` | ta clé `anon` / `publishable` (lecture, déjà utilisée dans `app.js`) |
| `SUPABASE_KEY_WRITE` | ta clé `service_role` (écriture — **Settings → API** dans Supabase, section "service_role secret") |

⚠️ La clé `service_role` donne un accès total à ta base : ne la mets
**jamais** dans `app.js`, un fichier public, ou un message. Ici, en tant
que secret GitHub Actions, elle reste chiffrée et invisible.

## Étape 4 — Vérifier que ça tourne

1. Dans le dépôt GitHub, va dans l'onglet **Actions**.
2. Tu devrais voir le workflow **"OCP MineGuard — Entraînement &
   synchronisation ML"**.
3. Clique dessus, puis **"Run workflow"** (bouton à droite) pour le
   déclencher manuellement une première fois, sans attendre les 30
   minutes.
4. Regarde les logs : chaque étape (installation, entraînement,
   synchronisation) s'affiche en détail. Si `predict_and_sync.py` se
   termine par "✅ Synchronisation terminée", c'est bon.
5. Recharge `ml.html` dans ton navigateur — la date "entraîné le..."
   dans le bandeau doit correspondre à l'heure du run que tu viens de
   déclencher.

Ensuite, plus rien à faire : le workflow se relance tout seul toutes les
30 minutes, tant que la base de données évolue (ajouts ou suppressions
de lectures capteurs), le modèle et les prédictions se remettent à jour
automatiquement.

## À savoir

- **Délai possible** : GitHub ne garantit pas une précision à la minute
  près sur les tâches planifiées (cron) — un léger retard de quelques
  minutes peut arriver en cas de forte charge sur leurs serveurs. C'est
  normal et sans conséquence pour ce cas d'usage.
- **Gratuit** : les dépôts privés incluent 2 000 minutes/mois d'exécution
  gratuite. Ce pipeline prend ~1-2 minutes par run ; à raison de 48
  runs/jour (toutes les 30 min), ça reste largement dans le quota
  gratuit (~50-100 minutes/jour max, largement sous la limite mensuelle).
- **Suivi des échecs** : si le workflow échoue (ex: clé expirée, panne
  Supabase), GitHub t'envoie un email automatiquement à l'adresse liée à
  ton compte.
- **Modifier la fréquence** : pour changer les 30 minutes, édite la ligne
  `cron: "*/30 * * * *"` dans `.github/workflows/ml_pipeline.yml` (ex:
  `"0 * * * *"` pour toutes les heures pile).
