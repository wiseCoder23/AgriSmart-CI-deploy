# 🌱 AgriSmart CI — Déploiement

API Flask du système de recommandation agricole pour la Côte d'Ivoire.

## Structure

```
AgriSmart-CI-deploy/
├── api/
│   ├── app_flask.py
│   └── templates/
│       └── index.html
├── models/
│   └── agrismart_model/    ← .pkl + metadata.json
├── Procfile
├── runtime.txt
└── requirements.txt
```

## Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Interface UI |
| `/health` | GET | Statut API |
| `/stats` | GET | Statistiques globales |
| `/cultures` | GET | Liste des cultures |
| `/agriculteurs` | GET | Liste des agriculteurs |
| `/profile/<id>` | GET | Profil agriculteur |
| `/recommend` | POST | Recommandations |
| `/cold-start` | POST | Recommandations sans historique |
| `/explain/<id>` | GET | Explication d'une recommandation |

## Déploiement Render

- **Build Command :** `pip install -r requirements.txt`
- **Start Command :** `gunicorn --chdir api app_flask:app`

---

Projet académique — ENSEA Abidjan ISE3 | Edoh Yawo Anderson
