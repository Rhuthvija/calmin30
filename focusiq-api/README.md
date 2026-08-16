# FocusIQ API

Serves the trained FocusIQ models (Random Forest for `focus_score`, SVM for
`procrastination_risk`) over HTTP as JSON.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then it's live at `http://127.0.0.1:5000`.

## Endpoints

### `GET /health`
Quick check that the service is up and which fields `/predict` expects.

### `POST /predict`
Body (all 10 fields required, all numeric):

```json
{
  "study_hours": 6,
  "sleep_hours": 8,
  "phone_usage_hrs": 3,
  "stress_level": 4,
  "motivation_level": 7,
  "failures": 0,
  "freetime": 3,
  "goout": 2,
  "famrel": 4,
  "absences": 2
}
```

Response:

```json
{
  "focus_score": 71.1,
  "procrastination_risk": "Low",
  "procrastination_confidence_pct": 42.6
}
```

Tested with: a "good habits" profile (returned focus 71.1, Low risk) and a
"poor habits" profile (returned focus 26.5, High risk, 88% confidence) — both
directionally correct. Missing fields and non-numeric values return a 400 with
a clear error message.

## Deploying

This is a standard Flask app — deploys as-is to Render, Railway, Fly.io, or
any host that runs a Python web service. Before going live:

1. In `app.py`, change `Access-Control-Allow-Origin: "*"` to your actual
   Caffeine domain, so only your site can call the API from a browser.
2. Don't run with `debug=True` in production — use a WSGI server like
   `gunicorn app:app` instead of `python app.py`.
3. Make sure the host's Python installs `scikit-learn==1.8.0` (see
   `model_manifest.json`) — a different sklearn version can fail to load
   the `.joblib` files or silently change predictions.
