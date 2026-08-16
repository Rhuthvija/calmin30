"""
FocusIQ Prediction API
----------------------
Serves the trained Random Forest (focus_score) and SVM (procrastination_risk)
models from FocusIQ_v3.ipynb over a simple HTTPS-ready JSON API.

Run locally:
    pip install -r requirements.txt
    python app.py
    -> served at http://127.0.0.1:5000

Endpoints:
    GET  /health    -> quick check that the service + models are loaded
    POST /predict    -> the real prediction endpoint
"""

import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load models once at startup (not per-request) ───────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_model     = joblib.load(os.path.join(BASE_DIR, "focus_rf_model.joblib"))
svm_model    = joblib.load(os.path.join(BASE_DIR, "procrastination_svm_model.joblib"))
scaler       = joblib.load(os.path.join(BASE_DIR, "feature_scaler.joblib"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "procrastination_label_encoder.joblib"))
FEATURE_ORDER = joblib.load(os.path.join(BASE_DIR, "feature_order.joblib"))

# Reasonable bounds per feature -- mirrors the ranges the models were
# trained on (see Part 2 of the notebook). Used only to validate/clip
# input, not to change the model itself.
FEATURE_BOUNDS = {
    "study_hours":      (0, 12),
    "sleep_hours":       (0, 12),
    "phone_usage_hrs":   (0, 14),
    "stress_level":      (1, 10),
    "motivation_level":  (1, 10),
    "failures":          (0, 4),
    "freetime":          (1, 5),
    "goout":             (1, 5),
    "famrel":            (1, 5),
    "absences":          (0, 100),
}


# ── CORS: allow your Caffeine frontend to call this from the browser ───
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"  # tighten to your domain before going live
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "expected_features": FEATURE_ORDER
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # CORS preflight -- browsers send this before the real POST
        return jsonify({}), 200

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    # ── Validate every required feature is present and numeric ──────────
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    values = []
    for feat in FEATURE_ORDER:
        try:
            v = float(data[feat])
        except (TypeError, ValueError):
            return jsonify({"error": f"Field '{feat}' must be a number"}), 400
        lo, hi = FEATURE_BOUNDS[feat]
        v = max(lo, min(hi, v))  # clip to the range the model was trained on
        values.append(v)

    # ── Scale (exact same scaler fit during training) then predict ──────
    X_scaled = scaler.transform([values])

    focus_score = float(rf_model.predict(X_scaled)[0])
    focus_score = round(max(0, min(100, focus_score)), 1)

    proc_encoded = svm_model.predict(X_scaled)[0]
    proc_label = label_encoder.inverse_transform([proc_encoded])[0]
    proc_probabilities = svm_model.predict_proba(X_scaled)[0]
    proc_confidence = round(float(max(proc_probabilities)) * 100, 1)

    return jsonify({
        "focus_score": focus_score,
        "procrastination_risk": proc_label,
        "procrastination_confidence_pct": proc_confidence
    })


if __name__ == "__main__":
    # debug=True is fine locally; turn off before deploying
    app.run(host="0.0.0.0", port=5001, debug=True)
