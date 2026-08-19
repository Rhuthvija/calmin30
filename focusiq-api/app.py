"""
FocusIQ Prediction API (v4 -- 8-feature model)
-----------------------------------------------
Serves the retrained Random Forest (focus_score) and SVM (procrastination_risk)
models. Dropped: failures, famrel, absences (low predictive weight, high user
friction). Added: exercise_done.

Run locally:
    pip install -r requirements.txt
    python app.py

Endpoints:
    GET  /health
    POST /predict
"""

import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_model      = joblib.load(os.path.join(BASE_DIR, "focus_rf_model.joblib"))
svm_model     = joblib.load(os.path.join(BASE_DIR, "procrastination_svm_model.joblib"))
scaler        = joblib.load(os.path.join(BASE_DIR, "feature_scaler.joblib"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "procrastination_label_encoder.joblib"))
FEATURE_ORDER = joblib.load(os.path.join(BASE_DIR, "feature_order.joblib"))
# ['study_hours','sleep_hours','phone_usage_hrs','stress_level',
#  'motivation_level','freetime','goout','exercise_done']

FEATURE_BOUNDS = {
    "study_hours":     (0, 12),
    "sleep_hours":      (0, 12),
    "phone_usage_hrs":  (0, 14),
    "stress_level":     (1, 10),
    "motivation_level": (1, 10),
    "freetime":         (1, 5),
    "goout":            (1, 5),
    "exercise_done":    (0, 1),
}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
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
        return jsonify({}), 200

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

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
        v = max(lo, min(hi, v))
        values.append(v)

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
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
