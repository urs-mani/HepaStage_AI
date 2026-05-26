from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "..", "models", "model_metadata.json")

model = None
scaler = None
model_metadata = {}
history = []
MAX_HISTORY = 100

FEATURES = ["Age", "Bilirubin", "Albumin", "SGOT", "Bilirubin_Albumin_ratio", "Age_SGOT", "Albumin_SGOT"]
VALIDATION_FEATURES = ["Age", "Bilirubin", "Albumin", "SGOT"]

STAGE_INFO = {
    1: {
        "label": "Stage 1 — Mild",
        "description": "Minimal hepatic fibrosis. Liver function is largely preserved. Lifestyle modifications and monitoring are typically sufficient.",
        "color": "#22c55e",
        "risk": "Low",
        "problem": "Early-stage fibrosis with minimal scarring",
        "solution": "Disease prevention, lifestyle changes, annual monitoring"
    },
    2: {
        "label": "Stage 2 — Moderate",
        "description": "Moderate fibrosis present. Liver shows early signs of structural changes. Medical management and regular follow-up are recommended.",
        "color": "#f59e0b",
        "risk": "Moderate",
        "problem": "Progressive fibrosis with bridging",
        "solution": "Pharmacological interventions, bi-annual monitoring"
    },
    3: {
        "label": "Stage 3 — Severe",
        "description": "Advanced fibrosis approaching cirrhosis. Significant hepatic dysfunction. Specialist care and possible intervention required.",
        "color": "#f97316",
        "risk": "High",
        "problem": "Advanced fibrosis approaching cirrhosis",
        "solution": "Specialist management, quarterly evaluations"
    },
    4: {
        "label": "Stage 4 — Critical",
        "description": "Cirrhosis confirmed. Severe liver damage with high risk of complications. Urgent specialist evaluation and treatment planning needed.",
        "color": "#ef4444",
        "risk": "Critical",
        "problem": "Established cirrhosis with decompensation",
        "solution": "Urgent intensive care, transplant evaluation"
    }
}

RANGES = {
    "Age":       (18, 100),
    "Bilirubin": (0.1, 20.0),
    "Albumin":   (1.0, 5.5),
    "SGOT":      (5, 500)
}

FEATURE_DOCS = {
    "Age": "Patient age in years. Higher age typically increases risk of advanced fibrosis.",
    "Bilirubin": "Total bilirubin mg/dL; indicator of liver excretory function.",
    "Albumin": "Serum albumin g/dL; lower values indicate impaired liver synthetic function.",
    "SGOT": "AST enzyme U/L; elevation reflects hepatocellular damage."
}


def load_model():
    global model, scaler, model_metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl not found. Please run: python train_model.py")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully.")

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)
        logger.info("Metadata loaded successfully.")


def validate_input(data):
    errors = []
    values = {}
    warnings = []

    for feature in VALIDATION_FEATURES:
        if feature not in data:
            errors.append(f"Missing field: {feature}")
            continue

        raw = data[feature]
        if isinstance(raw, bool):
            errors.append(f"'{feature}' must be numeric, not bool.")
            continue

        try:
            val = float(raw)
        except (ValueError, TypeError):
            errors.append(f"'{feature}' must be a numeric value.")
            continue

        lo, hi = RANGES[feature]
        if val < lo:
            warnings.append(f"{feature} is below expected minimum ({val} < {lo}).")
        if val > hi:
            warnings.append(f"{feature} is above expected maximum ({val} > {hi}).")

        if not (0 <= val <= 1000):
            errors.append(f"{feature} value is unrealistic.")
            continue

        values[feature] = val

    return values, errors, warnings


def _record_history(entry):
    global history
    history.insert(0, entry)
    if len(history) > MAX_HISTORY:
        history = history[:MAX_HISTORY]


def _construct_prediction_body(prediction, probabilities, values):
    stage_data = STAGE_INFO.get(prediction, {})
    classes = getattr(model, "classes_", list(range(1, 5)))
    prob_map = {f"stage_{int(c)}": round(probabilities[i] * 100, 1) for i, c in enumerate(classes)}

    return {
        "stage": prediction,
        "label": stage_data.get("label"),
        "description": stage_data.get("description"),
        "risk": stage_data.get("risk"),
        "problem": stage_data.get("problem"),
        "solution": stage_data.get("solution"),
        "color": stage_data.get("color"),
        "probabilities": prob_map,
        "inputs": values,
        "model": {
            "selected": model_metadata.get("best_model"),
            "score": model_metadata.get("results", {}).get(model_metadata.get("best_model", ""), {}).get("test_accuracy")
        },
        "scaler": {"loaded": scaler is not None},
    }


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "HepaStage AI API is running",
        "model_loaded": model is not None,
        "model_info": model_metadata.get("best_model"),
        "version": "1.0"
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json() if request.is_json else None
    if not payload:
        return jsonify({"error": "Request must be JSON."}), 400

    values, errors, warnings = validate_input(payload)
    if errors:
        return jsonify({"error": "Validation failed.", "details": errors}), 422

    eps = 0.1
    ratio = values["Bilirubin"] / max(values["Albumin"], eps)
    age_sgot = values["Age"] * values["SGOT"]
    alb_sgot = values["Albumin"] * values["SGOT"]
    features_data = [values["Age"], values["Bilirubin"], values["Albumin"], values["SGOT"], ratio, age_sgot, alb_sgot]
    features = pd.DataFrame([features_data], columns=FEATURES)
    if scaler is not None:
        features = pd.DataFrame(scaler.transform(features), columns=FEATURES)

    pred = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]

    response = _construct_prediction_body(pred, proba, values)
    response["warnings"] = warnings

    _record_history({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": "/predict",
        "request": values,
        "response": response
    })

    return jsonify(response)


@app.route("/analyze", methods=["POST"])
def analyze():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    data = request.get_json()
    values, errors, warnings = validate_input(data)
    if errors:
        return jsonify({"error": "Validation failed.", "details": errors}), 422

    use_scaler = bool(data.get("use_scaler", True))
    eps = 0.1
    ratio = values["Bilirubin"] / max(values["Albumin"], eps)
    age_sgot = values["Age"] * values["SGOT"]
    alb_sgot = values["Albumin"] * values["SGOT"]
    features_data = [values["Age"], values["Bilirubin"], values["Albumin"], values["SGOT"], ratio, age_sgot, alb_sgot]
    features = pd.DataFrame([features_data], columns=FEATURES)
    if use_scaler and scaler is not None:
        features = pd.DataFrame(scaler.transform(features), columns=FEATURES)

    pred = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]

    response = _construct_prediction_body(pred, proba, values)
    response["use_scaler"] = use_scaler and scaler is not None
    response["warnings"] = warnings

    _record_history({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": "/analyze",
        "request": {**values, "use_scaler": use_scaler},
        "response": response
    })

    return jsonify(response)


@app.route("/ranges", methods=["GET"])
def ranges():
    return jsonify({"ranges": RANGES})


@app.route("/info", methods=["GET"])
def info():
    rich_info = {k: {**v} for k, v in STAGE_INFO.items()}
    return jsonify({"stages": rich_info})


@app.route("/features", methods=["GET"])
def features():
    return jsonify({"features": FEATURE_DOCS})


@app.route("/api-docs", methods=["GET"])
def docs():
    return jsonify({
        "endpoints": {
            "/": "GET health",
            "/predict": "POST prediction (default, scaler applied if available)",
            "/analyze": "POST analysis with use_scaler optional",
            "/ranges": "GET accepted input ranges",
            "/info": "GET stage medical details",
            "/features": "GET biomarker docs",
            "/history": "GET prediction history"
        },
        "models": model_metadata,
        "note": "All output is for research and education only. Not clinical advice."
    })


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({"count": len(history), "records": history})


if __name__ == "__main__":
    try:
        load_model()
    except FileNotFoundError as e:
        logger.warning(str(e))
    app.run(debug=True, port=5000)
