import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check karega ki frontend.html kahan rakha hai (.vscode/templates ya templates)
if os.path.exists(os.path.join(BASE_DIR, ".vscode", "templates", "frontend.html")):
    TEMPLATE_DIR = os.path.join(BASE_DIR, ".vscode", "templates")
elif os.path.exists(os.path.join(BASE_DIR, "templates", "frontend.html")):
    TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
else:
    TEMPLATE_DIR = BASE_DIR

# Check karega ki style.css kahan rakhi hai (.vscode/static ya static)
if os.path.exists(os.path.join(BASE_DIR, ".vscode", "static")):
    STATIC_DIR = os.path.join(BASE_DIR, ".vscode", "static")
else:
    STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

with open(os.path.join(BASE_DIR, "vectrayield_model.pkl"), "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
threshold = bundle["threshold"]
features = bundle["feature_names"]
medians = bundle["medians"]
top_sensors = features[:6]

@app.route("/")
def home():
    default_vals = {s: round(medians[s], 2) for s in top_sensors}
    return render_template("frontend.html", sensors=default_vals, threshold=round(threshold * 100, 2))

@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.json
    wafer_data = medians.copy()
    
    for sensor, val in data.items():
        if sensor in wafer_data and val != "":
            wafer_data[sensor] = float(val)
            
    df_input = pd.DataFrame([wafer_data])[features]
    fail_prob = float(pipeline.predict_proba(df_input)[:, 1][0])
    status = "FAIL" if fail_prob >= threshold else "PASS"
    
    return jsonify({
        "probability": round(fail_prob * 100, 2),
        "threshold": round(threshold * 100, 2),
        "status": status
    })

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    df = pd.read_csv(file)
    
    for col in features:
        if col not in df.columns:
            df[col] = medians[col]
            
    df_ordered = df[features]
    probs = pipeline.predict_proba(df_ordered)[:, 1]
    preds = np.where(probs >= threshold, "FAIL", "PASS")
    
    results = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        results.append({
            "wafer_id": f"Wafer #{i + 1}",
            "risk": round(float(prob) * 100, 2),
            "status": pred
        })
        
    return jsonify({
        "total": len(results),
        "failed": int(np.sum(preds == "FAIL")),
        "passed": int(np.sum(preds == "PASS")),
        "results": results
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)