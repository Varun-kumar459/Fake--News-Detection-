"""Flask web app + JSON API:  python app.py"""
import json

from flask import Flask, jsonify, render_template, request

import config
from src.predictor import predict

app = Flask(__name__)


def load_metrics():
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text())
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    result, text, error = None, "", None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if len(text.split()) < 5:
            error = "Paste at least a full sentence (5+ words) so the model has something to read."
        else:
            try:
                result = predict(text)
            except (FileNotFoundError, RuntimeError) as e:
                error = str(e)
    return render_template("index.html", result=result, text=text,
                           error=error, metrics=load_metrics())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text.split()) < 5:
        return jsonify(error="Field 'text' must contain at least 5 words."), 400
    return jsonify(predict(text))


if __name__ == "__main__":
    app.run(debug=True)
