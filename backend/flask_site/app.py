import os, requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about-breast-cancer")
def about_bc():
    return render_template("about_bc.html")

@app.route("/about-us")
def about_us():
    return render_template("about_us.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/try")
def try_it_out():
    return render_template("model.html")

@app.post("/ml/predict")
def ml_predict():
    data = request.get_json() or {}
    try:
        r = requests.post(f"{API_BASE}/predict", json=data, timeout=10)
        return (r.text, r.status_code, r.headers.items())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502

@app.post("/ml/predict_csv")
def ml_predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400
    f = request.files["file"]
    try:
        r = requests.post(f"{API_BASE}/predict_csv", files={"file": (f.filename, f.stream, f.mimetype)}, timeout=30)
        return (r.text, r.status_code, r.headers.items())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502

if __name__ == "__main__":
    app.run(port=5000, debug=True)
