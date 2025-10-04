from flask import Flask
from predict import bp as predict_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from your React dev server

app.register_blueprint(predict_bp)

@app.route("/")
def home():
    return "Backend running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)