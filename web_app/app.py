
from flask import Flask, request
from flask import render_template

import pickle 
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        x_dir = float(request.form["x_direction"])
        y_dir = float(request.form["y_direction"])
        z_dir = float(request.form["z_direction"])

        input_data = np.array([[x_dir, y_dir, z_dir]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            result = "Problem detected"
        else:
            result = "No problem"

    except Exception as e:
        result = f"Error: {e}"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
