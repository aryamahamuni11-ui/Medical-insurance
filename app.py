from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        'age': int(request.form['age']),
        'sex': request.form['sex'],
        'bmi': float(request.form['bmi']),
        'children': int(request.form['children']),
        'smoker': request.form['smoker'],
        'region': request.form['region']
    }

    df = pd.DataFrame([data])
    
    log_pred = model.predict(df)
    prediction = np.expm1(log_pred)

    return render_template("index.html", result=round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)