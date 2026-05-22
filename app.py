from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    passenger_count = int(request.form["passenger_count"])
    hour = int(request.form["hour"])
    day = int(request.form["day"])
    month = int(request.form["month"])
    distance_km = float(request.form["distance_km"])

    features = np.array([[passenger_count, hour, day, month, distance_km]])

    prediction = model.predict(features)[0]

    usd = round(prediction, 2)

    inr = round(usd * 83, 2)
    eur = round(usd * 0.92, 2)
    gbp = round(usd * 0.79, 2)

    result = f"""
    💵 USD: ${usd} <br>
    🇮🇳 INR: ₹{inr} <br>
    🇪🇺 EUR: €{eur} <br>
    🇬🇧 GBP: £{gbp}
    """

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)