from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get form data
        passenger_count = int(request.form.get("passenger_count"))
        hour = int(request.form.get("hour"))
        day = int(request.form.get("day"))
        month = int(request.form.get("month"))
        distance_km = float(request.form.get("distance_km"))

        # Create feature array
        features = np.array([[passenger_count, hour, day, month, distance_km]])

        # 🔥 DEBUG (check values)
        print("INPUT FEATURES:", features)

        # Predict
        prediction = model.predict(features)[0]

        # 🔥 DEBUG (check output)
        print("PREDICTION:", prediction)

        # Currency conversion
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

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)