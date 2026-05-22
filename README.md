📄 🚖 CAB FARE PREDICTION – PROJECT REPORT
1. 🧠 Project Title

Cab Fare Prediction using Machine Learning and Flask Web App

2. 🎯 Objective

Is project ka goal hai:

Cab fare ko predict karna using machine learning
Input features jaise:
Passenger count
Time (hour, day, month)
Distance (KM)
Output: Estimated fare in multiple currencies (USD, INR, EUR, GBP)




3. 📊 Dataset Description

Dataset me mainly ye features use kiye gaye:

pickup_datetime
pickup_latitude
pickup_longitude
dropoff_latitude
dropoff_longitude
passenger_count
fare_amount (target variable)



4. 🧹 Data Preprocessing

Steps:

Missing values remove kiye
Duplicate rows delete kiye
Invalid fare values filter kiye
Datetime ko convert karke:
hour
day
month extract kiya
Distance calculate kiya Haversine formula se



5. ⚙️ Feature Engineering

New features created:

Hour of ride
Day of month
Month
Distance (KM between pickup and drop)



6. 🤖 Model Used
Linear Regression (basic ML model)
Input → Features
Output → Fare prediction


7. 🌐 Web App (Flask)

Tech stack:

Flask (backend)
HTML + CSS (frontend)
Leaflet.js (map integration)
NumPy (processing)

Features:

User input form
Live map for pickup/drop
Auto distance calculation
Fare prediction output


8. 💱 Currency Conversion

Predicted fare convert kiya gaya:

USD (base)
INR (×83)
EUR (×0.92)
GBP (×0.79)


9. 📦 Project Structure
cab_fare_prediction/
│
├── app.py
├── model.pkl
├── data/
├── src/
├── templates/
├── requirements.txt


10. 🚀 Results
Accurate fare prediction system
Interactive UI with map
Real-time distance calculation
Multi-currency output


11. 🔮 Future Improvements
Use XGBoost / Random Forest for better accuracy
Add real-time traffic data
User login system
Database for storing history
Deploy on cloud (Render)


12. 🏁 Conclusion

This project demonstrates end-to-end Machine Learning workflow:

Data preprocessing
Feature engineering
Model training
Web deployment

It is a real-world AI product prototype similar to Uber/Ola fare estimation systems.