# 🏨 Hotel Reservation Cancellation Predictor (ANN)

An Artificial Neural Network (ANN) that predicts whether a hotel booking will be cancelled, deployed as a Streamlit web app.


## 📌 Problem Statement

Hotels lose revenue when guests cancel at the last minute. This project predicts the cancellation risk of a booking from its details (lead time, price, market segment, special requests, etc.), so the hotel can act early with a deposit, a confirmation call, or overbooking plans.

This is a **binary classification** problem: `Canceled (1)` vs `Not_Canceled (0)`.

## 📊 Dataset

- **File:** `Hotel_Reservations.csv`
- **Size:** 36,275 bookings, 19 columns
- **Target:** `booking_status`
- **Class split:** ~67% not cancelled, ~33% cancelled
- **Features include:** number of adults/children, weekend/week nights, meal plan, room type, car parking, lead time, arrival date, market segment, repeated guest, previous cancellations, average room price, special requests

## 🔄 Project Workflow

1. **EDA:** distributions, correlations, class balance, VIF check (Matplotlib, Seaborn)
2. **Cleaning:** fixed data types, dropped `Booking_ID`, checked missing values and duplicates
3. **Outlier treatment:** IQR capping on `lead_time` and `avg_price_per_room`
4. **Preprocessing:** log transform + StandardScaler for numeric columns, One-Hot Encoding for categorical columns (27 final features)
5. **Split:** 80/20 stratified train-test split
6. **Model:** Keras ANN with class weights to handle imbalance
7. **Tuning:** Optuna (25 trials) for learning rate and batch size
8. **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
9. **Deployment:** Streamlit app

## 🧠 Model Architecture

| Layer | Details |
|---|---|
| Input | 27 features |
| Dense | 64 units, ReLU, Dropout 0.2 |
| Dense | 32 units, ReLU, Dropout 0.2 |
| Dense | 16 units, ReLU |
| Output | 1 unit, Sigmoid |

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam (learning rate 0.001)
- **Training:** batch size 64, up to 100 epochs, Early Stopping (patience 10), balanced class weights

## ✅ Results (Test Set)

| Metric | Score |
|---|---|
| Accuracy | 86.1% |
| Precision | 75.9% |
| Recall | 84.4% |
| F1-score | 0.80 |
| ROC-AUC | 0.936 |

- Train accuracy is 86.7% vs test accuracy 86.1%, so there is no significant overfitting.
- High recall means the model catches most bookings that actually get cancelled.
- Optuna tuning gave almost the same score, which shows the baseline model was already close to what this data can give.

## 🖥️ Streamlit App

The app lets a user enter booking details in the sidebar and then shows:

- Cancellation probability (with a progress bar)
- Risk level (above 50% = High Risk, below 50% = Low Risk)
- Suggested next actions for the hotel team
- The exact input sent to the model

<!-- Add a screenshot of your app here -->
<!-- ![App Screenshot](screenshots/app.png) -->

## 📁 Project Structure

```
├── app.py                         # Streamlit app
├── Hotel_Cancellation_ANN.ipynb   # Full notebook: EDA, training, evaluation
├── Hotel_Reservations.csv         # Dataset
├── artifacts/
│   ├── hotel_cancellation_ann.keras   # Trained model
│   └── preprocessor.joblib            # Fitted preprocessing pipeline
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Usage

```bash
# 1. Clone the repository
git clone https://github.com/Rudra-323/Hotel-Reservation-Cancellation-Prediction.git
cd Hotel-Reservation-Cancellation-Prediction

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Make sure the `artifacts/` folder contains `hotel_cancellation_ann.keras` and `preprocessor.joblib`. These are created by running the notebook.

### Suggested `requirements.txt`

```
tensorflow
scikit-learn
pandas
numpy
joblib
streamlit
matplotlib
seaborn
statsmodels
optuna
```

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · TensorFlow/Keras · Optuna · Streamlit

## ⚠️ Limitations

- The data covers only 2017–2018, so guest behaviour may have changed.
- The ANN is a black box and does not explain why a booking is risky.
- A fixed 50% threshold may not suit every hotel's cost of a false alarm vs a missed cancellation.
- Models like XGBoost were not compared.

## 🚀 Future Scope

- Compare with XGBoost / LightGBM
- Add SHAP for explainable predictions
- Add probability calibration and an adjustable risk threshold
- Train on more recent data from more hotels
- Deploy online (Streamlit Cloud / Docker) with an API and monitoring

## 👤 Author

**Rudrakanta Mandala**
[LinkedIn](https://www.linkedin.com/in/rudrakanta)) · [GitHub](https://github.com/Rudra-323)

Built as part of the ANN Capstone Project at Innomatics Research Labs.
