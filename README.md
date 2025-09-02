# 🏡 Housing Price Prediction

Ever wondered how much a house might cost in California?  
This project is a **machine learning pipeline** that takes real housing data and trains a model to predict home prices. It’s a mix of **data preprocessing, model training, and prediction automation** — all in one script.

---

## ✨ What this project does
- Cleans and prepares the raw housing data (`housing.csv`).
- Splits the data smartly (so income groups are fairly represented).
- Handles missing values and scales the numbers so the model doesn’t get confused.
- Turns categorical values (like “ocean proximity”) into useful features for the model.
- Trains a **Random Forest Regressor** — a powerful algorithm that works great for tabular data.
- Saves everything (the trained model + preprocessing pipeline) so you don’t have to retrain every time.
- Lets you drop in new data (`input.csv`) and get predictions instantly in `output.csv`.

---

## 🛠️ Tools & Tech
- **Python**
- **scikit-learn** (for ML magic ✨)
- **pandas & numpy** (for data wrangling)
- **joblib** (to save and reload your models easily)

---

## 🚀 How to use it

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Get the dataset

Place housing.csv in the project folder.

### 3️⃣ Train the model (first run)
```bash
python main.py
```
The script will train the model and save:

- model.pkl → the trained Random Forest model

- pipeline.pkl → the preprocessing pipeline

- input.csv → a test dataset to play with

You’ll see:
```
 model is trained.
```
### 4️⃣ Make predictions (next runs)

On later runs, it will skip training and instead:

- Load the saved model and pipeline

- Transform input.csv

- Write predictions to output.csv

  You’ll see:
  ```
  Results saved to output.csv
  ```
 ---
## 📂 Project Layout
```
├── main.py          # The script that runs everything
├── housing.csv      # Input dataset (you provide this)
├── model.pkl        # Trained Random Forest model
├── pipeline.pkl     # Preprocessing steps
├── input.csv        # Auto-generated test data
├── output.csv       # Predictions get saved here
├── requirements.txt # Dependencies
```
---

## 🌱 What you can add more

- Add more models (Linear Regression, Decision Tree, etc.) to compare.

- Tune hyperparameters for even better accuracy.

- Wrap this into a simple web app (Flask/FastAPI) so anyone can try it out.

---


