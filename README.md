# Kidney-Disease-Detection

A user-friendly **Streamlit** web application that predicts **Chronic Kidney Disease (CKD)** based on patient health metrics using a machine learning model.

---

## 🧠 Features

- Predicts risk of Chronic Kidney Disease using clinical inputs.
- Interactive user interface built with Streamlit.
- Lightweight, fast, and easy to deploy.
- Based on a trained Scikit-learn classification model.

---

## 📦 Tech Stack

- **Python**: programming language
- **Scikit-Learn**: model training and evaluation
- **Pandas, NumPy**: data manipulation
- **Streamlit**: web interface

---

## 🚀 How to Run the Project

Follow these steps to set up and run the application:

```bash
# Step 1: Clone the repository
git clone https://github.com/Raghava-Ram/Kidney-disease-detection.git
cd Kidney-disease-detection

# Step 2: (Optional but recommended) Create and activate a virtual environment

# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Step 3: Install required dependencies
pip install streamlit scikit-learn pandas numpy

# Step 4: Run the Streamlit app
streamlit run app.py
