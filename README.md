# 🤖 Capstone Project – Customer Satisfaction (CSAT) Prediction using Deep Learning

This project focuses on predicting **Customer Satisfaction (CSAT)** scores using a Deep Learning Artificial Neural Network (ANN). The goal is to enhance service quality and customer retention by forecasting satisfaction levels based on support interactions in an e-commerce setting.

---

## 📌 Project Summary

Traditional survey-based methods of measuring CSAT can be slow and limited. In this project, we use historical customer support data from Shopzilla's e-commerce platform and apply deep learning techniques to **predict whether a customer is satisfied or not** based on interaction features.

---

## 🎯 Objectives

- Predict CSAT scores (binary: **Satisfied = 5**, **Not Satisfied < 5**)
- Apply custom feature engineering and preprocessing
- Build and train a Deep Learning model using TensorFlow/Keras
- Deploy model using Flask API for real-time predictions

---

## 🧠 Technologies Used

- Python 3.x
- Pandas, NumPy
- Scikit-learn (`Pipeline`, `ColumnTransformer`)
- TensorFlow / Keras
- Matplotlib & Seaborn (for visualization)
- Flask (for deployment)
- Git, GitHub

---

## 📂 Project Structure

├── templates/ # Flask HTML templates (if using frontend)
├── .gitignore # Ignoring unnecessary files like env folders
├── CSAT.ipynb # Notebook for EDA, model training & analysis
├── README.md # Project documentation
├── agent_stats.csv # Agent performance stats (for feature engineering)
├── supervisor_stats.csv # Supervisor performance stats
├── app.py # Flask API code
├── csat_model.h5 # Trained deep learning model (HDF5 format)
├── csat_model.keras # Alternate Keras model format
├── csat_pipeline.pkl # Preprocessing pipeline for training
├── inference_preprocessor.py # Script for preprocessing during inference
├── eCommerce_Customer_support_data.csv # Raw dataset
├── feature_columns.pkl # Selected feature names used during training
├── label_encoder.pkl # Trained LabelEncoder for 'Sub-category'
├── robust_scaler.pkl # RobustScaler for skewed features
├── standard_scaler.pkl # StandardScaler for normal features

## 🧪 Features & Engineering

The following features are used to train the model:

- **Categorical:** Channel name, Category, Sub-category, Agent Name, Shift, Supervisor, Manager
- **Numerical:** Item price, Connected handling time, Agent/Supervisor stats
- **Datetime-derived:** Response time, Issue day of week, Issue hour, Order-issue gap
- **Textual:** Customer Remarks (optional for future embedding)

---

## ⚙️ Model Architecture (Keras)

- Input Layer with concatenated numeric and embedded categorical features
- Embedding Layer for high-cardinality categorical variables (like Sub-category)
- Dense layers with ReLU activation
- Dropout for regularization

## 🧪 Model Evaluation

- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- Training vs Validation Loss plots

---

## 📈 Results & Insights

- Embedding layers improved performance for high-cardinality features
- Handling class imbalance using `class_weight` was essential
- Final model achieved promising results on test data

Mritunjay Mishra
Data Scientist | Python Developer | Educator
📍 Faridabad, India
