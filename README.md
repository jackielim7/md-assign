# Obesity Classification App

A Streamlit web application that predicts an individual’s **obesity level** based on personal, lifestyle, and physical condition data such as diet habits, daily activity, and transportation mode.  
This project was developed as part of my **Model Deployment course assignment** at **BINUS University**.  
The backend model was trained using **XGBoost**, with complete preprocessing, encoding, and deployment pipeline.

---

## Demo App
You can try the live demo here:  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://md-assign-atpk4x4z2zceyrzcdrrku2.streamlit.app/)

---

## Project Overview

The goal of this project is to predict **obesity categories** (e.g., Normal, Overweight, Obese) based on user-provided input.  
The overall workflow includes:

1. **Data Preprocessing** – handling categorical features, scaling, and feature transformation.  
2. **Feature Encoding** – applying:
   - Label encoding for binary categories  
   - Ordinal encoding for ordered categories (e.g., CAEC, CALC)  
   - One-hot encoding for transportation modes  
3. **Model Training** – using **XGBoost Classifier** for multi-class classification.  
4. **Model Evaluation** – testing model performance on accuracy and classification metrics.  
5. **Model Deployment** – deploying the model using **Streamlit**, loading the `.pkl` model, and visualizing predictions interactively.

---

## Saved Files
| File Name             | Description                                |
| ---------------------- | ------------------------------------------ |
| `EDA-Train.ipynb`  | Jupyter Notebook for model training        |
| `trained_model.pkl`    | Trained and serialized XGBoost model       |
| `streamlit_app.py`               | Streamlit web app for deployment           |

---

## Author
**Jackie Lim**  
📧 [linkedin.com/in/jackie-lim7/](https://linkedin.com/in/jackie-lim7/)  
🎓 Model Deployment Course — BINUS University  
