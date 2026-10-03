# Airbnb Dynamic Price Prediction

A complete end-to-end machine learning pipeline and interactive web application designed to predict short-term rental dynamic pricing for Airbnb properties. 

Built with a focus on maintainability, this project features a modular architecture, custom data preprocessors, and a streamlined frontend for real-time price forecasting.

## Features
* **Interactive Web Interface:** A lightweight, user-friendly dashboard built entirely in Python using Streamlit.
* **High-Performance ML:** Utilizes an optimized XGBoost regression model for accurate price predictions based on property features and location data.
* **Modular Pipeline:** Structured with maintainable Python packages separating data ingestion, preprocessing, and prediction logic.
* **Cloud Ready:** Fully configured with `requirements.txt` for seamless, one-click deployment on Streamlit Community Cloud.

## Project Structure
```text
End-to-End-Airbnb-Price-Prediction/
├── artifacts/               # Serialized model (.pkl) and preprocessor objects
├── notebooks/               # Jupyter notebooks for EDA and model training (Airbnb_Setup.ipynb)
├── src/                     # Source code containing the custom data and prediction pipelines
├── app.py                   # Streamlit application entry point
├── requirements.txt         # Project dependencies for cloud deployment
└── setup.py                 # Local package configuration
