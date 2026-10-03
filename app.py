import sys
import streamlit as st
st.write(f"Python Binary: `{sys.executable}`")
import xgboost
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.Airbnb.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set up visual configuration
st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 Airbnb Nightly Price Predictor")
st.markdown("Enter property details below to calculate an estimated nightly price using our optimized XGBoost machine learning model.")
st.markdown("---")

# Interface Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header("📍 Location & Type")
    city = st.selectbox("City", ["NYC", "LA", "SF", "DC", "Chicago", "Boston"])
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
    property_type = st.selectbox("Property Type", ["Apartment", "House", "Condominium", "Loft", "Townhouse", "Other"])
    latitude = st.number_input("Latitude", value=40.7128, format="%.4f")
    longitude = st.number_input("Longitude", value=-74.0060, format="%.4f")

with col2:
    st.header("🛏️ Property Details")
    accommodates = st.number_input("Accommodates (Guests)", min_value=1, max_value=16, value=2)
    bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    beds = st.number_input("Beds", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    basic_amenities_count = st.number_input("Basic Amenities Count", min_value=0, max_value=50, value=10)
    premium_amenities_count = st.number_input("Premium Amenities Count", min_value=0, max_value=20, value=2)

with col3:
    st.header("⭐ Host & Listing Info")
    cancellation_policy = st.selectbox("Cancellation Policy", ["flexible", "moderate", "strict", "super_strict_30", "super_strict_60"])
    review_scores_rating = st.slider("Review Score Rating (0-100)", min_value=0, max_value=100, value=95)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10)
    host_response_rate = st.slider("Host Response Rate (%)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

    st.markdown("**Check all that apply:**")
    has_reviews_input = st.checkbox("Listing Has Reviews?", value=True)
    host_has_profile_pic_input = st.checkbox("Host Has Profile Pic?", value=True)
    host_identity_verified_input = st.checkbox("Host Identity Verified?", value=True)
    instant_bookable_input = st.checkbox("Instant Bookable?", value=False)

# Convert boolean inputs to binary numerical values
has_reviews = 1 if has_reviews_input else 0
host_has_profile_pic = 1 if host_has_profile_pic_input else 0
host_identity_verified = 1 if host_identity_verified_input else 0
instant_bookable = 1 if instant_bookable_input else 0

st.markdown("---")

# Predict Button
if st.button("💰 Predict Nightly Price", type="primary", use_container_width=True):
    with st.spinner("Running XGBoost Algorithm..."):
        try:
            data = CustomData(
                accommodates=accommodates, bathrooms=bathrooms, bedrooms=bedrooms,
                beds=beds, latitude=latitude, longitude=longitude,
                number_of_reviews=number_of_reviews, review_scores_rating=review_scores_rating,
                basic_amenities_count=basic_amenities_count, premium_amenities_count=premium_amenities_count,
                host_response_rate=host_response_rate, has_reviews=has_reviews,
                host_has_profile_pic=host_has_profile_pic, host_identity_verified=host_identity_verified,
                instant_bookable=instant_bookable, room_type=room_type,
                property_type=property_type, cancellation_policy=cancellation_policy, city=city
            )

            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            st.success("Prediction Complete!")
            st.metric(label="Calculated Optimal Nightly Price", value=f"${results[0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
