import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.Airbnb.exception import CustomException
from src.Airbnb.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Load the saved model and preprocessor rules
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

            # Clean and scale the user's input data
            data_scaled = preprocessor.transform(features)

            # Make the prediction (this outputs a log_price)
            log_prediction = model.predict(data_scaled)

            # Reverse the logarithm using np.exp() to get real dollars
            actual_price_prediction = np.exp(log_prediction)

            return actual_price_prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        accommodates: int,
        bathrooms: float,
        bedrooms: float,
        beds: float,
        latitude: float,
        longitude: float,
        number_of_reviews: int,
        review_scores_rating: float,
        basic_amenities_count: int,
        premium_amenities_count: int,
        host_response_rate: float,
        has_reviews: int,
        host_has_profile_pic: int,
        host_identity_verified: int,
        instant_bookable: int,
        room_type: str,
        property_type: str,
        cancellation_policy: str,
        city: str
    ):
        self.accommodates = accommodates
        self.bathrooms = bathrooms
        self.bedrooms = bedrooms
        self.beds = beds
        self.latitude = latitude
        self.longitude = longitude
        self.number_of_reviews = number_of_reviews
        self.review_scores_rating = review_scores_rating
        self.basic_amenities_count = basic_amenities_count
        self.premium_amenities_count = premium_amenities_count
        self.host_response_rate = host_response_rate
        self.has_reviews = has_reviews
        self.host_has_profile_pic = host_has_profile_pic
        self.host_identity_verified = host_identity_verified
        self.instant_bookable = instant_bookable
        self.room_type = room_type
        self.property_type = property_type
        self.cancellation_policy = cancellation_policy
        self.city = city

    def get_data_as_data_frame(self):
        try:
            # Package the user's inputs into a dictionary
            custom_data_input_dict = {
                "accommodates": [self.accommodates],
                "bathrooms": [self.bathrooms],
                "bedrooms": [self.bedrooms],
                "beds": [self.beds],
                "latitude": [self.latitude],
                "longitude": [self.longitude],
                "number_of_reviews": [self.number_of_reviews],
                "review_scores_rating": [self.review_scores_rating],
                "basic_amenities_count": [self.basic_amenities_count],
                "premium_amenities_count": [self.premium_amenities_count],
                "host_response_rate": [self.host_response_rate],
                "has_reviews": [self.has_reviews],
                "host_has_profile_pic": [self.host_has_profile_pic],
                "host_identity_verified": [self.host_identity_verified],
                "instant_bookable": [self.instant_bookable],
                "room_type": [self.room_type],
                "property_type": [self.property_type],
                "cancellation_policy": [self.cancellation_policy],
                "city": [self.city],
            }

            # Convert dictionary to a Pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
