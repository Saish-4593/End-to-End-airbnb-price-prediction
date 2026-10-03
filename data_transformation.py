import os
import sys
import pandas as pd
import numpy as np
import pickle
import re
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.Airbnb.logger import logging
from src.Airbnb.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    pricing_data_path: str = os.path.join('artifacts', 'cleaned_data_pricing.csv')
    ratings_data_path: str = os.path.join('artifacts', 'cleaned_data_ratings.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        self.basic_essentials = [
            'Wireless Internet', 'Air conditioning', 'Kitchen', 
            'TV', 'Heating', 'Washer', 'Dryer'
        ]
        self.premium_essentials = [
            'Free parking on premises', 'Elevator', 'Indoor fireplace', 
            'Pets allowed', 'Self Check-In', '24-hour check-in', 
            'Laptop friendly workspace', 'Hot tub', 'Pool', 'Gym'
        ]

    def clean_amenities(self, x):
        if pd.isna(x):
            return []
        x = x.replace('{', '').replace('}', '').replace('"', '')
        x = re.sub(r'translation missing:[^,]+', '', x)
        return [a.strip() for a in x.split(',') if a.strip()]

    def custom_cleaning_pipeline(self, df, knn_model=None, is_train=True):
        logging.info("Applying mode imputation for bedrooms, bathrooms, and beds.")
        df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].mode()[0])
        df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mode()[0])
        df['beds'] = df['beds'].fillna(df['beds'].mode()[0])

        logging.info("Extracting basic and premium amenities.")
        df['amenities_list'] = df['amenities'].apply(self.clean_amenities)
        df['basic_amenities_count'] = df['amenities_list'].apply(
            lambda x: sum(1 for item in x if item in self.basic_essentials)
        )
        df['premium_amenities_count'] = df['amenities_list'].apply(
            lambda x: sum(1 for item in x if item in self.premium_essentials)
        )
        df = df.drop(columns=['amenities', 'amenities_list'])

        logging.info("Applying Spatial KNN for missing neighbourhoods.")
        if is_train:
            known_df = df[df['neighbourhood'].notnull()]
            X_train = known_df[['latitude', 'longitude']]
            y_train = known_df['neighbourhood']

            knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
            knn_model.fit(X_train, y_train)

        unknown_mask = df['neighbourhood'].isnull()
        if unknown_mask.sum() > 0:
            X_test = df.loc[unknown_mask, ['latitude', 'longitude']]
            predicted_neighbourhoods = knn_model.predict(X_test)
            df.loc[unknown_mask, 'neighbourhood'] = predicted_neighbourhoods

        logging.info("Dropping defined columns and null rows.")
        columns_to_drop = ['thumbnail_url', 'first_review', 'last_review', 'zipcode']
        df = df.drop(columns=columns_to_drop)
        df = df.dropna(subset=['host_has_profile_pic', 'host_identity_verified', 'host_since'])

        logging.info("Formatting host metrics.")
        df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', '')
        df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
        df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].median())

        binary_mapping = {'t': 1, 'f': 0}
        df['host_has_profile_pic'] = df['host_has_profile_pic'].map(binary_mapping).fillna(0)
        df['host_identity_verified'] = df['host_identity_verified'].map(binary_mapping).fillna(0)
        df['instant_bookable'] = df['instant_bookable'].map(binary_mapping).fillna(0)

        return df, knn_model

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'accommodates', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude', 
                'number_of_reviews', 'review_scores_rating', 'basic_amenities_count', 
                'premium_amenities_count', 'host_response_rate', 'has_reviews',
                'host_has_profile_pic', 'host_identity_verified', 'instant_bookable'
            ]

            categorical_columns = ['room_type', 'property_type', 'cancellation_policy', 'city']

            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

            cat_pipeline = Pipeline(steps=[
                ("one_hot_encoder", OneHotEncoder(drop='first', handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                sparse_threshold=0  # THE FIX: Forces dense array output for np.c_
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Applying custom logic to Train Set")
            train_df, trained_knn = self.custom_cleaning_pipeline(train_df, is_train=True)

            logging.info("Applying custom logic to Test Set")
            test_df, _ = self.custom_cleaning_pipeline(test_df, knn_model=trained_knn, is_train=False)

            df_ratings = train_df.copy()
            df_ratings = df_ratings.dropna(subset=['review_scores_rating'])
            df_ratings = df_ratings[df_ratings['number_of_reviews'] >= 3]
            os.makedirs(os.path.dirname(self.data_transformation_config.ratings_data_path), exist_ok=True)
            df_ratings.to_csv(self.data_transformation_config.ratings_data_path, index=False)

            def prepare_pricing(df):
                df_pricing = df.copy()
                df_pricing['has_reviews'] = (df_pricing['number_of_reviews'] > 0).astype(int)
                df_pricing['review_scores_rating'] = df_pricing['review_scores_rating'].fillna(df_pricing['review_scores_rating'].median())
                columns_to_drop_for_ml = ['id', 'name', 'description', 'host_since', 'neighbourhood']
                df_pricing = df_pricing.drop(columns=columns_to_drop_for_ml, errors='ignore')
                return df_pricing

            train_pricing = prepare_pricing(train_df)
            test_pricing = prepare_pricing(test_df)

            train_pricing.to_csv(self.data_transformation_config.pricing_data_path, index=False)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "log_price"

            input_feature_train_df = train_pricing.drop(columns=[target_column_name])
            target_feature_train_df = train_pricing[target_column_name]

            input_feature_test_df = test_pricing.drop(columns=[target_column_name])
            target_feature_test_df = test_pricing[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as file_obj:
                pickle.dump(preprocessing_obj, file_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
