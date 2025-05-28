# import packages and libraries
import numpy as np
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# define function to clean data
def preprocess_data(df):
    # drop irrelevant columns
    df = df.drop(['id', 'name', 'host_name', 'last_review'], axis=1)

    # create binary indicator for presence of reviews
    df['has_reviews'] = df['reviews_per_month'].notnull().astype(int)

    # apply lof transformation for target variable
    df['log_price'] = np.log1p(df['price'])

    # define target and feature variables
    X = df.drop(['price', 'log_price'], axis=1)
    y = df['log_price']

    # define feature types
    categorical = ['neighbourhood_group', 'neighbourhood', 'room_type']
    numerical = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
             'calculated_host_listings_count', 'availability_365', 'has_reviews']

    # numerical transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # full preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ])

    # train-test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return preprocessor, X_train, X_test, y_train, y_test
