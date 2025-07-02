import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import streamlit as st

class ReviewPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = [
            'price',
            'freight_value',
            'product_name_lenght',
            'product_description_lenght',
            'product_photos_qty',
            'product_weight_g',
            'product_length_cm',
            'product_height_cm',
            'product_width_cm',
            'payment_sequential',
            'payment_installments',
            'payment_value',
            'delivery_days'
        ]
        
        self.categorical_columns = [
            'product_category_name',
            'seller_state',
            'customer_state',
            'payment_type'
        ]

    def prepare_features(self, df):
        """Prepare features for the model"""
        # Calculate delivery days
        df['delivery_days'] = (
            pd.to_datetime(df['order_delivered_customer_date']) - 
            pd.to_datetime(df['order_purchase_timestamp'])
        ).dt.days

        # Fill missing values
        df['delivery_days'] = df['delivery_days'].fillna(df['delivery_days'].median())
        df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].median())
        df['product_length_cm'] = df['product_length_cm'].fillna(df['product_length_cm'].median())
        df['product_height_cm'] = df['product_height_cm'].fillna(df['product_height_cm'].median())
        df['product_width_cm'] = df['product_width_cm'].fillna(df['product_width_cm'].median())

        return df

    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        numeric_features = self.feature_columns
        categorical_features = self.categorical_columns

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def train(self, df):
        """Train the model"""
        # Prepare data
        df = self.prepare_features(df)
        
        # Create preprocessing pipeline
        self.create_preprocessor()
        
        # Prepare features and target
        X = df[self.feature_columns + self.categorical_columns]
        y = df['review_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }

    def predict(self, data):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        data = self.prepare_features(data)
        
        # Make predictions
        predictions = self.model.predict(data[self.feature_columns + self.categorical_columns])
        
        return predictions

    def save_model(self, path='models/review_predictor.joblib'):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path='models/review_predictor.joblib'):
        """Load the model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            self.model = joblib.load(path)
            self.create_preprocessor()  # Recreate preprocessor 
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    def load_data(self, connection_string):
        """Load data from the database"""
        try:
            engine = create_engine(
                connection_string,
                pool_pre_ping=True,  # Vérifie la santé des connexions
                pool_recycle=3600,   # Recyclage des connexions après 1 heure
                pool_size=5,         # Nombre maximum de connexions
                max_overflow=10      # Connexions supplémentaires autorisées
            )
            with engine.connect() as connection:
                query = "SELECT * FROM transformed_data"
                df = pd.read_sql(query, connection)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur

    def dispose(self):
        """Dispose of the model and resources"""
        try:
            self.model = None
            self.preprocessor = None
        except:
            pass 