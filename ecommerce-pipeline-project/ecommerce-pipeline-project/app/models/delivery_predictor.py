import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class DeliveryPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.numeric_features = [
            'price',
            'freight_value',
            'product_weight_g',
            'product_length_cm',
            'product_height_cm',
            'product_width_cm',
            'payment_installments',
            'payment_value'
        ]
        
        self.categorical_features = [
            'product_category_name',
            'seller_state',
            'customer_state',
            'payment_type'
        ]
        
    def prepare_features(self, df):
        """Prepare features for delivery time prediction"""
        # Calculate delivery time
        df['delivery_time'] = (
            pd.to_datetime(df['order_delivered_customer_date']) - 
            pd.to_datetime(df['order_purchase_timestamp'])
        ).dt.days
        
        # Add day of week
        df['order_day'] = pd.to_datetime(df['order_purchase_timestamp']).dt.dayofweek
        
        # Add month
        df['order_month'] = pd.to_datetime(df['order_purchase_timestamp']).dt.month
        
        # Fill missing values
        numeric_cols = self.numeric_features
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
    
    def train(self, df):
        """Train the delivery time prediction model"""
        # Prepare data
        df = self.prepare_features(df)
        
        # Create preprocessing pipeline
        self.create_preprocessor()
        
        # Prepare features and target
        X = df[self.numeric_features + self.categorical_features]
        y = df['delivery_time']
        
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
    
    def predict_delivery_time(self, data):
        """Predict delivery time for new orders"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        data = self.prepare_features(data)
        
        # Make predictions
        predictions = self.model.predict(data[self.numeric_features + self.categorical_features])
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get feature names after preprocessing
        feature_names = (
            self.numeric_features +
            self.model.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(self.categorical_features)
            .tolist()
        )
        
        # Get feature importances
        importances = self.model.named_steps['regressor'].feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path='models/delivery_predictor.joblib'):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    def load_model(self, path='models/delivery_predictor.joblib'):
        """Load the model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = joblib.load(path)
        self.create_preprocessor()  # Recreate preprocessor 