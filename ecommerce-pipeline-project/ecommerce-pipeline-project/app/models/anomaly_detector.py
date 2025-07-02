import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'payment_value',
            'freight_value',
            'payment_installments',
            'product_weight_g',
            'product_length_cm',
            'product_height_cm',
            'product_width_cm',
            'delivery_time',
            'price_per_weight',
            'price_per_volume',
            'freight_ratio'
        ]
        
    def prepare_features(self, df):
        """Prepare features for anomaly detection"""
        # Calculate delivery time
        df['delivery_time'] = (
            pd.to_datetime(df['order_delivered_customer_date']) - 
            pd.to_datetime(df['order_purchase_timestamp'])
        ).dt.days
        
        # Calculate price per weight
        df['price_per_weight'] = df['price'] / df['product_weight_g']
        
        # Calculate price per volume
        df['product_volume'] = (
            df['product_length_cm'] * 
            df['product_height_cm'] * 
            df['product_width_cm']
        )
        df['price_per_volume'] = df['price'] / df['product_volume']
        
        # Calculate freight ratio
        df['freight_ratio'] = df['freight_value'] / df['price']
        
        # Fill missing values
        numeric_cols = self.feature_columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        return df
    
    def train(self, df, contamination=0.01):
        """Train the anomaly detection model"""
        # Prepare features
        df = self.prepare_features(df)
        
        # Scale features
        X = self.scaler.fit_transform(df[self.feature_columns])
        
        # Train model
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        self.model.fit(X)
        
        # Get anomaly scores
        scores = self.model.score_samples(X)
        
        # Calculate threshold
        threshold = np.percentile(scores, contamination * 100)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'order_id': df['order_id'],
            'anomaly_score': scores,
            'is_anomaly': predictions == -1,
            'threshold': threshold
        })
        
        return results
    
    def detect_anomalies(self, data):
        """Detect anomalies in new orders"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        data = self.prepare_features(data)
        
        # Scale features
        X = self.scaler.transform(data[self.feature_columns])
        
        # Get anomaly scores
        scores = self.model.score_samples(X)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'order_id': data['order_id'],
            'anomaly_score': scores,
            'is_anomaly': predictions == -1
        })
        
        return results
    
    def get_anomaly_details(self, data, anomaly_results):
        """Get detailed information about anomalies"""
        # Merge anomaly results with original data
        details = pd.merge(
            data,
            anomaly_results,
            on='order_id'
        )
        
        # Filter anomalies
        anomalies = details[details['is_anomaly']]
        
        # Calculate feature statistics for anomalies
        anomaly_stats = anomalies[self.feature_columns].describe()
        
        # Get top contributing features
        feature_contributions = pd.DataFrame({
            'feature': self.feature_columns,
            'contribution': np.abs(
                self.model.estimators_[0].feature_importances_
            )
        }).sort_values('contribution', ascending=False)
        
        return {
            'anomaly_details': anomalies,
            'anomaly_stats': anomaly_stats,
            'feature_contributions': feature_contributions
        }
    
    def save_model(self, path='models/anomaly_detector.joblib'):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path='models/anomaly_detector.joblib'):
        """Load the model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns'] 