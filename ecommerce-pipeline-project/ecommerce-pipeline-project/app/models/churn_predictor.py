import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'recency',
            'frequency',
            'monetary',
            'satisfaction',
            'product_diversity',
            'avg_delivery_time',
            'days_since_last_order',
            'order_frequency',
            'avg_order_value',
            'return_rate'
        ]
        
    def prepare_features(self, df):
        """Prepare features for churn prediction"""
        # Calculate days since last order
        latest_date = df['order_purchase_timestamp'].max()
        df['days_since_last_order'] = (latest_date - df['order_purchase_timestamp']).dt.days
        
        # Calculate order frequency (orders per month)
        df['order_frequency'] = df.groupby('customer_id')['order_id'].transform('count') / 12
        
        # Calculate average order value
        df['avg_order_value'] = df.groupby('customer_id')['payment_value'].transform('mean')
        
        # Calculate return rate (if available)
        if 'order_status' in df.columns:
            df['return_rate'] = df.groupby('customer_id')['order_status'].transform(
                lambda x: (x == 'delivered').mean()
            )
        else:
            df['return_rate'] = 0
        
        # Group by customer and calculate features
        features = df.groupby('customer_id').agg({
            'days_since_last_order': 'min',
            'order_frequency': 'first',
            'avg_order_value': 'first',
            'return_rate': 'first',
            'review_score': 'mean',
            'product_category_name': 'nunique'
        }).rename(columns={
            'review_score': 'satisfaction',
            'product_category_name': 'product_diversity'
        })
        
        # Add RFM features
        features['recency'] = features['days_since_last_order']
        features['frequency'] = features['order_frequency']
        features['monetary'] = features['avg_order_value']
        
        # Calculate average delivery time
        delivery_time = df.groupby('customer_id').apply(
            lambda x: (x['order_delivered_customer_date'] - x['order_purchase_timestamp']).mean().days
        )
        features['avg_delivery_time'] = delivery_time
        
        return features
    
    def define_churn(self, features, churn_threshold_days=90):
        """Define churn based on days since last order"""
        features['churn'] = (features['days_since_last_order'] > churn_threshold_days).astype(int)
        return features
    
    def train(self, df, churn_threshold_days=90):
        """Train the churn prediction model"""
        # Prepare features
        features = self.prepare_features(df)
        features = self.define_churn(features, churn_threshold_days)
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Prepare X and y
        X = features[self.feature_columns]
        y = features['churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def predict_churn(self, customer_id, df):
        """Predict churn probability for a specific customer"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features for the customer
        features = self.prepare_features(df)
        if customer_id not in features.index:
            raise ValueError(f"Customer {customer_id} not found in the data")
        
        # Get customer features
        customer_features = features.loc[customer_id, self.feature_columns].values.reshape(1, -1)
        
        # Scale features
        customer_features_scaled = self.scaler.transform(customer_features)
        
        # Predict churn probability
        churn_probability = self.model.predict_proba(customer_features_scaled)[0, 1]
        
        return {
            'churn_probability': churn_probability,
            'features': dict(zip(self.feature_columns, customer_features[0]))
        }
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save_model(self, path='models/churn_predictor.joblib'):
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
    
    def load_model(self, path='models/churn_predictor.joblib'):
        """Load the model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns'] 