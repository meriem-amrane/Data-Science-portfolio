import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

class CustomerSegmentation:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.rfm_data = None
        self.cluster_data = None
        
    def calculate_rfm(self, df):
        """Calculate RFM metrics for each customer"""
        # Get the latest date in the dataset
        latest_date = df['order_purchase_timestamp'].max()
        
        # Calculate RFM metrics
        rfm = df.groupby('customer_id').agg({
            'order_purchase_timestamp': lambda x: (latest_date - x.max()).days,  # Recency
            'order_id': 'count',  # Frequency
            'payment_value': 'sum',  # Monetary
            'review_score': 'mean'  # Average satisfaction
        }).rename(columns={
            'order_purchase_timestamp': 'recency',
            'order_id': 'frequency',
            'payment_value': 'monetary',
            'review_score': 'satisfaction'
        })
        
        # Calculate product diversity (number of unique categories)
        product_diversity = df.groupby('customer_id')['product_category_name'].nunique()
        rfm['product_diversity'] = product_diversity
        
        # Calculate average delivery time
        delivery_time = df.groupby('customer_id').apply(
            lambda x: (x['order_delivered_customer_date'] - x['order_purchase_timestamp']).mean().days
        )
        rfm['avg_delivery_time'] = delivery_time
        
        # Handle missing values
        rfm = rfm.fillna({
            'recency': rfm['recency'].median(),
            'frequency': 0,
            'monetary': 0,
            'satisfaction': rfm['satisfaction'].median(),
            'product_diversity': 0,
            'avg_delivery_time': rfm['avg_delivery_time'].median()
        })
        
        self.rfm_data = rfm
        return rfm
    
    def prepare_features(self, rfm_data):
        """Prepare features for clustering"""
        # Scale the features
        features = ['recency', 'frequency', 'monetary', 'satisfaction', 
                   'product_diversity', 'avg_delivery_time']
        
        # Impute missing values
        X = self.imputer.fit_transform(rfm_data[features])
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(X)
        
        return pd.DataFrame(scaled_data, columns=features, index=rfm_data.index)
    
    def find_optimal_clusters(self, scaled_data, max_clusters=10):
        """Find optimal number of clusters using silhouette score"""
        silhouette_scores = []
        K = range(2, max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            score = silhouette_score(scaled_data, kmeans.labels_)
            silhouette_scores.append(score)
        
        # Find optimal k
        optimal_k = K[np.argmax(silhouette_scores)]
        return optimal_k, silhouette_scores
    
    def train(self, df, n_clusters=None):
        """Train the clustering model"""
        # Calculate RFM metrics
        rfm_data = self.calculate_rfm(df)
        
        # Prepare features
        scaled_data = self.prepare_features(rfm_data)
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(scaled_data)
        
        # Train KMeans model
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(scaled_data)
        
        # Add cluster labels to RFM data
        self.cluster_data = rfm_data.copy()
        self.cluster_data['cluster'] = self.model.labels_
        
        return {
            'n_clusters': n_clusters,
            'cluster_sizes': self.cluster_data['cluster'].value_counts().to_dict()
        }
    
    def get_cluster_profiles(self):
        """Get detailed profiles for each cluster"""
        if self.cluster_data is None:
            raise ValueError("Model not trained yet!")
        
        # Calculate mean values for each cluster
        cluster_profiles = self.cluster_data.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'satisfaction': 'mean',
            'product_diversity': 'mean',
            'avg_delivery_time': 'mean'
        }).round(2)
        
        # Add cluster sizes
        cluster_sizes = self.cluster_data['cluster'].value_counts()
        cluster_profiles['size'] = cluster_sizes
        
        return cluster_profiles
    
    def plot_cluster_analysis(self):
        """Create visualizations for cluster analysis"""
        if self.cluster_data is None:
            raise ValueError("Model not trained yet!")
        
        # 1. RFM Distribution by Cluster
        fig_rfm = make_subplots(rows=1, cols=3, subplot_titles=('Recency', 'Frequency', 'Monetary'))
        
        for i, metric in enumerate(['recency', 'frequency', 'monetary']):
            fig_rfm.add_trace(
                go.Box(
                    y=self.cluster_data[metric],
                    x=self.cluster_data['cluster'],
                    name=metric.capitalize()
                ),
                row=1, col=i+1
            )
        
        fig_rfm.update_layout(height=400, showlegend=False)
        
        # 2. Cluster Sizes
        cluster_sizes = self.cluster_data['cluster'].value_counts()
        fig_sizes = px.pie(
            values=cluster_sizes.values,
            names=[f'Cluster {i}' for i in cluster_sizes.index],
            title='Cluster Distribution'
        )
        
        # 3. Cluster Characteristics
        cluster_profiles = self.get_cluster_profiles()
        fig_radar = go.Figure()
        
        for cluster in cluster_profiles.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=cluster_profiles.loc[cluster, ['recency', 'frequency', 'monetary', 
                                               'satisfaction', 'product_diversity']].values,
                theta=['Recency', 'Frequency', 'Monetary', 'Satisfaction', 'Product Diversity'],
                name=f'Cluster {cluster}'
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        
        return {
            'rfm_distribution': fig_rfm,
            'cluster_sizes': fig_sizes,
            'cluster_characteristics': fig_radar
        }
    
    def get_customer_segment(self, customer_id):
        """Get the segment for a specific customer"""
        if self.cluster_data is None:
            raise ValueError("Model not trained yet!")
        
        if customer_id not in self.cluster_data.index:
            raise ValueError(f"Customer {customer_id} not found in the data")
        
        return self.cluster_data.loc[customer_id, 'cluster']
    
    def save_model(self, path='models/customer_segmentation.joblib'):
        """Save the model and data"""
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'cluster_data': self.cluster_data
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path='models/customer_segmentation.joblib'):
        """Load the model and data"""
        import joblib
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.cluster_data = model_data['cluster_data'] 