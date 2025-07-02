import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import joblib
import os

class ProductRecommender:
    def __init__(self):
        self.product_matrix = None
        self.product_similarity = None
        self.product_categories = None
        self.customer_history = None
        
    def prepare_data(self, df):
        """Prepare data for product recommendations"""
        # Create customer-product matrix
        customer_product = df.groupby(['customer_id', 'product_id']).size().reset_index(name='purchase_count')
        
        # Create product-category mapping
        self.product_categories = df.groupby('product_id')['product_category_name'].first()
        
        # Create customer purchase history
        self.customer_history = df.groupby('customer_id')['product_id'].apply(list).to_dict()
        
        # Create product matrix
        self.product_matrix = customer_product.pivot(
            index='product_id',
            columns='customer_id',
            values='purchase_count'
        ).fillna(0)
        
        return self.product_matrix
    
    def train(self, df):
        """Train the recommendation model"""
        # Prepare data
        self.prepare_data(df)
        
        # Calculate product similarity
        self.product_similarity = cosine_similarity(self.product_matrix)
        
        # Convert to DataFrame for easier lookup
        self.product_similarity = pd.DataFrame(
            self.product_similarity,
            index=self.product_matrix.index,
            columns=self.product_matrix.index
        )
        
        return {
            'n_products': len(self.product_matrix),
            'n_customers': len(self.product_matrix.columns)
        }
    
    def get_similar_products(self, product_id, n_recommendations=5):
        """Get similar products based on purchase patterns"""
        if self.product_similarity is None:
            raise ValueError("Model not trained yet!")
        
        if product_id not in self.product_similarity.index:
            raise ValueError(f"Product {product_id} not found in the model")
        
        # Get similarity scores
        similar_products = self.product_similarity[product_id].sort_values(ascending=False)
        
        # Remove the product itself
        similar_products = similar_products[similar_products.index != product_id]
        
        # Get top N recommendations
        recommendations = similar_products.head(n_recommendations)
        
        # Add category information
        recommendations = pd.DataFrame({
            'product_id': recommendations.index,
            'similarity_score': recommendations.values,
            'category': [self.product_categories.get(pid, 'Unknown') for pid in recommendations.index]
        })
        
        return recommendations
    
    def get_customer_recommendations(self, customer_id, n_recommendations=5):
        """Get product recommendations for a specific customer"""
        if self.product_similarity is None:
            raise ValueError("Model not trained yet!")
        
        if customer_id not in self.customer_history:
            raise ValueError(f"Customer {customer_id} not found in the data")
        
        # Get customer's purchase history
        customer_products = self.customer_history[customer_id]
        
        # Calculate recommendation scores
        recommendation_scores = defaultdict(float)
        
        for product_id in customer_products:
            if product_id in self.product_similarity.index:
                similar_products = self.product_similarity[product_id]
                for similar_product, score in similar_products.items():
                    if similar_product not in customer_products:
                        recommendation_scores[similar_product] += score
        
        # Convert to DataFrame and sort
        recommendations = pd.DataFrame({
            'product_id': list(recommendation_scores.keys()),
            'score': list(recommendation_scores.values())
        }).sort_values('score', ascending=False)
        
        # Get top N recommendations
        recommendations = recommendations.head(n_recommendations)
        
        # Add category information
        recommendations['category'] = [
            self.product_categories.get(pid, 'Unknown') 
            for pid in recommendations['product_id']
        ]
        
        return recommendations
    
    def get_category_recommendations(self, category, n_recommendations=5):
        """Get popular products in a specific category"""
        if self.product_matrix is None:
            raise ValueError("Model not trained yet!")
        
        # Get products in the category
        category_products = self.product_categories[
            self.product_categories == category
        ].index
        
        if len(category_products) == 0:
            raise ValueError(f"Category {category} not found in the data")
        
        # Calculate total purchases for each product
        product_popularity = self.product_matrix.loc[category_products].sum(axis=1)
        
        # Get top N products
        recommendations = pd.DataFrame({
            'product_id': product_popularity.index,
            'popularity_score': product_popularity.values
        }).sort_values('popularity_score', ascending=False)
        
        return recommendations.head(n_recommendations)
    
    def save_model(self, path='models/product_recommender.joblib'):
        """Save the model"""
        if self.product_similarity is None:
            raise ValueError("No model to save!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'product_similarity': self.product_similarity,
            'product_categories': self.product_categories,
            'customer_history': self.customer_history
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path='models/product_recommender.joblib'):
        """Load the model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        self.product_similarity = model_data['product_similarity']
        self.product_categories = model_data['product_categories']
        self.customer_history = model_data['customer_history'] 