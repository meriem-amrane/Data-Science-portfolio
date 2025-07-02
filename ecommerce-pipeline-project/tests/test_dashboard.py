import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDashboard(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'order_id': ['1', '2', '3'],
            'customer_id': ['C1', 'C2', 'C3'],
            'product_category_name': ['Electronics', 'Books', 'Electronics'],
            'price': [100.0, 50.0, 200.0],
            'order_purchase_timestamp': [
                datetime.now(),
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=2)
            ],
            'review_score': [5, 4, 3],
            'customer_state': ['CA', 'NY', 'TX']
        })

    def test_data_filtering(self):
        """Test data filtering functionality"""
        # Test category filtering
        filtered_data = self.sample_data[self.sample_data['product_category_name'] == 'Electronics']
        self.assertEqual(len(filtered_data), 2)
        self.assertTrue(all(cat == 'Electronics' for cat in filtered_data['product_category_name']))

        # Test price filtering
        filtered_data = self.sample_data[self.sample_data['price'] > 100]
        self.assertEqual(len(filtered_data), 1)
        self.assertEqual(filtered_data['price'].iloc[0], 200.0)

    def test_metrics_calculation(self):
        """Test metrics calculation"""
        # Test total sales
        total_sales = self.sample_data['price'].sum()
        self.assertEqual(total_sales, 350.0)

        # Test average price
        avg_price = self.sample_data['price'].mean()
        self.assertEqual(avg_price, 116.67)

        # Test unique customers
        unique_customers = len(self.sample_data['customer_id'].unique())
        self.assertEqual(unique_customers, 3)

    def test_review_analysis(self):
        """Test review analysis functionality"""
        # Test average review score
        avg_review = self.sample_data['review_score'].mean()
        self.assertEqual(avg_review, 4.0)

        # Test review distribution
        review_dist = self.sample_data['review_score'].value_counts()
        self.assertEqual(review_dist[5], 1)
        self.assertEqual(review_dist[4], 1)
        self.assertEqual(review_dist[3], 1)

    def test_geographical_analysis(self):
        """Test geographical analysis functionality"""
        # Test state-wise sales
        state_sales = self.sample_data.groupby('customer_state')['price'].sum()
        self.assertEqual(state_sales['CA'], 100.0)
        self.assertEqual(state_sales['NY'], 50.0)
        self.assertEqual(state_sales['TX'], 200.0)

if __name__ == '__main__':
    unittest.main() 