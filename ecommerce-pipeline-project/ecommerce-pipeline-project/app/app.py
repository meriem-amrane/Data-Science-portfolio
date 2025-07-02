"""
E-commerce Analytics Platform
Main application file using Streamlit for interactive data analysis and ML predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="E-commerce Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Models"])

# Load models with caching
@st.cache_resource
def load_models():
    """
    Load all machine learning models with caching for better performance.
    Returns a dictionary containing all models.
    """
    models = {}
    try:
        models['churn'] = joblib.load('models/churn_model.joblib')
        models['delivery'] = joblib.load('models/delivery_model.joblib')
        models['recommender'] = joblib.load('models/recommender_model.joblib')
        models['anomaly'] = joblib.load('models/anomaly_model.joblib')
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    return models

def show_home():
    """
    Display the home page with overview statistics and feature highlights.
    """
    st.title("E-commerce Analytics Platform")
    
    # Hero section
    st.markdown("""
    ### Analyze your data, predict trends, and optimize performance
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales", "$1,234,567", "+12%")
    with col2:
        st.metric("Orders", "12,345", "+8%")
    with col3:
        st.metric("Customers", "8,765", "+15%")
    with col4:
        st.metric("Average Rating", "4.5/5", "+0.2")
    
    # Features section
    st.markdown("### Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📊 Analytics Dashboard
        Visualize KPIs, sales trends, and seller performance in real-time.
        """)
    
    with col2:
        st.markdown("""
        #### 👥 Customer Segmentation
        Analyze customers using RFM methodology and personalize marketing strategies.
        """)
    
    with col3:
        st.markdown("""
        #### 🧠 Advanced Predictions
        Predict churn, delivery times, and detect anomalies.
        """)

def show_dashboard():
    """
    Display the analytics dashboard with interactive visualizations and data export options.
    """
    st.title("Analytics Dashboard")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Sales trend
    st.subheader("Sales Trend")
    sales_data = pd.DataFrame({
        'date': pd.date_range(start=start_date, end=end_date),
        'sales': np.random.normal(1000, 200, (end_date - start_date).days + 1),
        'orders': np.random.normal(50, 10, (end_date - start_date).days + 1)
    })
    
    fig = px.line(sales_data, x='date', y='sales', title='Sales Evolution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.download_button(
        "📥 Export Data",
        sales_data.to_csv(index=False).encode('utf-8'),
        "sales_data.csv",
        "text/csv",
        key='download-csv'
    )
    
    # Top categories
    st.subheader("Top Categories")
    categories_data = pd.DataFrame({
        'category': ['Electronics', 'Fashion', 'Home', 'Sports', 'Beauty'],
        'sales': np.random.normal(5000, 1000, 5)
    })
    
    fig = px.bar(categories_data, x='category', y='sales', title='Sales by Category')
    st.plotly_chart(fig, use_container_width=True)
    
    # Seller performance
    st.subheader("Seller Performance")
    sellers_data = pd.DataFrame({
        'seller': [f'Seller {i}' for i in range(1, 6)],
        'sales': np.random.normal(2000, 500, 5),
        'rating': np.random.normal(4.5, 0.5, 5)
    })
    
    fig = px.scatter(sellers_data, x='sales', y='rating', text='seller',
                    title='Seller Performance')
    st.plotly_chart(fig, use_container_width=True)
    
    # Reviews analysis
    st.subheader("Review Analysis")
    reviews_data = pd.DataFrame({
        'rating': np.random.normal(4.2, 0.8, 1000),
        'sentiment': np.random.choice(['Positive', 'Neutral', 'Negative'], 1000)
    })
    
    fig = px.histogram(reviews_data, x='rating', title='Rating Distribution')
    st.plotly_chart(fig, use_container_width=True)

def show_models():
    """
    Display the machine learning models section with interactive predictions and analysis.
    """
    st.title("Machine Learning Models")
    
    # Load models
    models = load_models()
    
    # Customer segmentation
    st.header("Customer Segmentation (RFM)")
    st.markdown("""
    Analyze your customers based on Recency, Frequency, and Monetary value.
    """)
    
    if st.button("Run RFM Analysis"):
        with st.spinner("Analysis in progress..."):
            # Simulate RFM analysis
            rfm_data = pd.DataFrame({
                'segment': ['VIP', 'Loyal', 'Regular', 'Occasional', 'Inactive'],
                'count': np.random.normal(1000, 200, 5)
            })
            
            fig = px.pie(rfm_data, values='count', names='segment',
                        title='Customer Segment Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    # Churn prediction
    st.header("Churn Prediction")
    st.markdown("""
    Predict the probability of a customer leaving your platform.
    """)
    
    with st.form("churn_prediction"):
        st.subheader("Customer Information")
        col1, col2 = st.columns(2)
        with col1:
            days_since_last_purchase = st.number_input("Days since last purchase", 0, 365)
            total_orders = st.number_input("Total number of orders", 0, 100)
        with col2:
            avg_order_value = st.number_input("Average order value ($)", 0.0, 1000.0)
            customer_rating = st.number_input("Customer average rating", 1.0, 5.0)
        
        if st.form_submit_button("Predict Churn"):
            with st.spinner("Calculating..."):
                # Simulate churn prediction
                churn_prob = np.random.random()
                st.metric("Churn Probability", f"{churn_prob:.1%}")
    
    # Delivery prediction
    st.header("Delivery Prediction")
    st.markdown("""
    Estimate delivery time for new orders.
    """)
    
    with st.form("delivery_prediction"):
        st.subheader("Order Details")
        col1, col2 = st.columns(2)
        with col1:
            distance = st.number_input("Distance (km)", 0.0, 1000.0)
            weight = st.number_input("Weight (kg)", 0.0, 50.0)
        with col2:
            priority = st.selectbox("Priority", ["Standard", "Express", "Premium"])
            weather = st.selectbox("Weather conditions", ["Good", "Average", "Poor"])
        
        if st.form_submit_button("Predict Delivery Time"):
            with st.spinner("Calculating..."):
                # Simulate delivery prediction
                delivery_time = np.random.normal(3, 1)
                st.metric("Estimated delivery time", f"{delivery_time:.1f} days")
    
    # Product recommendations
    st.header("Product Recommendations")
    st.markdown("""
    Discover recommended products for your customers.
    """)
    
    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations..."):
            # Simulate product recommendations
            recommendations = pd.DataFrame({
                'product': [f'Product {i}' for i in range(1, 6)],
                'score': np.random.normal(0.8, 0.1, 5)
            })
            
            fig = px.bar(recommendations, x='product', y='score',
                        title='Top 5 Recommended Products')
            st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly detection
    st.header("Anomaly Detection")
    st.markdown("""
    Identify unusual patterns in your data.
    """)
    
    if st.button("Detect Anomalies"):
        with st.spinner("Analysis in progress..."):
            # Simulate anomaly detection
            anomalies = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=100),
                'value': np.random.normal(100, 10, 100)
            })
            anomalies.loc[10:15, 'value'] *= 3  # Create some anomalies
            
            fig = px.line(anomalies, x='timestamp', y='value',
                         title='Anomaly Detection')
            st.plotly_chart(fig, use_container_width=True)

def main():
    """
    Main application function that handles page routing.
    """
    if page == "Home":
        show_home()
    elif page == "Dashboard":
        show_dashboard()
    else:
        show_models()

if __name__ == "__main__":
    main()
