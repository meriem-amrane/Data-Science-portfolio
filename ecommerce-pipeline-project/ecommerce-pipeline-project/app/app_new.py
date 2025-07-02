import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from models.review_predictor import ReviewPredictor
from models.customer_segmentation import CustomerSegmentation
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric:hover {
        background-color: #e6e9ef;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
    .tab-content {
        padding: 1rem;
    }
    .review-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .cluster-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def init_connection():
    try:
        connection_string = "postgresql://neondb_owner:npg_0rujoZn4byLY@ep-withered-water-a8c7kc7f-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

# Initialize connection
engine = init_connection()

if engine is not None:
    try:
        # Application title
        st.title("🛍️ E-commerce Analytics Dashboard")

        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Load data
        @st.cache_data(ttl=3600)
        def load_data():
            query = "SELECT * FROM transformed_data"
            return pd.read_sql(query, engine)

        df = load_data()

        # Sidebar filters
        st.sidebar.subheader("Category Filters")
        categories = sorted(df['product_category_name'].unique())
        selected_categories = st.sidebar.multiselect(
            "Select categories",
            categories,
            default=categories[:3]
        )

        # Date range filter
        st.sidebar.subheader("Date Range")
        min_date = df['order_purchase_timestamp'].min()
        max_date = df['order_purchase_timestamp'].max()
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Filter data
        filtered_df = df[
            (df['product_category_name'].isin(selected_categories)) &
            (df['order_purchase_timestamp'].dt.date >= date_range[0]) &
            (df['order_purchase_timestamp'].dt.date <= date_range[1])
        ]

        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "📈 Detailed Analysis", 
            "🔍 Data Exploration",
            "📊 Advanced Analytics",
            "🤖 Review Prediction",
            "👥 Customer Segmentation"
        ])

        # ... [Previous tabs code remains unchanged] ...

        with tab6:
            st.header("👥 Customer Segmentation")
            
            # Initialize or load the segmentation model
            @st.cache_resource
            def load_segmentation_model():
                segmenter = CustomerSegmentation()
                model_path = 'models/customer_segmentation.joblib'
                
                if os.path.exists(model_path):
                    segmenter.load_model(model_path)
                else:
                    with st.spinner("Training the customer segmentation model... This might take a few minutes."):
                        metrics = segmenter.train(df)
                        segmenter.save_model()
                        st.success(f"Model trained successfully! Number of clusters: {metrics['n_clusters']}")
                
                return segmenter

            segmenter = load_segmentation_model()
            
            # Display cluster analysis
            st.subheader("Cluster Analysis")
            
            # Get cluster profiles
            cluster_profiles = segmenter.get_cluster_profiles()
            
            # Display cluster sizes
            st.write("### Cluster Distribution")
            cluster_sizes = cluster_profiles['size']
            fig_sizes = px.pie(
                values=cluster_sizes.values,
                names=[f'Cluster {i}' for i in cluster_sizes.index],
                title='Customer Segments Distribution'
            )
            st.plotly_chart(fig_sizes, use_container_width=True)
            
            # Display cluster characteristics
            st.write("### Cluster Characteristics")
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
                showlegend=True,
                title="Cluster Characteristics Radar Chart"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Display detailed cluster profiles
            st.write("### Detailed Cluster Profiles")
            
            # Format the cluster profiles for display
            display_profiles = cluster_profiles.copy()
            display_profiles = display_profiles.round(2)
            display_profiles.columns = [
                'Recency (days)', 'Frequency', 'Monetary (R$)', 
                'Satisfaction', 'Product Diversity', 'Size'
            ]
            
            # Add cluster descriptions
            cluster_descriptions = {
                0: "High-value, frequent customers",
                1: "New, high-potential customers",
                2: "At-risk customers",
                3: "Loyal, moderate-value customers",
                4: "Occasional, high-satisfaction customers"
            }
            
            display_profiles['Description'] = [cluster_descriptions.get(i, f"Cluster {i}") 
                                             for i in display_profiles.index]
            
            st.dataframe(display_profiles)
            
            # Customer lookup
            st.write("### Customer Segment Lookup")
            customer_id = st.text_input("Enter Customer ID to find their segment")
            
            if customer_id:
                try:
                    segment = segmenter.get_customer_segment(customer_id)
                    st.success(f"Customer {customer_id} belongs to {cluster_descriptions.get(segment, f'Cluster {segment}')}")
                    
                    # Display customer's metrics
                    customer_metrics = segmenter.cluster_data.loc[customer_id]
                    st.write("#### Customer Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Recency (days)', 'Frequency', 'Monetary (R$)', 
                                 'Satisfaction', 'Product Diversity', 'Average Delivery Time'],
                        'Value': [
                            customer_metrics['recency'],
                            customer_metrics['frequency'],
                            customer_metrics['monetary'],
                            customer_metrics['satisfaction'],
                            customer_metrics['product_diversity'],
                            customer_metrics['avg_delivery_time']
                        ]
                    })
                    st.dataframe(metrics_df)
                    
                except ValueError as e:
                    st.error(str(e))

        # Footer
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center'>
                <p>Dashboard created with Streamlit and Neon</p>
                <p>Last updated: {}</p>
            </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.error("Unable to connect to the database. Please check your connection settings.") 