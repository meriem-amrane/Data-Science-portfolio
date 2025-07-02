import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neon database connection
NEON_DATABASE_URL = os.getenv('NEON_DATABASE_URL')

# Page config
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 E-commerce Analytics Dashboard")

@st.cache_data
def load_data():
    """Load data from Neon database"""
    try:
        engine = create_engine(NEON_DATABASE_URL)
        with engine.connect() as connection:
            # Load main metrics
            df = pd.read_sql("""
                SELECT 
                    o.order_id,
                    o.order_purchase_timestamp,
                    o.order_status,
                    oi.price,
                    oi.freight_value,
                    p.product_category_name,
                    c.customer_state,
                    s.seller_state,
                    r.review_score
                FROM orders o
                JOIN order_items oi ON o.order_id = oi.order_id
                JOIN products p ON oi.product_id = p.product_id
                JOIN customers c ON o.customer_id = c.customer_id
                JOIN sellers s ON oi.seller_id = s.seller_id
                LEFT JOIN reviews r ON o.order_id = r.order_id
            """, connection)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = pd.to_datetime(df['order_purchase_timestamp']).min()
    max_date = pd.to_datetime(df['order_purchase_timestamp']).max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Category filter
    categories = ['All'] + sorted(df['product_category_name'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Filter data
    mask = (pd.to_datetime(df['order_purchase_timestamp']).dt.date >= date_range[0]) & \
           (pd.to_datetime(df['order_purchase_timestamp']).dt.date <= date_range[1])
    if selected_category != 'All':
        mask &= (df['product_category_name'] == selected_category)
    filtered_df = df[mask]

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_orders = len(filtered_df['order_id'].unique())
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col2:
        total_revenue = filtered_df['price'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col3:
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        st.metric("Average Order Value", f"${avg_order_value:,.2f}")
    
    with col4:
        avg_rating = filtered_df['review_score'].mean()
        st.metric("Average Rating", f"{avg_rating:.1f} ⭐")

    # Charts
    st.subheader("Sales Analysis")
    
    # Time series of daily sales
    daily_sales = filtered_df.groupby(pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.date)['price'].sum().reset_index()
    fig_sales = px.line(daily_sales, 
                       x='order_purchase_timestamp', 
                       y='price',
                       title='Daily Sales Trend')
    st.plotly_chart(fig_sales, use_container_width=True)

    # Top categories
    col1, col2 = st.columns(2)
    
    with col1:
        top_categories = filtered_df.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(10)
        fig_categories = px.bar(top_categories,
                              title='Top 10 Categories by Revenue')
        st.plotly_chart(fig_categories, use_container_width=True)
    
    with col2:
        # Customer state distribution
        state_dist = filtered_df.groupby('customer_state')['price'].sum().sort_values(ascending=False).head(10)
        fig_states = px.bar(state_dist,
                           title='Top 10 States by Revenue')
        st.plotly_chart(fig_states, use_container_width=True)

    # Review analysis
    st.subheader("Customer Satisfaction")
    
    # Rating distribution
    rating_dist = filtered_df['review_score'].value_counts().sort_index()
    fig_ratings = px.bar(rating_dist,
                        title='Rating Distribution')
    st.plotly_chart(fig_ratings, use_container_width=True)

else:
    st.error("Failed to load data. Please check your database connection.") 