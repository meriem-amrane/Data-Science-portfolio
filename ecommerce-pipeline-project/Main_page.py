import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import psycopg2
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load custom CSS
with open('app/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Database connection
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        # Create engine with connection pooling and timeout
        connection_string = "postgresql://neondb_owner:npg_0rujoZn4byLY@ep-withered-water-a8c7kc7f-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
        engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        
        # Test connection with timeout
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.commit()
        
        # Load data with chunking for large datasets
        query = text("SELECT * FROM transformed_data")
        chunks = []
        for chunk in pd.read_sql(query, engine, chunksize=10000):
            chunks.append(chunk)
        
        if not chunks:
            st.error("No data found in the database.")
            return None
            
        df = pd.concat(chunks, ignore_index=True)
        
        # Convert timestamp columns to datetime with error handling
        timestamp_columns = ['order_purchase_timestamp', 'order_delivered_customer_date']
        for col in timestamp_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Remove rows with invalid dates
                    df = df[df[col].notna()]
                except Exception as e:
                    st.warning(f"Warning: Could not convert {col} to datetime: {str(e)}")
        
        if df.empty:
            st.error("No valid data after processing.")
            return None
            
        return df
        
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()

if df is not None and not df.empty:
    try:
        # Get min and max dates for the date range slider
        min_date = pd.to_datetime(df['order_purchase_timestamp']).dt.date.min()
        max_date = pd.to_datetime(df['order_purchase_timestamp']).dt.date.max()

        # Sidebar filters
        st.sidebar.title("Filters")

        # Date range selector
        st.sidebar.subheader("Date Range")
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Product Category Filter
        st.sidebar.subheader("Product Category")
        all_categories = ['All'] + sorted(df['product_category_name'].unique().tolist())
        selected_category = st.sidebar.selectbox(
            "Select Category",
            options=all_categories,
            index=0
        )

        # Payment Method Filter
        st.sidebar.subheader("Payment Method")
        all_payment_methods = ['All'] + sorted(df['payment_type'].unique().tolist())
        selected_payment = st.sidebar.selectbox(
            "Select Payment Method",
            options=all_payment_methods,
            index=0
        )

        # Order Status Filter
        st.sidebar.subheader("Order Status")
        all_statuses = ['All'] + sorted(df['order_status'].unique().tolist())
        selected_status = st.sidebar.selectbox(
            "Select Order Status",
            options=all_statuses,
            index=0
        )

        # Price Range Filter
        st.sidebar.subheader("Price Range")
        min_price = float(df['price'].min())
        max_price = float(df['price'].max())
        price_range = st.sidebar.slider(
            "Select Price Range ($)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )

        # Apply all filters
        filtered_df = df.copy()
        
        # Date range filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.date >= start_date) &
                (pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.date <= end_date)
            ]

        # Category filter
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['product_category_name'] == selected_category]

        # Payment method filter
        if selected_payment != 'All':
            filtered_df = filtered_df[filtered_df['payment_type'] == selected_payment]

        # Order status filter
        if selected_status != 'All':
            filtered_df = filtered_df[filtered_df['order_status'] == selected_status]

        # Price range filter
        filtered_df = filtered_df[
            (filtered_df['price'] >= price_range[0]) &
            (filtered_df['price'] <= price_range[1])
        ]

        # Add filter summary
        st.sidebar.markdown("---")
        st.sidebar.subheader("Active Filters")
        st.sidebar.markdown(f"**Date Range:** {date_range[0]} to {date_range[1]}")
        st.sidebar.markdown(f"**Category:** {selected_category}")
        st.sidebar.markdown(f"**Payment Method:** {selected_payment}")
        st.sidebar.markdown(f"**Order Status:** {selected_status}")
        st.sidebar.markdown(f"**Price Range:** ${price_range[0]:,.2f} - ${price_range[1]:,.2f}")

        # Add clear filters button
        if st.sidebar.button("Clear All Filters"):
            st.rerun()

        # Main content
        st.title("E-commerce Analytics Dashboard")
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Business Overview", "Product Analysis", "Customer Insights"])
        with tab1:
            st.header("Business Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                total_sales = filtered_df['price'].sum()
                st.metric("Total Sales", f"${total_sales:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                total_orders = len(filtered_df['order_id'].unique())
                st.metric("Total Orders", f"{total_orders:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_order_value = total_sales / total_orders if total_orders > 0 else 0
                st.metric("Average Order Value", f"${avg_order_value:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                unique_customers = len(filtered_df['customer_id'].unique())
                st.metric("Unique Customers", f"{unique_customers:,}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Business Insights
            st.subheader("Business Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Payment Method Distribution
                payment_methods = filtered_df.groupby('payment_type')['price'].agg(['count', 'sum']).reset_index()
                fig_payment = px.pie(
                    payment_methods,
                    values='count',
                    names='payment_type',
                    title="Payment Methods Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_payment.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_payment, use_container_width=True, key="payment_methods_pie")
                
                # Payment Method Stats
                st.markdown("**Payment Method Statistics**")
                payment_stats = payment_methods.sort_values('sum', ascending=False)
                st.dataframe(
                    payment_stats.style.format({'sum': '${:,.2f}', 'count': '{:,.0f}'}),
                    hide_index=True
                )

            with col2:
                # Order Status Timeline
                status_timeline = filtered_df.groupby([
                    pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.date,
                    'order_status'
                ], observed=True)['order_id'].count().reset_index()
                
                fig_status = px.line(
                    status_timeline,
                    x='order_purchase_timestamp',
                    y='order_id',
                    color='order_status',
                    title="Order Status Timeline",
                    labels={
                        'order_purchase_timestamp': 'Date',
                        'order_id': 'Number of Orders',
                        'order_status': 'Status'
                    }
                )
                st.plotly_chart(fig_status, use_container_width=True, key="order_status_timeline")
                
                # Status Summary
                st.markdown("**Order Status Summary**")
                status_summary = filtered_df.groupby('order_status').agg({
                    'order_id': 'count',
                    'price': 'sum'
                }).reset_index()
                st.dataframe(
                    status_summary.style.format({'price': '${:,.2f}', 'order_id': '{:,.0f}'}),
                    hide_index=True
                )

            with col3:
                # Customer Purchase Frequency
                customer_freq = filtered_df.groupby('customer_id').agg({
                    'order_id': 'count',
                    'price': 'sum'
                }).reset_index()
                
                fig_freq = px.scatter(
                    customer_freq,
                    x='order_id',
                    y='price',
                    title="Customer Purchase Frequency vs Total Spent",
                    labels={
                        'order_id': 'Number of Orders',
                        'price': 'Total Spent ($)'
                    },
                    color_discrete_sequence=['#9C27B0']
                )
                st.plotly_chart(fig_freq, use_container_width=True, key="overview_customer_freq")
                
                # Customer Segments
                st.markdown("**Customer Segments**")
                customer_freq['segment'] = pd.qcut(
                    customer_freq['price'],
                    q=4,
                    labels=['Bronze', 'Silver', 'Gold', 'Platinum']
                )
                segment_stats = customer_freq.groupby('segment', observed=True).agg({
                    'customer_id': 'count',
                    'price': 'mean',
                    'order_id': 'mean'
                }).reset_index()
                st.dataframe(
                    segment_stats.style.format({
                        'price': '${:,.2f}',
                        'order_id': '{:.1f}',
                        'customer_id': '{:,.0f}'
                    }),
                    hide_index=True
                )

        with tab2:
            st.header("Product Analysis")
            
            # Product Performance Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                total_products = len(filtered_df['product_id'].unique())
                st.metric("Total Products", f"{total_products:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_product_price = filtered_df['price'].mean()
                st.metric("Average Product Price", f"${avg_product_price:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                total_categories = len(filtered_df['product_category_name'].unique())
                st.metric("Product Categories", f"{total_categories:,}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Product Analysis Visualizations
            st.subheader("Product Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                # Top Products by Sales
                top_products = filtered_df.groupby('product_id').agg({
                    'price': 'sum',
                    'order_id': 'count'
                }).reset_index()
                top_products = top_products.sort_values('price', ascending=False).head(10)
                
                fig_top_products = px.bar(
                    top_products,
                    x='product_id',
                    y='price',
                    title="Top 10 Products by Sales",
                    labels={
                        'product_id': 'Product ID',
                        'price': 'Total Sales ($)'
                    }
                )
                st.plotly_chart(fig_top_products, use_container_width=True, key="top_products_bar")
                
                # Product Category Distribution
                category_sales = filtered_df.groupby('product_category_name').agg({
                    'price': 'sum',
                    'order_id': 'count'
                }).reset_index()
                category_sales = category_sales.sort_values('price', ascending=False)
                
                fig_category = px.pie(
                    category_sales,
                    values='price',
                    names='product_category_name',
                    title="Sales by Product Category",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_category.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_category, use_container_width=True, key="category_sales_pie")
            
            with col2:
                # Product Price Distribution
                fig_price_dist = px.histogram(
                    filtered_df,
                    x='price',
                    title="Product Price Distribution",
                    labels={'price': 'Price ($)'},
                    nbins=50
                )
                st.plotly_chart(fig_price_dist, use_container_width=True, key="price_dist_hist")
                
                # Product Category Stats
                st.markdown("**Category Performance**")
                category_stats = category_sales.sort_values('price', ascending=False)
                st.dataframe(
                    category_stats.style.format({
                        'price': '${:,.2f}',
                        'order_id': '{:,.0f}'
                    }),
                    hide_index=True
                )

        with tab3:
            st.header("Customer Insights")
            # Customer Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                repeat_customers = len(filtered_df.groupby('customer_id').filter(lambda x: len(x) > 1)['customer_id'].unique())
                st.metric("Repeat Customers", f"{repeat_customers:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_customer_value = filtered_df.groupby('customer_id')['price'].sum().mean()
                st.metric("Average Customer Value", f"${avg_customer_value:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_orders_per_customer = filtered_df.groupby('customer_id')['order_id'].nunique().mean()
                st.metric("Avg Orders per Customer", f"{avg_orders_per_customer:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Customer Analysis Visualizations
            st.subheader("Customer Behavior")
            col1, col2 = st.columns(2)
            
            with col1:
                # Customer Purchase Frequency
                customer_freq = filtered_df.groupby('customer_id').agg({
                    'order_id': 'count',
                    'price': 'sum'
                }).reset_index()
                
                fig_freq = px.scatter(
                    customer_freq,
                    x='order_id',
                    y='price',
                    title="Customer Purchase Frequency vs Total Spent",
                    labels={
                        'order_id': 'Number of Orders',
                        'price': 'Total Spent ($)'
                    },
                    color_discrete_sequence=['#9C27B0']
                )
                st.plotly_chart(fig_freq, use_container_width=True, key="insights_customer_freq")
                
                # Customer Segments
                st.markdown("**Customer Segments**")
                customer_freq['segment'] = pd.qcut(
                    customer_freq['price'],
                    q=4,
                    labels=['Bronze', 'Silver', 'Gold', 'Platinum']
                )
                segment_stats = customer_freq.groupby('segment', observed=True).agg({
                    'customer_id': 'count',
                    'price': 'mean',
                    'order_id': 'mean'
                }).reset_index()
                st.dataframe(
                    segment_stats.style.format({
                        'price': '${:,.2f}',
                        'order_id': '{:.1f}',
                        'customer_id': '{:,.0f}'
                    }),
                    hide_index=True
                )
            
            with col2:
                # Customer Purchase Timeline
                customer_timeline = filtered_df.groupby([
                    pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.date
                ])['customer_id'].nunique().reset_index()
                
                fig_timeline = px.line(
                    customer_timeline,
                    x='order_purchase_timestamp',
                    y='customer_id',
                    title="Daily Active Customers",
                    labels={
                        'order_purchase_timestamp': 'Date',
                        'customer_id': 'Number of Customers'
                    }
                )
                st.plotly_chart(fig_timeline, use_container_width=True, key="customer_timeline")
                
                # Customer Retention
                st.markdown("**Customer Retention Analysis**")
                retention_data = filtered_df.groupby('customer_id').agg({
                    'order_purchase_timestamp': ['min', 'max'],
                    'order_id': 'count'
                }).reset_index()
                retention_data.columns = ['customer_id', 'first_purchase', 'last_purchase', 'total_orders']
                retention_data['days_active'] = (retention_data['last_purchase'] - retention_data['first_purchase']).dt.days
                
                retention_stats = retention_data.groupby('total_orders').agg({
                    'customer_id': 'count',
                    'days_active': 'mean'
                }).reset_index()
                st.dataframe(
                    retention_stats.style.format({
                        'days_active': '{:.1f}',
                        'customer_id': '{:,.0f}'
                    }),
                    hide_index=True
                )

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.error("Failed to load data. Please check your database connection and ensure the 'transformed_data' table exists.")
