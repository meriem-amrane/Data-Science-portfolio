import pandas as pd
import numpy as np
import seaborn as sns
from sqlalchemy import create_engine
import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.config.database import DATABASE_URL
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_connection():
    """
    Create and return a database connection
    """
    try:
        engine = create_engine(DATABASE_URL)
        logger.info("Database connection established successfully")
        return engine
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def load_data_from_db(engine):
    """
    Load data from database
    """
    try:
        # Load data from each table
        with engine.connect() as connection:
            customers = pd.read_sql('SELECT * FROM customers', connection)
            orders = pd.read_sql('SELECT * FROM orders', connection)
            order_items = pd.read_sql('SELECT * FROM order_items', connection)
            products = pd.read_sql('SELECT * FROM products', connection)
            sellers = pd.read_sql('SELECT * FROM sellers', connection)
            payments = pd.read_sql('SELECT * FROM payments', connection)
            reviews = pd.read_sql('SELECT * FROM reviews', connection)

        # Merge all tables
        df = (orders
              .merge(customers, on='customer_id', how='left')
              .merge(order_items, on='order_id', how='left')
              .merge(products, on='product_id', how='left')
              .merge(sellers, on='seller_id', how='left')
              .merge(payments, on='order_id', how='left')
              .merge(reviews, on='order_id', how='left'))
        
        logger.info("Data loaded successfully from database")
        return df
    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")
        raise

def save_to_database(df, engine, table_name):
    """
    Save cleaned data to database
    """
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"Data saved successfully to {table_name} table")
    except Exception as e:
        logger.error(f"Error saving data to database: {str(e)}")
        raise

def clean_df(df):
   
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
     
    print("----- Starting data cleaning process -----")
    
    #Data Conversions
    
     # 1. Convert date columns to datetime
    print("\n1. Converting date columns...")
    date_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'review_creation_date',
        'review_answer_timestamp',
        'shipping_limit_date'
    ]
    
    for col in date_columns:
        df_clean[col] = pd.to_datetime(df_clean[col] )
        
    # 2. Handling missing values
    print('\n2. Handling missing values')
    
    #For numerical data
    numerical_columns = df_clean.select_dtypes(include=['float64','int64']).columns
    
    for col in numerical_columns:
        if df_clean[col].isnull().sum() >0 :
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value,inplace=True)
            print(f"Filled missing values in {col} with median: {median_value}")
    
    #for categorical data
    categorical_columns = df_clean.select_dtypes(include = ['object']).columns
    for col in categorical_columns:
        if df_clean[col].isnull().sum()>0:
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value,inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_value}")
    
    # 3. Handling outliers in numerical columns 
    print('\n3. Handling outliers in numerical columns')
    for col in numerical_columns:
        if col not in ['order_id', 'customer_id', 'seller_id', 'product_id','review_id','order_item_id']:  
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
        # Count outliers
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"Found {outliers} outliers in {col}")
                # Cap outliers
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        
        # 4. Clean string columns
    print("\n4. Cleaning string columns...")
    for col in categorical_columns:
        if df_clean[col].dtype == 'object':
            # Convert to lowercase
            df_clean[col] = df_clean[col].str.lower()
            # Remove extra whitespace
            df_clean[col] = df_clean[col].str.strip()
            # Replace multiple spaces with single space
            df_clean[col] = df_clean[col].str.replace('\s+', ' ', regex=True)
    
    # 5. Remove duplicates
    print("\n5. Removing duplicates...")
    initial_rows = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    final_rows = len(df_clean)
    if initial_rows != final_rows:
        print(f"Removed {initial_rows - final_rows} duplicate rows")
    
    
     # 6. Create new features
    print("\n6. Creating new features...")
    
    # Calculate delivery time in days
    df_clean['delivery_time_days'] = (df_clean['order_delivered_customer_date'] - 
                                    df_clean['order_purchase_timestamp']).dt.days
    
    # Calculate if delivery was late
    df_clean['is_late_delivery'] = df_clean['delivery_time_days'] > (
        df_clean['order_estimated_delivery_date'] - 
        df_clean['order_purchase_timestamp']
    ).dt.days
    
    # Calculate total order value
    df_clean['total_order_value'] = df_clean['price'] + df_clean['freight_value']
    
    # Extract date components
    df_clean['purchase_year'] = df_clean['order_purchase_timestamp'].dt.year
    df_clean['purchase_month'] = df_clean['order_purchase_timestamp'].dt.month
    df_clean['purchase_day'] = df_clean['order_purchase_timestamp'].dt.day
    df_clean['purchase_hour'] = df_clean['order_purchase_timestamp'].dt.hour
    
    print("\nData cleaning completed!")
        

    
    return df_clean

def main():
    try:
        # Get database connection
        engine = get_database_connection()
        
        # Load data from database
        df = load_data_from_db(engine)
        
        # Clean the data
        df_cleaned = clean_df(df)
        
        # Save cleaned data back to database
        save_to_database(df_cleaned, engine, 'cleaned_data')
        
        # Also save to CSV as backup
        df_cleaned.to_csv("data/transformed_data.csv", index=False)
        logger.info("Data cleaning process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()