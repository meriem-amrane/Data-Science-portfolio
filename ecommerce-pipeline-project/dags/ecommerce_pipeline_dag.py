from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys
import os

# Add the project root directory to Python path to allow importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main function from our extract script
from extract.extract import main as extract_main

# Define default arguments for the DAG
# These arguments will be applied to all tasks in the DAG
default_args = {
    'owner': 'airflow',              # Owner of the DAG
    'depends_on_past': False,        # Don't wait for previous runs to complete
    'email_on_failure': False,       # Don't send emails on failure
    'email_on_retry': False,         # Don't send emails on retry
    'retries': 1,                    # Number of retries if task fails
    'retry_delay': timedelta(minutes=5),  # Wait 5 minutes between retries
}

# Create the DAG object
# This defines the workflow and its schedule
dag = DAG(
    'ecommerce_pipeline',            # Unique identifier for the DAG
    default_args=default_args,       # Default arguments defined above
    description='E-commerce data pipeline for processing CSV files',  # DAG description
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=days_ago(1),          # Start from yesterday
    catchup=False,                   # Don't run for past dates
    tags=['ecommerce', 'data-pipeline'],  # Tags for filtering in the UI
)

# Define the extract task
# This task will run our CSV processing script
extract_task = PythonOperator(
    task_id='extract_data',          # Unique identifier for the task
    python_callable=extract_main,    # Function to execute
    dag=dag,                         # DAG this task belongs to
)

# Task dependencies
# Currently, we only have one task, but we can add more tasks
# and define their dependencies here
extract_task  # This is our only task for now, but we can add more tasks later 