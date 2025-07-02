# Ecommerce Pipeline Project

This project is a comprehensive data science pipeline for e-commerce data, including ETL (Extract, Transform, Load), machine learning model training, and dashboard visualization.

## 📁 Project Structure

```
ecommerce-pipeline-project/
│
├── _config.yml                  # Configuration file for documentation or tools
├── airflow_dags/                # Airflow DAGs for orchestrating ETL and ML workflows
├── app/                         # Web application (dashboard) for data and model visualization
│   ├── app.py                   # Main app script
│   ├── models/                  # ML models used by the app
│   ├── style.css                # Stylesheet for the dashboard
│   └── templates/               # HTML templates for the dashboard
│       ├── dashboard.html
│       ├── index.html
│       └── models.html
├── dags/                        # Additional Airflow DAGs
│   └── ecommerce_pipeline_dag.py
├── dashboard/                   # Standalone dashboard app (alternative or legacy)
│   └── app.py
├── data/                        # Data files (not versioned in git)
├── DATA.md                      # Data documentation
├── database/                    # Database connection and ORM models
│   ├── __init__.py
│   ├── config.py
│   ├── connection.py
│   ├── models.py
│   └── setup.py
├── docs/                        # Project documentation (Sphinx)
│   ├── code_viewer.html
│   ├── conf.py
│   ├── images/
│   ├── index.rst
│   └── Makefile
├── extract/                     # Data extraction scripts
│   └── extract.py
├── Home.py                      # Main entry point or landing page script
├── index.md                     # Project index in markdown
├── LICENSE                      # License file
├── load/                        # Data loading scripts
├── models/                      # Machine learning models and related code
├── notebooks/                   # Jupyter notebooks for exploration and analysis
├── pages/                       # Streamlit or dashboard pages
│   ├── 1_📊_Modeles.py
│   └── 2_🎯_Model_Performance.py
├── README_MULTI_PAGES.md        # Additional README for multi-page apps
├── README.md                    # (This file)
├── requirements.txt             # Python dependencies
├── scripts/                     # Scripts for training ML models
│   ├── train_all_models_fast.py
│   ├── train_churn_model.py
│   ├── train_delivery_model.py
│   ├── train_price_model_optimized.py
│   ├── train_price_model.py
│   └── train_recommendation_model.py
├── src/                         # Source code (config, database, etc.)
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── database.py
│   └── database/
│       ├── connection.py
│       └── models.py
├── tests/                       # Unit and integration tests
│   └── test_dashboard.py
├── transform/                   # Data transformation scripts and notebooks
│   ├── 01_data_loading_and_setup.ipynb
│   ├── 02_data_cleaning.py
│   ├── data_description.ipynb
│   └── transform.py
├── tree.txt                     # Textual tree view of the project
└── venv/                        # Virtual environment (should be in .gitignore)
```

## 📝 Description of Main Components

- **airflow_dags/** & **dags/**: Define and schedule ETL and ML workflows using Apache Airflow.
- **app/**: Contains the dashboard web app for visualizing data and model results.
- **database/** & **src/database/**: Database configuration, connection, and ORM models.
- **extract/**, **load/**, **transform/**: Scripts for extracting, loading, and transforming data.
- **models/** & **scripts/**: Machine learning models and training scripts.
- **notebooks/**: Jupyter notebooks for data exploration and prototyping.
- **pages/**: Additional dashboard or Streamlit pages.
- **docs/**: Sphinx documentation for the project.
- **tests/**: Unit and integration tests.
- **data/**: Data files (not tracked by git; add to .gitignore).
- **venv/**: Python virtual environment (add to .gitignore).

## 🚀 Getting Started

1. **Clone the repository**
   ```sh
   git clone https://github.com/meriem-amrane/Data-Science-portfolio.git
   cd Data-Science-portfolio/ecommerce-pipeline-project
   ```
2. **Create and activate a virtual environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # (ou venv\Scripts\activate sous Windows)
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## 📊 Features
- Automated ETL pipeline with Airflow
- Data cleaning and transformation
- Machine learning model training and evaluation
- Interactive dashboard for results visualization

## 📄 License
MIT

---

*For more details, see the documentation in the `docs/` folder.*
