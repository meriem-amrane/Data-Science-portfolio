import pandas as pd

# Chargement des données
customers = pd.read_csv('data/olist_customers_dataset.csv')
geolocation = pd.read_csv('data/olist_geolocation_dataset.csv')
order_items = pd.read_csv('data/olist_order_items_dataset.csv')
order_payments = pd.read_csv('data/olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
orders = pd.read_csv('data/olist_orders_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')
sellers = pd.read_csv('data/olist_sellers_dataset.csv')
category_translation = pd.read_csv('data/product_category_name_translation.csv')

geo_mean = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
geo_mean.columns = ['customer_zip_code_prefix', 'lat', 'lng']

df = orders.merge(order_reviews, on='order_id', how='left') \
           .merge(order_items, on='order_id', how='left') \
           .merge(products, on='product_id', how='left') \
           .merge(category_translation, on='product_category_name', how='left') \
           .merge(sellers, on='seller_id', how='left') \
           .merge(order_payments, on='order_id', how='left') \
           .merge(customers, on='customer_id', how='left') \
           .merge(geo_mean, on='customer_zip_code_prefix', how='left')


df.to_csv('data/olist_dataset.csv', index=False)
print("✅ Fusion terminée. Fichier enregistré dans data/olist_dataset.csv")
