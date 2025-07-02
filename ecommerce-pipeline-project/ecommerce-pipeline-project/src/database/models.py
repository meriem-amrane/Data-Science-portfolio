from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from .connection import Base

class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True)
    customer_unique_id = Column(String, nullable=False)
    customer_zip_code_prefix = Column(String)
    customer_city = Column(String)
    customer_state = Column(String)

    # Relationships
    orders = relationship("Order", back_populates="customer")

class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String, primary_key=True)
    customer_id = Column(String, ForeignKey('customers.customer_id'))
    order_status = Column(String)
    order_purchase_timestamp = Column(DateTime)
    order_approved_at = Column(DateTime)
    order_delivered_carrier_date = Column(DateTime)
    order_delivered_customer_date = Column(DateTime)
    order_estimated_delivery_date = Column(DateTime)

    # Relationships
    customer = relationship("Customer", back_populates="orders")
    order_items = relationship("OrderItem", back_populates="order")
    payments = relationship("Payment", back_populates="order")
    reviews = relationship("Review", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True)
    order_id = Column(String, ForeignKey('orders.order_id'))
    product_id = Column(String, ForeignKey('products.product_id'))
    seller_id = Column(String, ForeignKey('sellers.seller_id'))
    shipping_limit_date = Column(DateTime)
    price = Column(Float)
    freight_value = Column(Float)

    # Relationships
    order = relationship("Order", back_populates="order_items")
    product = relationship("Product", back_populates="order_items")
    seller = relationship("Seller", back_populates="order_items")

class Product(Base):
    __tablename__ = "products"

    product_id = Column(String, primary_key=True)
    product_category_name = Column(String)
    product_name_lenght = Column(Integer)
    product_description_lenght = Column(Integer)
    product_photos_qty = Column(Integer)
    product_weight_g = Column(Float)
    product_length_cm = Column(Float)
    product_height_cm = Column(Float)
    product_width_cm = Column(Float)

    # Relationships
    order_items = relationship("OrderItem", back_populates="product")

class Seller(Base):
    __tablename__ = "sellers"

    seller_id = Column(String, primary_key=True)
    seller_zip_code_prefix = Column(String)
    seller_city = Column(String)
    seller_state = Column(String)

    # Relationships
    order_items = relationship("OrderItem", back_populates="seller")

class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True)
    order_id = Column(String, ForeignKey('orders.order_id'))
    payment_sequential = Column(Integer)
    payment_type = Column(String)
    payment_installments = Column(Integer)
    payment_value = Column(Float)

    # Relationships
    order = relationship("Order", back_populates="payments")

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True)
    order_id = Column(String, ForeignKey('orders.order_id'))
    review_score = Column(Integer)
    review_comment_title = Column(String)
    review_comment_message = Column(String)
    review_creation_date = Column(DateTime)
    review_answer_timestamp = Column(DateTime)

    # Relationships
    order = relationship("Order", back_populates="reviews") 