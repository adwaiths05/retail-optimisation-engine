from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector
import datetime

class Base(DeclarativeBase):
    pass

class Department(Base):
    __tablename__ = "departments"
    department_id = Column(Integer, primary_key=True)
    department = Column(String, nullable=False)
    products = relationship("Product", back_populates="department")

class Aisle(Base):
    __tablename__ = "aisles"
    aisle_id = Column(Integer, primary_key=True)
    aisle = Column(String, nullable=False)
    products = relationship("Product", back_populates="aisle")

class Product(Base):
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True)
    product_name = Column(String)
    
    # Linked to parent tables
    aisle_id = Column(Integer, ForeignKey("aisles.aisle_id"))
    department_id = Column(Integer, ForeignKey("departments.department_id"))
    
    # Business Metrics
    price = Column(Float, nullable=False)
    margin = Column(Float, nullable=False)
    stock = Column(Integer, default=100)
    embedding = Column(Vector(64)) 

    # Relationships
    aisle = relationship("Aisle", back_populates="products")
    department = relationship("Department", back_populates="products")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id = Column(Integer, primary_key=True)
    avg_margin_preference = Column(Float, default=0.0)
    total_purchases = Column(Integer, default=0)
    last_active = Column(DateTime, default=datetime.datetime.utcnow)

class ExperimentAssignment(Base):
    __tablename__ = "experiment_assignments"
    user_id = Column(Integer, primary_key=True)
    group_name = Column(String, nullable=False) 
    experiment_id = Column(String, primary_key=True)
    assigned_at = Column(DateTime, default=datetime.datetime.utcnow)

class ExperimentEvent(Base):
    __tablename__ = "experiment_events"
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer)
    experiment_id = Column(String)
    event_type = Column(String)
    product_id = Column(Integer)
    revenue = Column(Float, default=0.0)
    margin = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)