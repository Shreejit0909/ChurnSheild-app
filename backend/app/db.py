# backend/app/db.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

# ---------- Database Path ----------
DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# ---------- Engine & Session ----------
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ---------- Model ----------
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    churn_probability = Column(Float)
    risk_label = Column(String, index=True)
    features = Column(Text)  # JSON-encoded features
    created_at = Column(DateTime, default=datetime.utcnow)

    # --- New fields for batch tracking + retention actions ---
    batch_id = Column(String, index=True, nullable=True)     # UUID for batch uploads
    source = Column(String, default="ui")                    # e.g. "ui", "batch", "auto"
    action = Column(String, nullable=True)                   # recommended action (discount, call, etc.)
    action_status = Column(String, default="pending")        # pending / applied / failed
    contacted_at = Column(DateTime, nullable=True)           # when customer was contacted

# ---------- Create tables ----------
Base.metadata.create_all(bind=engine)
