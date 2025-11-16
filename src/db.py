# src/db.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 從 .env 讀取設定
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "nis_ai")

# 這裡要放「MySQL 裡的帳號」
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "AdminNIS2025!")

# MySQL 連線字串（用 mysqlconnector 驅動）
SQLALCHEMY_DATABASE_URL = (
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=False,          # 要看 SQL log 再改 True
    pool_pre_ping=True,  # 避免連線掛掉
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
