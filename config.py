# config.py - 資料庫設定檔
import os
from dotenv import load_dotenv

load_dotenv()

# 從環境變數讀取資料庫設定
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'service'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': int(os.getenv('DB_PORT', 5432))
}

# 或者直接使用連接字串
DATABASE_URL = os.getenv('DATABASE_URL')