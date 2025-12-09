#!/usr/bin/env python3
"""
詐騙訊息偵測系統 - 訓練腳本
===========================

功能：
1. 自動創建資料庫表結構
2. 支持從多種來源匯入訓練資料（CSV、手動添加）
3. 將所有數據儲存到 PostgreSQL 資料庫
4. 從資料庫載入數據進行訓練
5. 支持增量訓練（保留舊數據）
6. 儲存和載入訓練好的模型

使用方法：
    python train.py                    # 互動模式
    python train.py --import file.csv  # 直接匯入 CSV
    python train.py --train            # 直接從資料庫訓練
    python train.py --all file.csv     # 匯入並訓練
"""

import os
import sys
import csv
import argparse
from datetime import datetime

# 添加項目根目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import db_config


class DatabaseManager:
    """資料庫管理類 - 處理所有資料庫操作"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """連接資料庫"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config['port']
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"❌ 資料庫連接失敗: {e}")
            return False
    
    def close(self):
        """關閉資料庫連接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """創建所需的資料表"""
        print("\n📋 創建資料表...")
        
        # 訓練數據表 - 儲存所有訓練用的訊息
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                is_scam BOOLEAN NOT NULL,
                source VARCHAR(100) DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(content)
            )
        """)
        
        # 訊息表 - 儲存預測的訊息
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                is_scam BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 預測記錄表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                message_id INTEGER REFERENCES messages(id),
                scam_probability FLOAT,
                predicted_label BOOLEAN,
                features JSONB,
                predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 模型歷史表 - 記錄訓練歷史
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_history (
                id SERIAL PRIMARY KEY,
                model_version VARCHAR(50),
                training_samples INTEGER,
                scam_samples INTEGER,
                normal_samples INTEGER,
                accuracy FLOAT,
                model_path VARCHAR(255),
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)
        
        # 創建索引
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_data_is_scam ON training_data(is_scam);
            CREATE INDEX IF NOT EXISTS idx_training_data_source ON training_data(source);
            CREATE INDEX IF NOT EXISTS idx_messages_is_scam ON messages(is_scam);
            CREATE INDEX IF NOT EXISTS idx_predictions_message_id ON predictions(message_id);
        """)
        
        self.conn.commit()
        print("✓ 資料表創建完成")
    
    def import_csv(self, csv_file, source_name=None, clear_existing=False):
        """
        從 CSV 檔案匯入訓練資料到資料庫
        
        Args:
            csv_file: CSV 檔案路徑
            source_name: 資料來源名稱
            clear_existing: 是否清除現有數據
        
        Returns:
            tuple: (成功數, 重複數, 失敗數)
        """
        if not os.path.exists(csv_file):
            print(f"❌ 找不到檔案: {csv_file}")
            return 0, 0, 0
        
        if source_name is None:
            source_name = os.path.basename(csv_file)
        
        print(f"\n📁 正在讀取 {csv_file}...")
        
        if clear_existing:
            print("⚠️  清除現有訓練數據...")
            self.cursor.execute("DELETE FROM training_data")
            self.conn.commit()
        
        success_count = 0
        duplicate_count = 0
        error_count = 0
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        content = row.get('content', '').strip()
                        is_scam_raw = row.get('is_scam', '0')
                        
                        # 處理 is_scam 欄位（支持多種格式）
                        if isinstance(is_scam_raw, str):
                            is_scam = is_scam_raw.strip().lower() in ('1', 'true', 'yes', '是')
                        else:
                            is_scam = bool(is_scam_raw)
                        
                        if not content:
                            continue
                        
                        # 使用 UPSERT 避免重複
                        self.cursor.execute("""
                            INSERT INTO training_data (content, is_scam, source)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (content) DO UPDATE SET
                                is_scam = EXCLUDED.is_scam,
                                source = EXCLUDED.source,
                                updated_at = CURRENT_TIMESTAMP
                            RETURNING (xmax = 0) AS inserted
                        """, (content, is_scam, source_name))
                        
                        result = self.cursor.fetchone()
                        if result and result['inserted']:
                            success_count += 1
                        else:
                            duplicate_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        print(f"⚠️  處理行時出錯: {e}")
                
                self.conn.commit()
                
        except Exception as e:
            print(f"❌ 讀取 CSV 檔案失敗: {e}")
            return 0, 0, 0
        
        print(f"✓ 匯入完成:")
        print(f"  新增: {success_count} 筆")
        print(f"  更新: {duplicate_count} 筆")
        print(f"  錯誤: {error_count} 筆")
        
        return success_count, duplicate_count, error_count
    
    def add_training_sample(self, content, is_scam, source='manual'):
        """
        添加單筆訓練資料
        
        Args:
            content: 訊息內容
            is_scam: 是否為詐騙
            source: 資料來源
            
        Returns:
            bool: 是否成功添加
        """
        try:
            self.cursor.execute("""
                INSERT INTO training_data (content, is_scam, source)
                VALUES (%s, %s, %s)
                ON CONFLICT (content) DO UPDATE SET
                    is_scam = EXCLUDED.is_scam,
                    source = EXCLUDED.source,
                    updated_at = CURRENT_TIMESTAMP
            """, (content.strip(), is_scam, source))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ 添加失敗: {e}")
            return False
    
    def get_training_data(self):
        """
        從資料庫獲取所有訓練資料
        
        Returns:
            list: 訓練資料列表
        """
        self.cursor.execute("""
            SELECT content, is_scam 
            FROM training_data 
            ORDER BY id
        """)
        return self.cursor.fetchall()
    
    def get_statistics(self):
        """獲取訓練資料統計"""
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_scam THEN 1 ELSE 0 END) as scam_count,
                SUM(CASE WHEN NOT is_scam THEN 1 ELSE 0 END) as normal_count
            FROM training_data
        """)
        return self.cursor.fetchone()
    
    def get_source_statistics(self):
        """獲取各來源的數據統計"""
        self.cursor.execute("""
            SELECT 
                source,
                COUNT(*) as total,
                SUM(CASE WHEN is_scam THEN 1 ELSE 0 END) as scam_count,
                SUM(CASE WHEN NOT is_scam THEN 1 ELSE 0 END) as normal_count
            FROM training_data
            GROUP BY source
            ORDER BY total DESC
        """)
        return self.cursor.fetchall()
    
    def record_training(self, training_samples, scam_samples, normal_samples, 
                       accuracy, model_path, notes=''):
        """記錄訓練歷史"""
        version = datetime.now().strftime('v%Y%m%d_%H%M%S')
        
        self.cursor.execute("""
            INSERT INTO model_history 
            (model_version, training_samples, scam_samples, normal_samples, 
             accuracy, model_path, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (version, training_samples, scam_samples, normal_samples,
              accuracy, model_path, notes))
        
        self.conn.commit()
        return version
    
    def sync_to_messages_table(self):
        """將訓練數據同步到 messages 表（兼容舊版 scam_detector）"""
        print("\n🔄 同步數據到 messages 表...")
        
        # 清除 messages 表中的訓練數據
        self.cursor.execute("DELETE FROM messages WHERE is_scam IS NOT NULL")
        
        # 從 training_data 複製到 messages
        self.cursor.execute("""
            INSERT INTO messages (content, is_scam, created_at)
            SELECT content, is_scam, created_at
            FROM training_data
        """)
        
        self.conn.commit()
        print("✓ 數據同步完成")


class ModelTrainer:
    """模型訓練器"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.detector = None
    
    def train(self, model_path='scam_detector_model.pkl'):
        """
        訓練模型
        
        Args:
            model_path: 主要模型儲存路徑
            
        Returns:
            bool: 是否訓練成功
        """
        print("\n" + "="*60)
        print("🚀 開始訓練模型")
        print("="*60)
        
        # 先同步數據到 messages 表（兼容現有的 ScamDetector）
        self.db_manager.sync_to_messages_table()
        
        # 獲取統計
        stats = self.db_manager.get_statistics()
        if not stats or stats['total'] == 0:
            print("❌ 錯誤: 資料庫中沒有訓練資料")
            print("請先使用 --import 匯入訓練資料")
            return False
        
        print(f"\n📊 訓練資料統計:")
        print(f"  總計: {stats['total']} 筆")
        print(f"  詐騙訊息: {stats['scam_count']} 筆")
        print(f"  正常訊息: {stats['normal_count']} 筆")
        
        if stats['scam_count'] == 0 or stats['normal_count'] == 0:
            print("❌ 錯誤: 需要同時有詐騙和正常訊息才能訓練")
            return False
        
        # 載入 ScamDetector
        try:
            from scam_detector import ScamDetector
            self.detector = ScamDetector(db_config)
        except ImportError as e:
            print(f"❌ 無法載入 ScamDetector: {e}")
            return False
        
        # 訓練模型
        try:
            print("\n⏳ 正在訓練...")
            self.detector.train()
            
            # 確保 models 目錄存在
            os.makedirs('models', exist_ok=True)
            
            # 生成時間戳記檔名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_model_path = f'models/{timestamp}.pkl'
            
            # 儲存模型到主要路徑 (scam_detector_model.pkl)
            print(f"\n💾 正在儲存模型到 {model_path}...")
            self.detector.save_model(model_path)
            
            # 同時儲存一份到 models/ 目錄（帶時間戳記）
            print(f"💾 正在儲存模型備份到 {versioned_model_path}...")
            self.detector.save_model(versioned_model_path)
            
            # 記錄訓練歷史
            version = self.db_manager.record_training(
                training_samples=stats['total'],
                scam_samples=stats['scam_count'],
                normal_samples=stats['normal_count'],
                accuracy=0.0,  # 可以從訓練結果中提取
                model_path=model_path,
                notes=f"Trained with {stats['total']} samples. Backup: {versioned_model_path}"
            )
            
            print(f"\n✅ 訓練完成！模型版本: {version}")
            print(f"   主要模型: {model_path}")
            print(f"   備份模型: {versioned_model_path}")
            return True
            
        except Exception as e:
            print(f"❌ 訓練失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model(self, model_path='scam_detector_model.pkl'):
        """測試訓練好的模型"""
        print("\n" + "="*60)
        print("🧪 模型測試")
        print("="*60)
        
        try:
            from scam_detector import ScamDetector
            detector = ScamDetector(db_config)
            detector.load_model(model_path)
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            return
        
        # 測試案例
        test_cases = [
            ("恭喜中獎！請立即匯款領取獎金", True),
            ("明天下午開會記得帶報告", False),
            ("您的包裹需要補繳關稅800元 http://fake.com", True),
            ("週末要去爬山嗎？", False),
            ("緊急！家人出事需要手術費20萬", True),
            ("今天天氣不錯", False),
            ("投資比特幣，保證月收益35%，加LINE: crypto888", True),
            ("下班後要一起吃飯嗎？", False),
        ]
        
        print("\n測試結果:")
        correct = 0
        
        for msg, expected_scam in test_cases:
            result = detector.predict(msg, save_to_db=False)
            is_correct = result['is_scam'] == expected_scam
            
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"\n{status} {'詐騙' if expected_scam else '正常'} → 預測: {'詐騙' if result['is_scam'] else '正常'} ({result['scam_probability']:.1%})")
            print(f"   {msg[:50]}{'...' if len(msg) > 50 else ''}")
        
        accuracy = correct / len(test_cases) * 100
        print(f"\n📈 測試準確率: {accuracy:.1f}% ({correct}/{len(test_cases)})")


def interactive_mode(db_manager, trainer):
    """互動模式"""
    while True:
        print("\n" + "="*60)
        print("🤖 詐騙訊息偵測系統 - 訓練管理")
        print("="*60)
        
        # 顯示統計
        stats = db_manager.get_statistics()
        if stats and stats['total'] > 0:
            print(f"\n📊 資料庫現有數據:")
            print(f"   總計: {stats['total']} 筆 | 詐騙: {stats['scam_count']} | 正常: {stats['normal_count']}")
        else:
            print("\n📊 資料庫目前沒有訓練數據")
        
        print("\n選項:")
        print("  1. 從 CSV 匯入訓練資料")
        print("  2. 手動添加訓練樣本")
        print("  3. 從資料庫訓練模型")
        print("  4. 測試現有模型")
        print("  5. 查看數據來源統計")
        print("  0. 退出")
        
        choice = input("\n請選擇 (0-5): ").strip()
        
        if choice == '0':
            print("\n👋 再見！")
            break
            
        elif choice == '1':
            csv_file = input("請輸入 CSV 檔案路徑: ").strip()
            if csv_file:
                clear = input("是否清除現有數據？(y/N): ").strip().lower() == 'y'
                db_manager.import_csv(csv_file, clear_existing=clear)
            
        elif choice == '2':
            print("\n手動添加訓練樣本（輸入 'q' 退出）")
            while True:
                content = input("\n訊息內容: ").strip()
                if content.lower() == 'q':
                    break
                
                is_scam_input = input("是否為詐騙？(y/N): ").strip().lower()
                is_scam = is_scam_input == 'y'
                
                if db_manager.add_training_sample(content, is_scam):
                    print(f"✓ 已添加: {'詐騙' if is_scam else '正常'}訊息")
            
        elif choice == '3':
            trainer.train()
            
        elif choice == '4':
            trainer.test_model()
            
        elif choice == '5':
            sources = db_manager.get_source_statistics()
            if sources:
                print("\n📊 數據來源統計:")
                print("-" * 50)
                for s in sources:
                    print(f"  {s['source']}: {s['total']} 筆 (詐騙: {s['scam_count']}, 正常: {s['normal_count']})")
            else:
                print("目前沒有數據")


def main():
    parser = argparse.ArgumentParser(
        description='詐騙訊息偵測系統 - 訓練腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python train.py                        # 互動模式
  python train.py --import data.csv      # 匯入 CSV 到資料庫
  python train.py --train                # 從資料庫訓練模型
  python train.py --all data.csv         # 匯入 CSV 並訓練
  python train.py --test                 # 測試現有模型
  python train.py --stats                # 查看數據統計
        """
    )
    
    parser.add_argument('--import', dest='import_file', metavar='CSV',
                       help='從 CSV 檔案匯入訓練資料')
    parser.add_argument('--train', action='store_true',
                       help='從資料庫訓練模型')
    parser.add_argument('--test', action='store_true',
                       help='測試現有模型')
    parser.add_argument('--all', metavar='CSV',
                       help='匯入 CSV 並訓練（一鍵完成）')
    parser.add_argument('--stats', action='store_true',
                       help='顯示數據統計')
    parser.add_argument('--clear', action='store_true',
                       help='匯入前清除現有數據')
    parser.add_argument('--model', default='scam_detector_model.pkl',
                       help='模型檔案路徑 (預設: scam_detector_model.pkl)')
    
    args = parser.parse_args()
    
    # 初始化資料庫管理器
    print("🔌 正在連接資料庫...")
    db_manager = DatabaseManager(db_config)
    
    if not db_manager.connect():
        sys.exit(1)
    
    print("✓ 資料庫連接成功")
    
    # 創建資料表
    db_manager.create_tables()
    
    # 初始化訓練器
    trainer = ModelTrainer(db_manager)
    
    try:
        # 處理命令列參數
        if args.all:
            # 一鍵完成：匯入 + 訓練
            db_manager.import_csv(args.all, clear_existing=args.clear)
            trainer.train(args.model)
            trainer.test_model(args.model)
            
        elif args.import_file:
            db_manager.import_csv(args.import_file, clear_existing=args.clear)
                    
        elif args.train:
            trainer.train(args.model)
            
        elif args.test:
            trainer.test_model(args.model)
            
        elif args.stats:
            stats = db_manager.get_statistics()
            if stats and stats['total'] > 0:
                print(f"\n📊 資料庫統計:")
                print(f"  總計: {stats['total']} 筆")
                print(f"  詐騙: {stats['scam_count']} 筆")
                print(f"  正常: {stats['normal_count']} 筆")
                
                sources = db_manager.get_source_statistics()
                if sources:
                    print(f"\n📁 來源統計:")
                    for s in sources:
                        print(f"  {s['source']}: {s['total']} 筆")
            else:
                print("資料庫目前沒有訓練數據")
                
        else:
            # 無參數 -> 互動模式
            interactive_mode(db_manager, trainer)
            
    finally:
        db_manager.close()
        print("\n🔌 資料庫連接已關閉")


if __name__ == "__main__":
    main()