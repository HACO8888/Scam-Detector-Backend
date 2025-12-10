#!/usr/bin/env python3
"""
詐騙 / 行銷 / 一般 訊息偵測系統 - 訓練腳本
=====================================

功能：
1. 自動創建 / 更新資料庫表結構（含 label_id）
2. 從 CSV 匯入訓練資料 → 寫進 training_data
3. 將 training_data 同步到 messages（保留 label_id）
4. 從資料庫載入數據訓練 ScamDetector（目前仍用原本 train()，之後可升級三分類）
5. 支持互動模式 / 指令模式
6. 模型儲存為：
   - 主模型： scam_detector_model.pkl
   - 備份：   models/{timestamp}.pkl
   並在 model_history 紀錄一次訓練資訊

使用方法：
    python train.py                        # 互動模式
    python train.py --import data.csv      # 匯入 CSV 到資料庫
    python train.py --train                # 從資料庫訓練模型
    python train.py --all data.csv         # 匯入 CSV 並訓練
    python train.py --test                 # 測試現有模型
    python train.py --stats                # 查看數據統計
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
                host=self.db_config["host"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                port=self.db_config["port"],
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

    # -----------------------------
    # 資料表 / Migrate（包含 label_id）
    # -----------------------------
    def create_tables(self):
        """創建 / 更新所需的資料表"""
        print("\n📋 檢查 / 創建資料表...")

        # 訓練數據表 - 儲存所有訓練用的訊息
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_data (
                id SERIAL PRIMARY KEY,
                content   TEXT       NOT NULL,
                is_scam   BOOLEAN    NOT NULL,
                label_id  INTEGER,
                source    VARCHAR(100) DEFAULT 'manual',
                created_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(content)
            )
        """
        )

        # 訊息表 - 儲存預測 / 訓練的訊息
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                content   TEXT     NOT NULL,
                is_scam   BOOLEAN,
                label_id  INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # 預測記錄表
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                message_id        INTEGER REFERENCES messages(id),
                scam_probability  FLOAT,
                predicted_label   BOOLEAN,
                features          JSONB,
                predicted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # 模型歷史表 - 記錄訓練歷史
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_history (
                id SERIAL PRIMARY KEY,
                model_version     VARCHAR(50),
                training_samples  INTEGER,
                scam_samples      INTEGER,
                normal_samples    INTEGER,
                accuracy          FLOAT,
                model_path        VARCHAR(255),
                trained_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes             TEXT
            )
        """
        )

        # 兼容舊 DB：補上缺少的欄位（簡單 migrate）
        self.cursor.execute(
            "ALTER TABLE training_data ADD COLUMN IF NOT EXISTS label_id INTEGER"
        )
        self.cursor.execute(
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS label_id INTEGER"
        )
        self.cursor.execute(
            "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS predicted_label INTEGER"
        )

        # 創建索引
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_training_data_is_scam   ON training_data(is_scam);
            CREATE INDEX IF NOT EXISTS idx_training_data_source    ON training_data(source);
            CREATE INDEX IF NOT EXISTS idx_training_data_label_id  ON training_data(label_id);
            CREATE INDEX IF NOT EXISTS idx_messages_is_scam        ON messages(is_scam);
            CREATE INDEX IF NOT EXISTS idx_messages_label_id       ON messages(label_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_message_id  ON predictions(message_id);
        """
        )

        self.conn.commit()
        print("✓ 資料表 / Migrate 完成")

    # -----------------------------
    # CSV → training_data
    # -----------------------------
    def import_csv(self, csv_file, source_name=None, clear_existing=False):
        """
        從 CSV 檔案匯入訓練資料到 training_data

        支援兩種格式：
        1) content,is_scam   (0/1)
        2) content,label     (0=normal,1=marketing,2=scam)
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
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]
                has_label = "label" in fieldnames

                for row in reader:
                    try:
                        content = row.get("content", "").strip()
                        if not content:
                            continue

                        label_id = None
                        is_scam = None

                        if has_label:
                            # 三分類
                            raw_label = row.get("label", "").strip()
                            if raw_label == "":
                                continue
                            label_id = int(raw_label)
                            if label_id == 2:
                                is_scam = True
                            elif label_id == 0:
                                is_scam = False
                            else:
                                is_scam = None  # marketing 就不要亂寫 is_scam
                        else:
                            # 只有 is_scam → 轉成 0/2
                            is_scam_raw = row.get("is_scam", "0")
                            if isinstance(is_scam_raw, str):
                                is_scam = is_scam_raw.strip().lower() in ("1", "true", "yes", "是")
                            else:
                                is_scam = bool(is_scam_raw)
                            label_id = 2 if is_scam else 0

                        self.cursor.execute(
                            """
                            INSERT INTO training_data (content, is_scam, label_id, source)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (content) DO UPDATE SET
                                is_scam   = EXCLUDED.is_scam,
                                label_id  = EXCLUDED.label_id,
                                source    = EXCLUDED.source,
                                updated_at = CURRENT_TIMESTAMP
                            RETURNING (xmax = 0) AS inserted
                            """,
                            (content, is_scam, label_id, source_name),
                        )

                        result = self.cursor.fetchone()
                        if result and result["inserted"]:
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

    def add_training_sample(self, content, is_scam, source="manual", label_id=None):
        """
        添加單筆訓練資料

        is_scam: 是否為詐騙（True/False）
        label_id: 0=normal,1=marketing,2=scam（可選，預設：2 if is_scam else 0）
        """
        try:
            if label_id is None:
                label_id = 2 if is_scam else 0

            self.cursor.execute(
                """
                INSERT INTO training_data (content, is_scam, label_id, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (content) DO UPDATE SET
                    is_scam   = EXCLUDED.is_scam,
                    label_id  = EXCLUDED.label_id,
                    source    = EXCLUDED.source,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (content.strip(), is_scam, label_id, source),
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ 添加失敗: {e}")
            return False

    def get_training_data(self):
        """從資料庫獲取所有訓練資料"""
        self.cursor.execute(
            """
            SELECT content, is_scam, label_id
            FROM training_data
            ORDER BY id
        """
        )
        return self.cursor.fetchall()

    def get_statistics(self):
        """獲取訓練資料統計（含 normal / marketing / scam）"""
        self.cursor.execute(
            """
            SELECT 
                COUNT(*) AS total,
                SUM(CASE WHEN COALESCE(label_id, CASE WHEN is_scam THEN 2 ELSE 0 END) = 2 THEN 1 ELSE 0 END) AS scam_count,
                SUM(CASE WHEN COALESCE(label_id, CASE WHEN is_scam THEN 2 ELSE 0 END) = 1 THEN 1 ELSE 0 END) AS marketing_count,
                SUM(CASE WHEN COALESCE(label_id, CASE WHEN is_scam THEN 2 ELSE 0 END) = 0 THEN 1 ELSE 0 END) AS normal_count
            FROM training_data
        """
        )
        return self.cursor.fetchone()

    def get_source_statistics(self):
        """獲取各來源的數據統計"""
        self.cursor.execute(
            """
            SELECT 
                source,
                COUNT(*) as total,
                SUM(CASE WHEN COALESCE(label_id, CASE WHEN is_scam THEN 2 ELSE 0 END) = 2 THEN 1 ELSE 0 END) AS scam_count,
                SUM(CASE WHEN COALESCE(label_id, CASE WHEN is_scam THEN 2 ELSE 0 END) = 1 THEN 1 ELSE 0 END) AS marketing_count,
                SUM(CASE WHEN COALESCE(label_id, CASE WHEN is_scam THEN 2 ELSE 0 END) = 0 THEN 1 ELSE 0 END) AS normal_count
            FROM training_data
            GROUP BY source
            ORDER BY total DESC
        """
        )
        return self.cursor.fetchall()

    def record_training(
        self,
        training_samples,
        scam_samples,
        normal_samples,
        accuracy,
        model_path,
        notes="",
    ):
        """記錄訓練歷史（沿用原本 schema：normal_samples = 非詐騙總數）"""
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        self.cursor.execute(
            """
            INSERT INTO model_history 
            (model_version, training_samples, scam_samples, normal_samples, 
             accuracy, model_path, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """,
            (version, training_samples, scam_samples, normal_samples, accuracy, model_path, notes),
        )

        self.conn.commit()
        return version

    # --------------------------------------
    # 將 training_data 同步到 messages 表
    # --------------------------------------
def sync_to_messages_table(self):
    """將訓練數據同步到 messages 表（給 ScamDetector 用）"""
    print("\n🔄 同步 training_data → messages ...")

    # 只刪「我們塞進來當訓練用」的那批，不動預測紀錄
    self.cursor.execute("DELETE FROM messages WHERE label_id IS NOT NULL OR is_scam IS NOT NULL")

    self.cursor.execute(
        """
        INSERT INTO messages (content, is_scam, label_id, created_at, updated_at)
        SELECT 
            content,
            is_scam,
            label_id,
            created_at,
            updated_at
        FROM training_data
        WHERE label_id IS NOT NULL
        """
    )

    self.conn.commit()
    print("✓ 數據同步完成")



# -------------------------------------------------
# 模型訓練器（仍呼叫你原本的 ScamDetector.train()）
# -------------------------------------------------
from scam_detector import ScamDetector


class ModelTrainer:
    """模型訓練器（真正用 DB 裡的 messages / label_id 訓練）"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.detector = ScamDetector(db_config)

    def train(self, model_path="scam_detector_model.pkl"):
        print("\n" + "=" * 60)
        print("🚀 從資料庫訓練三分類模型 (0=normal, 1=marketing, 2=scam)")
        print("=" * 60)

        # 1. 先把 training_data 同步到 messages
        self.db_manager.sync_to_messages_table()

        # 2. 交給 ScamDetector 自己從 messages 抓資料 & 訓練
        self.detector.train()

        # 3. 儲存模型（沿用你原本的方式）
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_model_path = f"models/{timestamp}.pkl"

        print(f"\n💾 正在儲存模型到 {model_path}...")
        self.detector.save_model(model_path)

        print(f"💾 正在儲存模型備份到 {versioned_model_path}...")
        self.detector.save_model(versioned_model_path)

        print("\n✅ 訓練完成！")
        return True

    def test_model(self, model_path="scam_detector_model.pkl"):
        """
        三分類模型測試：
        0 = normal
        1 = marketing
        2 = scam
        """
        print("\n" + "=" * 60)
        print("🧪 三分類模型測試")
        print("=" * 60)

        # 載入模型
        try:
            detector = ScamDetector(db_config)
            detector.load_model(model_path)
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return

        # 三分類測試資料
        test_cases = [
            ("恭喜中獎！請立即匯款領取獎金", 2),
            ("您的包裹需要補繳關稅800元 http://fake.com", 2),
            ("緊急！家人出事需要手術費20萬", 2),
            ("投資比特幣，保證月收益35%，加 LINE: crypto888", 2),
            ("⚠️ 您的帳戶異常，請立即登入驗證以免停用", 2),

            ("明天下午開會記得帶報告", 0),
            ("週末要去爬山嗎？", 0),
            ("今天天氣不錯", 0),
            ("下班後要一起吃飯嗎？", 0),
            ("我晚點會把檔案傳給你", 0),

            ("本週限時優惠，全館 5 折起！", 1),
            ("立即註冊會員，享受獨家折扣", 1),
            ("新品上市，買一送一活動開跑！", 1),
            ("訂閱我們的電子報，獲得最新促銷消息", 1),
            ("加入會員抽 iPhone 15！", 1),
        ]

        LABEL_MAP = ScamDetector.LABEL_MAP   # 0/1/2 → normal/marketing/scam

        print("\n📌 測試案例共：", len(test_cases), "筆")

        correct = 0

        for text, expected_label in test_cases:
            result = detector.predict(text, save_to_db=False)

            pred_label = result["label_id"]
            prob = result["probabilities"].get(LABEL_MAP[pred_label], 0.0)

            expected_text = LABEL_MAP[expected_label]
            pred_text = LABEL_MAP[pred_label]

            is_correct = (pred_label == expected_label)

            mark = "✓" if is_correct else "✗"
            if is_correct:
                correct += 1

            print("\n" + "-" * 60)
            print(f"{mark} 期望：{expected_text} ({expected_label}) → 預測：{pred_text} ({pred_label})")
            print(f"📨 訊息：{text}")
            print(f"📊 預測機率：{prob:.2%}")
            print(f"🔎 完整機率：{result['probabilities']}")

        accuracy = correct / len(test_cases)

        print("\n" + "=" * 60)
        print(f"📈 測試準確率：{accuracy:.1%} ({correct}/{len(test_cases)})")
        print("=" * 60)


# ------------------------------
# 互動模式
# ------------------------------
def interactive_mode(db_manager: DatabaseManager, trainer: ModelTrainer):
    """互動模式"""
    while True:
        print("\n" + "=" * 60)
        print("🤖 詐騙 / 行銷 訊息偵測系統 - 訓練管理")
        print("=" * 60)

        # 顯示統計
        stats = db_manager.get_statistics()
        if stats and stats["total"] > 0:
            print(f"\n📊 資料庫現有數據 (training_data):")
            print(
                f"   總計: {stats['total']} 筆 | "
                f"正常(0): {stats['normal_count']} | "
                f"行銷(1): {stats['marketing_count']} | "
                f"詐騙(2): {stats['scam_count']}"
            )
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

        if choice == "0":
            print("\n👋 再見！")
            break

        elif choice == "1":
            csv_file = input("請輸入 CSV 檔案路徑: ").strip()
            if csv_file:
                clear = (
                    input("是否清除現有數據？(y/N): ").strip().lower() == "y"
                )
                db_manager.import_csv(csv_file, clear_existing=clear)

        elif choice == "2":
            print("\n手動添加訓練樣本（輸入 'q' 退出）")
            while True:
                content = input("\n訊息內容: ").strip()
                if content.lower() == "q":
                    break

                label_raw = input(
                    "標籤：0=正常 / 1=行銷 / 2=詐騙（預設 2）: "
                ).strip()
                if label_raw == "":
                    label_id = 2
                else:
                    label_id = int(label_raw)

                is_scam = label_id == 2

                if db_manager.add_training_sample(
                    content, is_scam, source="manual", label_id=label_id
                ):
                    txt = "詐騙" if label_id == 2 else ("行銷" if label_id == 1 else "正常")
                    print(f"✓ 已添加: {txt} 訊息")

        elif choice == "3":
            trainer.train()

        elif choice == "4":
            trainer.test_model()

        elif choice == "5":
            sources = db_manager.get_source_statistics()
            if sources:
                print("\n📊 數據來源統計:")
                print("-" * 50)
                for s in sources:
                    print(
                        f"  {s['source']}: {s['total']} 筆 "
                        f"(正常: {s['normal_count']}, 行銷: {s['marketing_count']}, 詐騙: {s['scam_count']})"
                    )
            else:
                print("目前沒有數據")


# ------------------------------
# CLI 入口
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="詐騙 / 行銷 訊息偵測系統 - 訓練腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python train.py                        # 互動模式
  python train.py --import data.csv      # 匯入 CSV 到資料庫
  python train.py --train                # 從資料庫訓練模型
  python train.py --all data.csv         # 匯入 CSV 並訓練
  python train.py --test                 # 測試現有模型
  python train.py --stats                # 查看數據統計
        """,
    )

    parser.add_argument(
        "--import",
        dest="import_file",
        metavar="CSV",
        help="從 CSV 檔案匯入訓練資料",
    )
    parser.add_argument(
        "--train", action="store_true", help="從資料庫訓練模型"
    )
    parser.add_argument(
        "--test", action="store_true", help="測試現有模型"
    )
    parser.add_argument(
        "--all",
        metavar="CSV",
        help="匯入 CSV 並訓練（一鍵完成）",
    )
    parser.add_argument(
        "--stats", action="store_true", help="顯示數據統計"
    )
    parser.add_argument(
        "--clear", action="store_true", help="匯入前清除現有數據"
    )
    parser.add_argument(
        "--model",
        default="scam_detector_model.pkl",
        help="模型檔案路徑 (預設: scam_detector_model.pkl)",
    )

    args = parser.parse_args()

    # 初始化資料庫管理器
    print("🔌 正在連接資料庫...")
    db_manager = DatabaseManager(db_config)

    if not db_manager.connect():
        sys.exit(1)

    print("✓ 資料庫連接成功")

    # 創建 / Migrate 資料表
    db_manager.create_tables()

    # 初始化訓練器
    trainer = ModelTrainer(db_manager)

    try:
        # 處理命令列參數
        if args.all:
            # 一鍵完成：匯入 + 訓練 + 測試
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
            if stats and stats["total"] > 0:
                print(f"\n📊 資料庫統計:")
                print(
                    f"  總計: {stats['total']} 筆 | "
                    f"正常(0): {stats['normal_count']} | "
                    f"行銷(1): {stats['marketing_count']} | "
                    f"詐騙(2): {stats['scam_count']}"
                )

                sources = db_manager.get_source_statistics()
                if sources:
                    print(f"\n📁 來源統計:")
                    for s in sources:
                        print(
                            f"  {s['source']}: {s['total']} 筆 "
                            f"(正常: {s['normal_count']}, 行銷: {s['marketing_count']}, 詐騙: {s['scam_count']})"
                        )
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
