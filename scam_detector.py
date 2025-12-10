#!/usr/bin/env python3
import re
import jieba
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import psycopg2
from psycopg2.extras import RealDictCursor
import pickle
from datetime import datetime
from config import db_config


class ScamDetector:
    """詐騙 / 行銷 / 正常 訊息偵測器"""

    LABEL_MAP = {
        0: "normal",
        1: "marketing",
        2: "scam",
    }

    def __init__(self, db_config):
        self.db_config = db_config
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        )
        self.is_trained = False

        self.scam_keywords = [
            "匯款",
            "轉帳",
            "急需",
            "緊急",
            "家人出事",
            "車禍",
            "住院",
            "中獎",
            "恭喜",
            "免費",
            "限時",
            "優惠",
            "投資",
            "穩賺",
            "高報酬",
            "低風險",
            "保證獲利",
            "解除分期",
            "誤設定",
            "客服",
            "系統升級",
            "帳戶異常",
            "身分驗證",
            "點擊連結",
            "重設密碼",
            "領取",
            "獎金",
            "補助",
            "退稅",
            "包裹",
            "海關",
            "未領取",
            "重新配送",
            "確認資料",
            "輸入密碼",
        ]

        self.suspicious_patterns = [
            r"\d{10,}",
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            r"\$\d+|\d+元",
            r"line|LINE|賴",
        ]

    def connect_db(self):
        return psycopg2.connect(**self.db_config)

    def create_tables(self):
        conn = self.connect_db()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                content   TEXT      NOT NULL,
                is_scam   BOOLEAN,
                label_id  INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                message_id       INTEGER REFERENCES messages(id),
                scam_probability FLOAT,
                predicted_label  BOOLEAN,
                label_id         INTEGER,
                features         JSONB,
                predicted_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            "ALTER TABLE messages    ADD COLUMN IF NOT EXISTS label_id INTEGER"
        )
        cursor.execute(
            "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS label_id INTEGER"
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_is_scam       ON messages(is_scam);
            CREATE INDEX IF NOT EXISTS idx_messages_label_id      ON messages(label_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_message_id ON predictions(message_id);
            """
        )

        conn.commit()
        cursor.close()
        conn.close()
        print("✓ 資料表創建/更新完成")

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        words = jieba.cut(text)
        return " ".join(words)

    def extract_features(self, text: str) -> dict:
        features = {}
        keyword_count = sum(1 for keyword in self.scam_keywords if keyword in text)
        features["keyword_count"] = keyword_count
        features["has_keywords"] = keyword_count > 0

        pattern_count = 0
        for pattern in self.suspicious_patterns:
            pattern_count += len(re.findall(pattern, text))
        features["pattern_count"] = pattern_count

        features["text_length"] = len(text)
        features["has_url"] = bool(re.search(r"http[s]?://", text))
        features["has_phone"] = bool(re.search(r"\d{10}", text))
        features["has_money"] = bool(re.search(r"\$\d+|\d+元", text))
        features["exclamation_count"] = text.count("!")
        features["question_count"] = text.count("?")

        return features

    def load_training_data(self):
        conn = self.connect_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute(
            """
            SELECT 
                content,
                CASE 
                    WHEN label_id IS NOT NULL THEN label_id
                    WHEN is_scam = TRUE THEN 2
                    WHEN is_scam = FALSE THEN 0
                    ELSE NULL
                END AS label_id
            FROM messages
            WHERE (label_id IS NOT NULL) OR (is_scam IS NOT NULL)
            """
        )

        data = cursor.fetchall()
        cursor.close()
        conn.close()

        data = [row for row in data if row["label_id"] is not None]

        if not data:
            raise ValueError("資料庫中沒有可用的訓練資料")

        texts = [row["content"] for row in data]
        labels = [int(row["label_id"]) for row in data]

        return texts, labels

    def train(self):
        print("正在載入訓練資料（messages）...")
        texts, labels = self.load_training_data()

        total = len(labels)
        normal = sum(1 for x in labels if x == 0)
        marketing = sum(1 for x in labels if x == 1)
        scam = sum(1 for x in labels if x == 2)

        print(f"訓練資料數量: {total}")
        print(f"  正常(0): {normal}")
        print(f"  行銷(1): {marketing}")
        print(f"  詐騙(2): {scam}")

        self.train_from_lists(texts, labels)

    def train_from_lists(self, texts, labels):
        print("正在處理文本...")
        processed_texts = [self.preprocess_text(t) for t in texts]

        print("正在提取 TF-IDF 特徵...")
        X_tfidf = self.vectorizer.fit_transform(processed_texts)

        extra_features = []
        for t in texts:
            f = self.extract_features(t)
            extra_features.append(
                [
                    f["keyword_count"],
                    f["pattern_count"],
                    f["text_length"],
                    int(f["has_url"]),
                    int(f["has_phone"]),
                    int(f["has_money"]),
                    f["exclamation_count"],
                    f["question_count"],
                ]
            )

        X_combined = np.hstack([X_tfidf.toarray(), np.array(extra_features)])

        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) > 1 and min(counts) >= 2:
            stratify = labels
        else:
            stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X_combined,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        print("正在訓練多分類模型 (0/1/2)...")
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)

        used_labels = sorted(set(labels))
        target_names = [self.LABEL_MAP.get(l, str(l)) for l in used_labels]

        print("\n模型評估結果:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        print("\n混淆矩陣:")
        print(confusion_matrix(y_test, y_pred))

        self.is_trained = True
        print("\n✓ 模型訓練完成")

    def predict(self, text, save_to_db=True):
        if not self.is_trained:
            raise ValueError("模型尚未訓練，請先執行 train() 或 load_model()")

        processed_text = self.preprocess_text(text)
        X_tfidf = self.vectorizer.transform([processed_text])

        f = self.extract_features(text)
        extra_features = np.array(
            [
                [
                    f["keyword_count"],
                    f["pattern_count"],
                    f["text_length"],
                    int(f["has_url"]),
                    int(f["has_phone"]),
                    int(f["has_money"]),
                    f["exclamation_count"],
                    f["question_count"],
                ]
            ]
        )

        X_combined = np.hstack([X_tfidf.toarray(), extra_features])

        pred_label = int(self.classifier.predict(X_combined)[0])

        proba_map = {}
        scam_prob = 0.0
        confidence = 0.0

        if hasattr(self.classifier, "predict_proba"):
            prob_vec = self.classifier.predict_proba(X_combined)[0]
            classes = self.classifier.classes_
            for idx, cls in enumerate(classes):
                cls = int(cls)
                name = self.LABEL_MAP.get(cls, f"class_{cls}")
                p = float(prob_vec[idx])
                proba_map[name] = p
                if cls == pred_label:
                    confidence = p
                if cls == 2:
                    scam_prob = p

        if not proba_map:
            name = self.LABEL_MAP.get(pred_label, "unknown")
            proba_map[name] = 1.0
            confidence = 1.0
            scam_prob = 1.0 if pred_label == 2 else 0.0

        for lid, name in self.LABEL_MAP.items():
            proba_map.setdefault(name, 0.0)

        label_text = self.LABEL_MAP.get(pred_label, "unknown")
        warning_level = self._get_warning_level(pred_label, confidence)
        is_scam = pred_label == 2

        result = {
            "is_scam": is_scam,
            "scam_probability": scam_prob,
            "label": label_text,
            "label_id": pred_label,
            "probabilities": proba_map,
            "confidence": confidence,
            "features": f,
            "warning_level": warning_level,
        }

        if save_to_db:
            self._save_prediction(text, result)

        return result

    def _get_warning_level(self, label_id, confidence):
        if label_id == 2:
            if confidence >= 0.85:
                return "高危險"
            elif confidence >= 0.65:
                return "中度風險"
            elif confidence >= 0.4:
                return "低度風險"
            else:
                return "需留意"
        if label_id == 1:
            return "行銷資訊"
        if confidence >= 0.7:
            return "安全"
        return "大致安全"

    def _save_prediction(self, text, result):
        conn = self.connect_db()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO messages (content, is_scam, label_id)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (text, result["is_scam"], result["label_id"]),
        )
        message_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO predictions
            (message_id, scam_probability, predicted_label, label_id, features)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                message_id,
                result["scam_probability"],
                result["is_scam"],
                result["label_id"],
                json.dumps(result["features"]),
            ),
        )

        conn.commit()
        cursor.close()
        conn.close()

    def save_model(self, filename="scam_detector_model.pkl"):
        if not self.is_trained:
            raise ValueError("模型尚未訓練")

        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "scam_keywords": self.scam_keywords,
            "suspicious_patterns": self.suspicious_patterns,
        }

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        print(f"✓ 模型已儲存至 {filename}")

    def load_model(self, filename="scam_detector_model.pkl"):
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data["vectorizer"]
        self.classifier = model_data["classifier"]
        self.scam_keywords = model_data["scam_keywords"]
        self.suspicious_patterns = model_data["suspicious_patterns"]
        self.is_trained = True

        print(f"✓ 模型已從 {filename} 載入")


if __name__ == "__main__":
    detector = ScamDetector(db_config)
    detector.create_tables()
    print("\n三分類詐騙偵測系統就緒（請用 train.py 進行訓練）")
