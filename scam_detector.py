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
    """詐騙訊息偵測器"""
    
    def __init__(self, db_config):
        """
        初始化偵測器
        
        Args:
            db_config: PostgreSQL 資料庫連線設定
        """
        self.db_config = db_config
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # 詐騙關鍵字特徵（台灣常見詐騙用語）
        self.scam_keywords = [
            '匯款', '轉帳', '急需', '緊急', '家人出事', '車禍', '住院',
            '中獎', '恭喜', '免費', '限時', '優惠', '投資', '穩賺',
            '高報酬', '低風險', '保證獲利', '解除分期', '誤設定',
            '客服', '系統升級', '帳戶異常', '身分驗證', '點擊連結',
            '重設密碼', '領取', '獎金', '補助', '退稅', '包裹',
            '海關', '未領取', '重新配送', '確認資料', '輸入密碼'
        ]
        
        # 可疑模式
        self.suspicious_patterns = [
            r'\d{10,}',  # 長數字串（帳號）
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URL
            r'\$\d+|\d+元',  # 金額
            r'line|LINE|賴',  # 社交軟體
        ]
    
    def connect_db(self):
        """連接資料庫"""
        return psycopg2.connect(**self.db_config)
    
    def create_tables(self):
        """創建必要的資料表"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # 訊息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                is_scam BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 預測記錄表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                message_id INTEGER REFERENCES messages(id),
                scam_probability FLOAT,
                predicted_label BOOLEAN,
                features JSONB,
                predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 創建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_is_scam ON messages(is_scam);
            CREATE INDEX IF NOT EXISTS idx_predictions_message_id ON predictions(message_id);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✓ 資料表創建完成")
    
    def preprocess_text(self, text):
        """
        文本預處理
        
        Args:
            text: 原始文本
            
        Returns:
            處理後的文本
        """
        # 轉小寫
        text = text.lower()
        
        # 移除特殊字符但保留中文
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 使用 jieba 分詞
        words = jieba.cut(text)
        
        return ' '.join(words)
    
    def extract_features(self, text):
        """
        提取特徵
        
        Args:
            text: 文本內容
            
        Returns:
            特徵字典
        """
        features = {}
        
        # 關鍵字特徵
        keyword_count = sum(1 for keyword in self.scam_keywords if keyword in text)
        features['keyword_count'] = keyword_count
        features['has_keywords'] = keyword_count > 0
        
        # 模式特徵
        pattern_count = 0
        for pattern in self.suspicious_patterns:
            pattern_count += len(re.findall(pattern, text))
        features['pattern_count'] = pattern_count
        
        # 文本特徵
        features['text_length'] = len(text)
        features['has_url'] = bool(re.search(r'http[s]?://', text))
        features['has_phone'] = bool(re.search(r'\d{10}', text))
        features['has_money'] = bool(re.search(r'\$\d+|\d+元', text))
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        return features
    
    def load_training_data(self):
        """從資料庫載入訓練資料"""
        conn = self.connect_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT content, is_scam 
            FROM messages 
            WHERE is_scam IS NOT NULL
        """)
        
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not data:
            raise ValueError("資料庫中沒有標記的訓練資料")
        
        texts = [row['content'] for row in data]
        labels = [row['is_scam'] for row in data]
        
        return texts, labels
    
    def train(self):
        """訓練模型"""
        print("正在載入訓練資料...")
        texts, labels = self.load_training_data()
        
        print(f"訓練資料數量: {len(texts)} 筆")
        print(f"詐騙訊息: {sum(labels)} 筆")
        print(f"正常訊息: {len(labels) - sum(labels)} 筆")
        
        # 預處理文本
        print("正在處理文本...")
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 提取 TF-IDF 特徵
        print("正在提取特徵...")
        X_tfidf = self.vectorizer.fit_transform(processed_texts)
        
        # 提取額外特徵
        extra_features = []
        for text in texts:
            features = self.extract_features(text)
            extra_features.append([
                features['keyword_count'],
                features['pattern_count'],
                features['text_length'],
                int(features['has_url']),
                int(features['has_phone']),
                int(features['has_money']),
                features['exclamation_count'],
                features['question_count']
            ])
        
        # 合併特徵
        X_combined = np.hstack([X_tfidf.toarray(), np.array(extra_features)])
        
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, labels, test_size=0.2, random_state=42
        )
        
        # 訓練模型
        print("正在訓練模型...")
        self.classifier.fit(X_train, y_train)
        
        # 評估模型
        y_pred = self.classifier.predict(X_test)
        print("\n模型評估結果:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['正常訊息', '詐騙訊息']))
        print("\n混淆矩陣:")
        print(confusion_matrix(y_test, y_pred))
        
        self.is_trained = True
        print("\n✓ 模型訓練完成")
    
    def predict(self, text, save_to_db=True):
        """
        預測訊息是否為詐騙
        
        Args:
            text: 訊息內容
            save_to_db: 是否儲存到資料庫
            
        Returns:
            dict: 包含預測結果和機率的字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練，請先執行 train() 方法")
        
        # 預處理
        processed_text = self.preprocess_text(text)
        
        # TF-IDF 特徵
        X_tfidf = self.vectorizer.transform([processed_text])
        
        # 額外特徵
        features = self.extract_features(text)
        extra_features = np.array([[
            features['keyword_count'],
            features['pattern_count'],
            features['text_length'],
            int(features['has_url']),
            int(features['has_phone']),
            int(features['has_money']),
            features['exclamation_count'],
            features['question_count']
        ]])
        
        # 合併特徵
        X_combined = np.hstack([X_tfidf.toarray(), extra_features])
        
        # 預測
        prediction = self.classifier.predict(X_combined)[0]
        probability = self.classifier.predict_proba(X_combined)[0]
        
        result = {
            'is_scam': bool(prediction),
            'scam_probability': float(probability[1]),
            'confidence': max(probability),
            'features': features,
            'warning_level': self._get_warning_level(probability[1])
        }
        
        # 儲存到資料庫
        if save_to_db:
            self._save_prediction(text, result)
        
        return result
    
    def _get_warning_level(self, scam_prob):
        """根據詐騙機率返回警告等級"""
        if scam_prob >= 0.8:
            return "高危險"
        elif scam_prob >= 0.5:
            return "中度風險"
        elif scam_prob >= 0.3:
            return "低度風險"
        else:
            return "安全"
    
    def _save_prediction(self, text, result):
        """將預測結果儲存到資料庫"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # 先插入訊息
        cursor.execute("""
            INSERT INTO messages (content) 
            VALUES (%s) 
            RETURNING id
        """, (text,))
        
        message_id = cursor.fetchone()[0]
        
        # 插入預測結果
        cursor.execute("""
            INSERT INTO predictions 
            (message_id, scam_probability, predicted_label, features)
            VALUES (%s, %s, %s, %s)
        """, (
            message_id,
            result['scam_probability'],
            result['is_scam'],
            json.dumps(result['features'])
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def save_model(self, filename='scam_detector_model.pkl'):
        """
        儲存模型
        
        Args:
            filename: 模型儲存路徑（預設為 scam_detector_model.pkl）
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'scam_keywords': self.scam_keywords,
            'suspicious_patterns': self.suspicious_patterns
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ 模型已儲存至 {filename}")
    
    def load_model(self, filename='scam_detector_model.pkl'):
        """載入模型"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.scam_keywords = model_data['scam_keywords']
        self.suspicious_patterns = model_data['suspicious_patterns']
        self.is_trained = True
        
        print(f"✓ 模型已從 {filename} 載入")


if __name__ == "__main__":
    # 初始化偵測器
    detector = ScamDetector(db_config)
    
    # 創建資料表
    detector.create_tables()
    
    print("\n詐騙訊息偵測系統已啟動")