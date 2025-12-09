#!/usr/bin/env python3
"""
Flask Web API for Scam Detection
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from scam_detector import ScamDetector
from config import db_config
import os

app = Flask(__name__)
CORS(app)  # 允許跨域請求

# 初始化偵測器
detector = ScamDetector(db_config)

# 載入訓練好的模型
try:
    detector.load_model('scam_detector_model.pkl')
    print("✓ 模型載入成功")
except FileNotFoundError:
    print("⚠️  警告: 找不到模型檔案，請先訓練模型")
    print("   執行: python train.py")


@app.route('/')
def index():
    """提供前端頁面"""
    return send_file('scam_detector_web.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    詐騙偵測 API
    
    Request Body:
    {
        "message": "要檢測的訊息"
    }
    
    Response:
    {
        "is_scam": true/false,
        "scam_probability": 0.0-1.0,
        "confidence": 0.0-1.0,
        "warning_level": "安全/低度風險/中度風險/高危險",
        "features": {
            "keyword_count": int,
            "has_url": bool,
            "has_phone": bool,
            "pattern_count": int
        }
    }
    """
    try:
        # 取得請求資料
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': '請提供 message 參數'
            }), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({
                'error': '訊息不能為空'
            }), 400
        
        # 執行偵測
        result = detector.predict(message, save_to_db=True)
        
        # 返回結果
        return jsonify({
            'success': True,
            'is_scam': result['is_scam'],
            'scam_probability': result['scam_probability'],
            'confidence': result['confidence'],
            'warning_level': result['warning_level'],
            'features': {
                'keyword_count': result['features']['keyword_count'],
                'has_url': result['features']['has_url'],
                'has_phone': result['features']['has_phone'],
                'pattern_count': result['features']['pattern_count']
            }
        })
    
    except Exception as e:
        print(f"偵測錯誤: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-detect', methods=['POST'])
def batch_detect():
    """
    批次偵測 API
    
    Request Body:
    {
        "messages": ["訊息1", "訊息2", "訊息3"]
    }
    
    Response:
    {
        "results": [
            {
                "message": "訊息1",
                "is_scam": true/false,
                "scam_probability": 0.0-1.0,
                ...
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({
                'error': '請提供 messages 參數（陣列）'
            }), 400
        
        messages = data['messages']
        
        if not isinstance(messages, list):
            return jsonify({
                'error': 'messages 必須是陣列'
            }), 400
        
        # 批次偵測
        results = []
        for msg in messages:
            if msg.strip():
                result = detector.predict(msg, save_to_db=True)
                results.append({
                    'message': msg,
                    'is_scam': result['is_scam'],
                    'scam_probability': result['scam_probability'],
                    'warning_level': result['warning_level']
                })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        print(f"批次偵測錯誤: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """
    取得系統統計資訊
    
    Response:
    {
        "total_predictions": int,
        "scam_count": int,
        "normal_count": int,
        "avg_scam_probability": float
    }
    """
    try:
        import psycopg
        
        conn_string = f"host={db_config['host']} dbname={db_config['database']} user={db_config['user']} password={db_config['password']} port={db_config['port']}"
        conn = psycopg.connect(conn_string)
        cursor = conn.cursor()
        
        # 查詢統計
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN predicted_label = true THEN 1 ELSE 0 END) as scam_count,
                AVG(scam_probability) as avg_prob
            FROM predictions
        """)
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'total_predictions': result[0],
            'scam_count': result[1] or 0,
            'normal_count': (result[0] - (result[1] or 0)),
            'avg_scam_probability': float(result[2]) if result[2] else 0
        })
    
    except Exception as e:
        print(f"統計錯誤: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康檢查端點"""
    return jsonify({
        'status': 'ok',
        'model_loaded': detector.is_trained
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': '找不到該端點'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': '伺服器內部錯誤'
    }), 500


if __name__ == '__main__':
    # 檢查模型是否存在
    if not os.path.exists('scam_detector_model.pkl'):
        print("\n" + "="*60)
        print("⚠️  警告: 找不到模型檔案")
        print("="*60)
        print("\n請先訓練模型:")
        print("  python train.py")
        print("\n或者執行:")
        print("  python example_usage.py")
        print("  選擇選項 2 訓練模型")
        print("\n" + "="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("🚀 詐騙偵測 API 伺服器啟動")
        print("="*60)
        print("\n可用端點:")
        print("  GET  /                - 網頁介面")
        print("  POST /api/detect      - 偵測單一訊息")
        print("  POST /api/batch-detect - 批次偵測")
        print("  GET  /api/stats       - 系統統計")
        print("  GET  /api/health      - 健康檢查")
        print("\n伺服器位址: http://localhost:8000")
        print("="*60 + "\n")
    
    # 啟動伺服器
    # debug=True 用於開發，生產環境請設為 False
    app.run(host='0.0.0.0', port=8000, debug=True)