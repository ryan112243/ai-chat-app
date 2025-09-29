"""
AI聊天助手Web應用
專注於多域AI對話功能
"""

from flask import Flask, render_template, request, jsonify
import torch
import os
import json
import glob
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'ai_chat_secret_key'

# 模型路徑配置
MODELS_DIR = '../models/multi_domain'

@app.route('/')
def index():
    """主頁 - AI聊天界面"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """處理聊天請求"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        domain = data.get('domain', 'dialogue')
        
        if not message.strip():
            return jsonify({
                'success': False,
                'error': '請輸入有效的消息'
            })
        
        # 生成AI回應
        response = generate_ai_response(message, domain)
        
        return jsonify({
            'success': True,
            'response': response,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'處理請求時發生錯誤: {str(e)}'
        })

def generate_ai_response(message, domain):
    """根據域生成AI回應"""
    try:
        # 載入最新的多域模型
        model_path = get_latest_model_path()
        if not model_path:
            return generate_default_response(message, domain)
        
        # 根據域分派到相應的處理函數
        if domain == 'math':
            return generate_math_response(message)
        elif domain == 'programming':
            return generate_programming_response(message)
        elif domain == 'dialogue':
            return generate_dialogue_response(message)
        elif domain == 'writing':
            return generate_writing_response(message)
        elif domain == 'mun':
            return generate_mun_response(message)
        else:
            return generate_default_response(message, domain)
            
    except Exception as e:
        return f"抱歉，處理您的請求時遇到了問題：{str(e)}"

def get_latest_model_path():
    """獲取最新的模型路徑"""
    try:
        model_files = glob.glob(os.path.join(MODELS_DIR, 'multi_domain_model_*.pth'))
        if not model_files:
            return None
        return max(model_files, key=os.path.getctime)
    except:
        return None

def generate_math_response(message):
    """生成數學相關回應"""
    return f"數學助手：我來幫您解決數學問題。關於「{message}」，讓我為您分析和計算..."

def generate_programming_response(message):
    """生成程式設計相關回應"""
    return f"程式設計助手：我來幫您解決編程問題。關於「{message}」，讓我為您提供代碼解決方案..."

def generate_dialogue_response(message):
    """生成對話回應"""
    return f"智能助手：我理解您的問題「{message}」。讓我為您提供詳細的回答和建議..."

def generate_writing_response(message):
    """生成寫作相關回應"""
    return f"寫作助手：我來幫您進行創意寫作。關於「{message}」，讓我為您創作和潤色..."

def generate_mun_response(message):
    """生成模擬聯合國相關回應"""
    return f"MUN助手：作為外交顧問，關於「{message}」，我將為您提供專業的外交分析和建議..."

def generate_default_response(message, domain):
    """生成默認回應"""
    return f"AI助手（{domain}）：感謝您的提問「{message}」。我正在為您準備回應..."

if __name__ == '__main__':
    print("AI聊天助手啟動中...")
    print("訪問地址: http://localhost:5001")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )