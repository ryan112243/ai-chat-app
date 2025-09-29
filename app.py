"""
AI聊天助手Web應用
專注於多域AI對話功能
"""

from flask import Flask, render_template, request, jsonify
import torch
import os
import json
import glob
import asyncio
from datetime import datetime
from ai_service import ai_service, AIProvider

app = Flask(__name__)
app.secret_key = 'ai_chat_secret_key'

# 模型路徑配置
MODELS_DIR = '../models/multi_domain'

@app.route('/')
def index():
    """主頁 - AI聊天界面"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        domain = data.get('domain', 'dialogue')
        
        if not user_message.strip():
            return jsonify({'success': False, 'error': '消息不能為空'})
        
        # 記錄用戶消息
        app.logger.info(f"用戶消息 [{domain}]: {user_message}")
        
        # 使用異步方式生成AI回應
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            ai_response = loop.run_until_complete(generate_ai_response(user_message, domain))
        finally:
            loop.close()
        
        # 記錄AI回應
        app.logger.info(f"AI回應: {ai_response[:100]}...")
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"聊天處理錯誤: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'處理請求時發生錯誤: {str(e)}'
        })

async def generate_ai_response(message, domain):
    """使用先進AI模型生成回應"""
    try:
        # 使用AI服務管理器生成回應
        response = await ai_service.generate_response(message, domain)
        
        # 記錄回應品質資訊
        print(f"AI回應生成 - 提供商: {response.provider}, 模型: {response.model}, "
              f"信心度: {response.confidence}, 回應時間: {response.response_time:.2f}秒")
        
        return response.content
        
    except Exception as e:
        print(f"AI回應生成錯誤: {str(e)}")
        return generate_enhanced_fallback_response(message, domain)

def get_latest_model_path():
    """獲取最新的模型路徑"""
    try:
        model_files = glob.glob(os.path.join(MODELS_DIR, 'multi_domain_model_*.pth'))
        if not model_files:
            return None
        return max(model_files, key=os.path.getctime)
    except:
        return None

def generate_enhanced_fallback_response(message, domain):
    """生成增強的備用回應"""
    enhanced_responses = {
        'math': f"""數學專家助手：關於您的問題「{message}」

我來為您提供數學方面的幫助：
• 如果是計算問題，我建議您檢查數字和運算符號
• 如果是概念問題，可以嘗試從基礎定義開始理解
• 對於複雜問題，建議分步驟解決
• 您也可以使用數學工具如計算器、圖形軟體等輔助

需要更具體的幫助嗎？請提供更多詳細資訊。""",

        'programming': f"""程式設計專家：關於您的編程問題「{message}」

讓我為您提供編程建議：
• 檢查語法和邏輯錯誤
• 確認變數名稱和資料類型
• 查看相關文檔和範例代碼
• 使用除錯工具逐步檢查
• 考慮代碼的可讀性和效能

如需更詳細的協助，請分享您的代碼片段。""",

        'writing': f"""寫作指導專家：關於您的寫作需求「{message}」

我來協助您提升寫作品質：
• 明確寫作目的和目標讀者
• 建立清晰的文章結構和大綱
• 使用生動具體的例子和描述
• 注意語法、標點和用詞準確性
• 多次修改和潤色您的作品

需要針對特定寫作類型的建議嗎？""",

        'dialogue': f"""智慧對話夥伴：關於「{message}」這個話題

這確實是個值得深入探討的問題：
• 讓我們從不同角度來分析
• 考慮相關的背景和脈絡
• 探索可能的解決方案或觀點
• 分享相關的經驗和見解
• 提出進一步思考的問題

您希望從哪個方面開始討論呢？""",

        'mun': f"""模擬聯合國專家：關於國際議題「{message}」

作為您的外交顧問，我建議：
• 研究相關的國際法和條約
• 分析各主要國家的立場和利益
• 考慮歷史先例和案例
• 評估可能的談判策略
• 準備多種解決方案選項

需要針對特定國家立場或程序的建議嗎？"""
    }
    
    return enhanced_responses.get(domain, f"""AI智慧助手：感謝您的問題「{message}」

雖然目前遇到一些技術限制，但我仍想為您提供幫助：
• 這是一個很有意思的問題
• 建議您可以從多個角度思考
• 歡迎提供更多背景資訊
• 我會持續學習以提供更好的服務

有什麼其他我可以協助的嗎？""")

if __name__ == '__main__':
    print("AI聊天助手啟動中...")
    print("訪問地址: http://localhost:5001")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )