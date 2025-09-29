# AI聊天助手

一個專注於多域AI對話的Web應用程序，支持數學、程式設計、寫作、智能對話和模擬聯合國等多個領域。

## 功能特色

- 🤖 **多域AI助手**：支持5個不同的AI領域
- 💬 **實時對話**：流暢的聊天體驗
- 🎨 **現代化界面**：響應式設計，支持桌面和移動設備
- 🔄 **智能切換**：輕鬆在不同AI領域間切換

## 支持的AI領域

1. **智能對話** - 日常對話和問答
2. **數學計算** - 數學問題解答和計算
3. **程式設計** - 代碼編寫和調試幫助
4. **創意寫作** - 文章、詩歌和創意內容
5. **模擬聯合國** - 外交文件和國際關係分析

## 本地運行

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動應用
python app.py
```

訪問 http://localhost:5001

## 部署到Render

1. 將代碼推送到GitHub
2. 在Render中創建新的Web Service
3. 連接GitHub倉庫
4. Render會自動使用render.yaml配置進行部署

## 技術棧

- **後端**: Flask, Python
- **前端**: HTML5, CSS3, JavaScript, Bootstrap 5
- **AI模型**: PyTorch
- **部署**: Render, Gunicorn