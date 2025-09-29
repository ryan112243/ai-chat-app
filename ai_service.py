"""
AI服務模組 - 整合多個先進AI模型
支援OpenAI GPT、Anthropic Claude、Google Gemini等
"""

import os
import json
import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"

@dataclass
class AIResponse:
    content: str
    provider: str
    model: str
    tokens_used: int
    response_time: float
    confidence: float

class AIServiceManager:
    """AI服務管理器 - 統一管理多個AI提供商"""
    
    def __init__(self):
        self.providers = {}
        self.fallback_order = [
            AIProvider.OPENAI,
            AIProvider.ANTHROPIC,
            AIProvider.GOOGLE,
            AIProvider.HUGGINGFACE
        ]
        self._initialize_providers()
    
    def _initialize_providers(self):
        """初始化AI提供商"""
        # OpenAI GPT
        if os.getenv('OPENAI_API_KEY'):
            self.providers[AIProvider.OPENAI] = OpenAIService()
        
        # Anthropic Claude
        if os.getenv('ANTHROPIC_API_KEY'):
            self.providers[AIProvider.ANTHROPIC] = AnthropicService()
        
        # Google Gemini
        if os.getenv('GOOGLE_API_KEY'):
            self.providers[AIProvider.GOOGLE] = GoogleService()
        
        # Hugging Face (免費備用)
        self.providers[AIProvider.HUGGINGFACE] = HuggingFaceService()
    
    async def generate_response(self, message: str, domain: str = "general", 
                              preferred_provider: Optional[AIProvider] = None) -> AIResponse:
        """生成AI回應，支援多提供商容錯"""
        
        # 構建提示詞
        prompt = self._build_domain_prompt(message, domain)
        
        # 確定使用順序
        providers_to_try = []
        if preferred_provider and preferred_provider in self.providers:
            providers_to_try.append(preferred_provider)
        
        for provider in self.fallback_order:
            if provider not in providers_to_try and provider in self.providers:
                providers_to_try.append(provider)
        
        # 嘗試各個提供商
        last_error = None
        for provider in providers_to_try:
            try:
                service = self.providers[provider]
                response = await service.generate(prompt, domain)
                if response and response.content:
                    return response
            except Exception as e:
                last_error = e
                continue
        
        # 所有提供商都失敗，返回智能默認回應
        return self._generate_intelligent_fallback(message, domain, last_error)
    
    def _build_domain_prompt(self, message: str, domain: str) -> str:
        """根據領域構建專業提示詞"""
        domain_prompts = {
            'math': f"""你是一位專業的數學導師。請針對以下數學問題提供詳細、準確的解答：
問題：{message}

請提供：
1. 清晰的解題步驟
2. 相關的數學概念解釋
3. 如果可能，提供多種解法
4. 實際應用例子""",
            
            'programming': f"""你是一位資深的程式設計專家。請針對以下編程問題提供專業建議：
問題：{message}

請提供：
1. 清晰的代碼解決方案
2. 代碼解釋和最佳實踐
3. 可能的優化建議
4. 相關的技術背景""",
            
            'writing': f"""你是一位專業的寫作指導老師。請針對以下寫作需求提供幫助：
需求：{message}

請提供：
1. 創意和結構建議
2. 具體的寫作技巧
3. 範例或模板
4. 改進建議""",
            
            'dialogue': f"""你是一位智慧的對話夥伴。請針對以下話題進行深入、有趣的對話：
話題：{message}

請提供：
1. 深思熟慮的回應
2. 相關的背景知識
3. 引發思考的問題
4. 實用的建議""",
            
            'mun': f"""你是一位經驗豐富的外交顧問和模擬聯合國專家。請針對以下議題提供專業分析：
議題：{message}

請提供：
1. 國際法和外交角度的分析
2. 各國立場和利益考量
3. 可能的解決方案
4. 談判策略建議"""
        }
        
        return domain_prompts.get(domain, f"""請針對以下問題提供專業、詳細的回答：
問題：{message}

請確保回答準確、有用且易於理解。""")
    
    def _generate_intelligent_fallback(self, message: str, domain: str, error: Exception) -> AIResponse:
        """生成智能備用回應"""
        fallback_responses = {
            'math': f"我理解您的數學問題「{message}」。雖然目前無法提供完整的計算，但我建議您可以嘗試分解問題、查找相關公式，或使用數學工具來輔助解決。",
            'programming': f"關於您的編程問題「{message}」，我建議您可以查閱官方文檔、搜索相關的代碼範例，或在開發者社群中尋求幫助。",
            'writing': f"對於您的寫作需求「{message}」，我建議您可以先列出大綱、收集相關資料，並參考優秀的寫作範例來提升您的作品。",
            'dialogue': f"關於「{message}」這個話題，這確實是一個值得深入討論的問題。我建議您可以從多個角度來思考，並歡迎進一步交流。",
            'mun': f"關於「{message}」這個國際議題，建議您研究相關的國際法條文、各國官方立場，以及歷史先例來形成全面的分析。"
        }
        
        content = fallback_responses.get(domain, f"感謝您的問題「{message}」。雖然目前遇到一些技術問題，但我會持續改進來為您提供更好的服務。")
        
        return AIResponse(
            content=content,
            provider="fallback",
            model="intelligent_fallback",
            tokens_used=0,
            response_time=0.1,
            confidence=0.6
        )

class OpenAIService:
    """OpenAI GPT服務"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"
    
    async def generate(self, prompt: str, domain: str) -> AIResponse:
        """使用OpenAI GPT生成回應"""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一位專業、友善且知識豐富的AI助手。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        tokens_used = result['usage']['total_tokens']
        
        return AIResponse(
            content=content,
            provider="openai",
            model=self.model,
            tokens_used=tokens_used,
            response_time=time.time() - start_time,
            confidence=0.9
        )

class AnthropicService:
    """Anthropic Claude服務"""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-sonnet-20240229"
    
    async def generate(self, prompt: str, domain: str) -> AIResponse:
        """使用Anthropic Claude生成回應"""
        start_time = time.time()
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result['content'][0]['text']
        tokens_used = result['usage']['input_tokens'] + result['usage']['output_tokens']
        
        return AIResponse(
            content=content,
            provider="anthropic",
            model=self.model,
            tokens_used=tokens_used,
            response_time=time.time() - start_time,
            confidence=0.95
        )

class GoogleService:
    """Google Gemini服務"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
    
    async def generate(self, prompt: str, domain: str) -> AIResponse:
        """使用Google Gemini生成回應"""
        start_time = time.time()
        
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result['candidates'][0]['content']['parts'][0]['text']
        
        return AIResponse(
            content=content,
            provider="google",
            model="gemini-pro",
            tokens_used=len(content) // 4,  # 估算
            response_time=time.time() - start_time,
            confidence=0.85
        )

class HuggingFaceService:
    """Hugging Face免費服務（備用）"""
    
    def __init__(self):
        self.base_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
        self.api_key = os.getenv('HUGGINGFACE_API_KEY', '')
    
    async def generate(self, prompt: str, domain: str) -> AIResponse:
        """使用Hugging Face模型生成回應"""
        start_time = time.time()
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {"inputs": prompt}
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result[0]['generated_text'] if isinstance(result, list) else str(result)
        
        return AIResponse(
            content=content,
            provider="huggingface",
            model="DialoGPT-large",
            tokens_used=len(content) // 4,
            response_time=time.time() - start_time,
            confidence=0.7
        )

# 全域AI服務管理器實例
ai_service = AIServiceManager()