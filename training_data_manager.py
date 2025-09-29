"""
訓練數據管理器 - 收集和整理大數據訓練集
支援多域對話數據的收集、清理和增強
"""

import os
import json
import csv
import requests
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

@dataclass
class TrainingExample:
    """訓練範例數據結構"""
    domain: str
    user_input: str
    ai_response: str
    quality_score: float
    source: str
    timestamp: str
    metadata: Dict[str, Any] = None

class TrainingDataManager:
    """訓練數據管理器"""
    
    def __init__(self, data_dir: str = "./training_data"):
        self.data_dir = data_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """確保必要的目錄存在"""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, "raw"),
            os.path.join(self.data_dir, "processed"),
            os.path.join(self.data_dir, "domains"),
            os.path.join(self.data_dir, "quality_filtered")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def collect_conversation_data(self, conversations: List[Dict]) -> List[TrainingExample]:
        """從對話記錄中收集訓練數據"""
        training_examples = []
        
        for conv in conversations:
            if self._is_valid_conversation(conv):
                example = TrainingExample(
                    domain=conv.get('domain', 'general'),
                    user_input=conv['user_message'],
                    ai_response=conv['ai_response'],
                    quality_score=self._calculate_quality_score(conv),
                    source="user_conversations",
                    timestamp=conv.get('timestamp', datetime.now().isoformat()),
                    metadata={'conversation_id': conv.get('id')}
                )
                training_examples.append(example)
        
        return training_examples
    
    def generate_synthetic_data(self, domain: str, count: int = 100) -> List[TrainingExample]:
        """生成合成訓練數據"""
        synthetic_examples = []
        
        domain_templates = self._get_domain_templates(domain)
        
        for i in range(count):
            template = random.choice(domain_templates)
            user_input = self._generate_user_input(template, domain)
            ai_response = self._generate_ai_response(template, domain)
            
            example = TrainingExample(
                domain=domain,
                user_input=user_input,
                ai_response=ai_response,
                quality_score=0.8,  # 合成數據的基礎品質分數
                source="synthetic_generation",
                timestamp=datetime.now().isoformat(),
                metadata={'template_id': template['id'], 'generation_round': i}
            )
            synthetic_examples.append(example)
        
        return synthetic_examples
    
    def _get_domain_templates(self, domain: str) -> List[Dict]:
        """獲取領域特定的模板"""
        templates = {
            'math': [
                {
                    'id': 'calculation',
                    'user_patterns': [
                        "計算 {expression}",
                        "幫我算一下 {expression}",
                        "{expression} 等於多少？",
                        "求解 {expression}"
                    ],
                    'expressions': [
                        "2 + 3 × 4", "√16 + 5²", "sin(30°)", "log₂(8)",
                        "∫x²dx", "lim(x→0) sin(x)/x", "3x + 5 = 14"
                    ]
                },
                {
                    'id': 'concept',
                    'user_patterns': [
                        "什麼是{concept}？",
                        "解釋一下{concept}",
                        "{concept}的定義是什麼？",
                        "如何理解{concept}？"
                    ],
                    'concepts': [
                        "微積分", "線性代數", "機率論", "統計學",
                        "幾何學", "三角函數", "複數", "矩陣"
                    ]
                }
            ],
            'programming': [
                {
                    'id': 'coding_problem',
                    'user_patterns': [
                        "如何用{language}實現{task}？",
                        "寫一個{language}程式來{task}",
                        "{language}中如何{task}？",
                        "幫我寫{task}的代碼"
                    ],
                    'languages': ['Python', 'JavaScript', 'Java', 'C++', 'Go'],
                    'tasks': [
                        '排序陣列', '搜尋元素', '計算階乘', '反轉字串',
                        '檢查回文', '生成斐波那契數列', '實現二分搜尋'
                    ]
                },
                {
                    'id': 'debugging',
                    'user_patterns': [
                        "這段代碼有什麼問題？{code}",
                        "為什麼我的程式不工作？{code}",
                        "幫我除錯：{code}",
                        "修復這個錯誤：{code}"
                    ],
                    'code_snippets': [
                        "for i in range(10) print(i)",
                        "def factorial(n): return n * factorial(n-1)",
                        "list = [1,2,3]; list.append(4); print(list[4])"
                    ]
                }
            ],
            'writing': [
                {
                    'id': 'creative_writing',
                    'user_patterns': [
                        "幫我寫一篇關於{topic}的{type}",
                        "創作一個{genre}故事，主題是{topic}",
                        "寫一段{type}，描述{topic}",
                        "給我一個{topic}的{type}範例"
                    ],
                    'topics': ['友情', '冒險', '科技', '自然', '夢想', '成長'],
                    'types': ['短文', '詩歌', '故事', '散文', '日記'],
                    'genres': ['科幻', '奇幻', '懸疑', '愛情', '歷史']
                }
            ],
            'dialogue': [
                {
                    'id': 'general_chat',
                    'user_patterns': [
                        "你覺得{topic}怎麼樣？",
                        "我們來聊聊{topic}吧",
                        "關於{topic}，你有什麼看法？",
                        "談談你對{topic}的想法"
                    ],
                    'topics': [
                        '人工智慧的未來', '環保議題', '教育改革', '科技發展',
                        '文化差異', '生活哲學', '職業規劃', '健康生活'
                    ]
                }
            ],
            'mun': [
                {
                    'id': 'diplomatic_analysis',
                    'user_patterns': [
                        "分析{country}在{issue}上的立場",
                        "{issue}的國際法依據是什麼？",
                        "如何解決{issue}問題？",
                        "{country}應該如何應對{issue}？"
                    ],
                    'countries': ['美國', '中國', '俄羅斯', '德國', '日本', '印度'],
                    'issues': [
                        '氣候變遷', '核武擴散', '貿易爭端', '難民危機',
                        '網路安全', '太空軍備', '海洋權益', '人權問題'
                    ]
                }
            ]
        }
        
        return templates.get(domain, [])
    
    def _generate_user_input(self, template: Dict, domain: str) -> str:
        """生成用戶輸入"""
        pattern = random.choice(template['user_patterns'])
        
        # 根據模板類型填充變數
        if 'expressions' in template:
            expression = random.choice(template['expressions'])
            return pattern.format(expression=expression)
        elif 'concepts' in template:
            concept = random.choice(template['concepts'])
            return pattern.format(concept=concept)
        elif 'languages' in template and 'tasks' in template:
            language = random.choice(template['languages'])
            task = random.choice(template['tasks'])
            return pattern.format(language=language, task=task)
        elif 'topics' in template:
            topic = random.choice(template['topics'])
            if 'types' in template:
                type_val = random.choice(template['types'])
                if 'genres' in template:
                    genre = random.choice(template['genres'])
                    return pattern.format(topic=topic, type=type_val, genre=genre)
                return pattern.format(topic=topic, type=type_val)
            return pattern.format(topic=topic)
        elif 'countries' in template and 'issues' in template:
            country = random.choice(template['countries'])
            issue = random.choice(template['issues'])
            return pattern.format(country=country, issue=issue)
        
        return pattern
    
    def _generate_ai_response(self, template: Dict, domain: str) -> str:
        """生成AI回應"""
        response_templates = {
            'math': [
                "讓我來幫您解決這個數學問題。首先，我們需要...",
                "這是一個很好的數學問題。根據相關定理...",
                "我來為您詳細計算這個問題的步驟..."
            ],
            'programming': [
                "這是一個常見的編程問題。讓我為您提供解決方案...",
                "我來幫您分析這段代碼並提供改進建議...",
                "這個問題可以用以下方法解決..."
            ],
            'writing': [
                "我很樂意幫您創作。讓我們從以下角度開始...",
                "這是一個很有創意的寫作主題。我建議...",
                "讓我為您提供一些寫作靈感和結構建議..."
            ],
            'dialogue': [
                "這是一個很有趣的話題。從我的角度來看...",
                "關於這個問題，我認為我們可以從多個層面來討論...",
                "這確實是值得深入思考的議題..."
            ],
            'mun': [
                "從國際法的角度來分析這個問題...",
                "根據聯合國憲章和相關條約...",
                "這個議題涉及多方利益，需要平衡考慮..."
            ]
        }
        
        templates_list = response_templates.get(domain, ["讓我來幫助您解決這個問題..."])
        return random.choice(templates_list)
    
    def _is_valid_conversation(self, conv: Dict) -> bool:
        """檢查對話是否有效"""
        required_fields = ['user_message', 'ai_response']
        return all(field in conv and conv[field].strip() for field in required_fields)
    
    def _calculate_quality_score(self, conv: Dict) -> float:
        """計算對話品質分數"""
        score = 0.5  # 基礎分數
        
        # 回應長度評分
        response_length = len(conv['ai_response'])
        if response_length > 100:
            score += 0.2
        elif response_length > 50:
            score += 0.1
        
        # 用戶輸入品質評分
        user_input = conv['user_message']
        if len(user_input) > 10 and '?' in user_input:
            score += 0.1
        
        # 領域相關性評分
        domain = conv.get('domain', 'general')
        if domain != 'general':
            score += 0.1
        
        # 確保分數在0-1範圍內
        return min(max(score, 0.0), 1.0)
    
    def save_training_data(self, examples: List[TrainingExample], filename: str):
        """保存訓練數據"""
        filepath = os.path.join(self.data_dir, "processed", filename)
        
        data_to_save = []
        for example in examples:
            data_to_save.append({
                'domain': example.domain,
                'user_input': example.user_input,
                'ai_response': example.ai_response,
                'quality_score': example.quality_score,
                'source': example.source,
                'timestamp': example.timestamp,
                'metadata': example.metadata or {}
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(examples)} 個訓練範例到 {filepath}")
    
    def load_training_data(self, filename: str) -> List[TrainingExample]:
        """載入訓練數據"""
        filepath = os.path.join(self.data_dir, "processed", filename)
        
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            example = TrainingExample(
                domain=item['domain'],
                user_input=item['user_input'],
                ai_response=item['ai_response'],
                quality_score=item['quality_score'],
                source=item['source'],
                timestamp=item['timestamp'],
                metadata=item.get('metadata', {})
            )
            examples.append(example)
        
        return examples
    
    def filter_by_quality(self, examples: List[TrainingExample], 
                         min_score: float = 0.7) -> List[TrainingExample]:
        """根據品質分數過濾訓練數據"""
        return [ex for ex in examples if ex.quality_score >= min_score]
    
    def augment_data(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """數據增強 - 生成變體和同義詞替換"""
        augmented = []
        
        for example in examples:
            # 原始數據
            augmented.append(example)
            
            # 生成變體（簡單的同義詞替換）
            if example.quality_score > 0.8:
                variant = self._create_variant(example)
                if variant:
                    augmented.append(variant)
        
        return augmented
    
    def _create_variant(self, example: TrainingExample) -> Optional[TrainingExample]:
        """創建數據變體"""
        # 簡單的同義詞替換
        synonyms = {
            '幫我': ['協助我', '請幫助我', '能否幫我'],
            '如何': ['怎麼', '怎樣', '如何才能'],
            '什麼': ['甚麼', '何謂', '什麼是'],
            '計算': ['算出', '求出', '運算'],
            '解釋': ['說明', '闡述', '講解']
        }
        
        user_input = example.user_input
        for original, replacements in synonyms.items():
            if original in user_input:
                replacement = random.choice(replacements)
                new_input = user_input.replace(original, replacement, 1)
                
                return TrainingExample(
                    domain=example.domain,
                    user_input=new_input,
                    ai_response=example.ai_response,
                    quality_score=example.quality_score * 0.9,  # 略微降低變體分數
                    source=f"{example.source}_variant",
                    timestamp=datetime.now().isoformat(),
                    metadata={**example.metadata, 'is_variant': True}
                )
        
        return None
    
    def generate_comprehensive_dataset(self, domains: List[str], 
                                     examples_per_domain: int = 200) -> List[TrainingExample]:
        """生成全面的訓練數據集"""
        all_examples = []
        
        for domain in domains:
            print(f"正在為 {domain} 領域生成訓練數據...")
            
            # 生成合成數據
            synthetic_data = self.generate_synthetic_data(domain, examples_per_domain)
            
            # 數據增強
            augmented_data = self.augment_data(synthetic_data)
            
            # 品質過濾
            filtered_data = self.filter_by_quality(augmented_data, min_score=0.6)
            
            all_examples.extend(filtered_data)
            print(f"{domain} 領域生成了 {len(filtered_data)} 個高品質訓練範例")
        
        return all_examples

# 全域訓練數據管理器實例
training_manager = TrainingDataManager()