#!/usr/bin/env python3
"""
AI模型優化器
使用生成的訓練資料進行模型微調和優化
"""

import json
import os
import asyncio
from typing import List, Dict, Any
from training_data_manager import TrainingExample, training_manager
from ai_service import ai_service

class ModelOptimizer:
    def __init__(self):
        self.training_data_dir = "./training_data/processed"
        self.optimization_results = []
        
    def load_all_training_data(self) -> List[TrainingExample]:
        """載入所有訓練資料"""
        all_data = []
        
        if not os.path.exists(self.training_data_dir):
            print("❌ 訓練資料目錄不存在")
            return all_data
            
        for filename in os.listdir(self.training_data_dir):
            if filename.endswith('.json'):
                try:
                    data = training_manager.load_training_data(filename)
                    all_data.extend(data)
                    print(f"✅ 載入 {filename}: {len(data)} 個範例")
                except Exception as e:
                    print(f"❌ 載入 {filename} 失敗: {str(e)}")
        
        print(f"\n📊 總共載入 {len(all_data)} 個訓練範例")
        return all_data
    
    async def evaluate_model_performance(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """評估模型性能"""
        print("\n🔍 評估模型性能...")
        
        correct_predictions = 0
        total_predictions = len(test_examples)
        domain_performance = {}
        
        for i, example in enumerate(test_examples[:50]):  # 測試前50個範例
            try:
                # 使用AI服務生成回應
                response = await ai_service.generate_response(
                    message=example.user_input,
                    domain=example.domain
                )
                
                # 簡單的性能評估（基於回應長度和關鍵詞匹配）
                is_good_response = self._evaluate_response_quality(
                    example.ai_response, 
                    response.content if hasattr(response, 'content') else str(response)
                )
                
                if is_good_response:
                    correct_predictions += 1
                
                # 按領域統計
                if example.domain not in domain_performance:
                    domain_performance[example.domain] = {'correct': 0, 'total': 0}
                
                domain_performance[example.domain]['total'] += 1
                if is_good_response:
                    domain_performance[example.domain]['correct'] += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  進度: {i + 1}/50")
                    
            except Exception as e:
                print(f"  評估範例 {i+1} 時發生錯誤: {str(e)}")
        
        # 計算整體準確率
        overall_accuracy = correct_predictions / min(50, total_predictions) if total_predictions > 0 else 0
        
        # 計算各領域準確率
        domain_accuracies = {}
        for domain, stats in domain_performance.items():
            domain_accuracies[domain] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        results = {
            'overall_accuracy': overall_accuracy,
            'domain_accuracies': domain_accuracies,
            'total_evaluated': min(50, total_predictions)
        }
        
        return results
    
    def _evaluate_response_quality(self, expected: str, actual: str) -> bool:
        """評估回應品質"""
        if not actual or len(actual.strip()) < 10:
            return False
        
        # 基本品質檢查
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        # 計算詞彙重疊率
        overlap = len(expected_words.intersection(actual_words))
        overlap_ratio = overlap / len(expected_words) if expected_words else 0
        
        # 長度檢查
        length_ratio = len(actual) / len(expected) if expected else 1
        
        # 綜合評分
        quality_score = (overlap_ratio * 0.6) + (min(length_ratio, 2.0) / 2.0 * 0.4)
        
        return quality_score > 0.3
    
    async def optimize_model_parameters(self, training_data: List[TrainingExample]):
        """優化模型參數"""
        print("\n⚙️ 優化模型參數...")
        
        # 分析訓練資料特徵
        domain_stats = {}
        for example in training_data:
            domain = example.domain
            if domain not in domain_stats:
                domain_stats[domain] = {
                    'count': 0,
                    'avg_input_length': 0,
                    'avg_output_length': 0,
                    'complexity_scores': []
                }
            
            stats = domain_stats[domain]
            stats['count'] += 1
            stats['avg_input_length'] += len(example.user_input)
            stats['avg_output_length'] += len(example.ai_response)
            
            # 計算複雜度分數
            complexity = self._calculate_complexity(example.user_input)
            stats['complexity_scores'].append(complexity)
        
        # 計算平均值
        for domain, stats in domain_stats.items():
            if stats['count'] > 0:
                stats['avg_input_length'] /= stats['count']
                stats['avg_output_length'] /= stats['count']
                stats['avg_complexity'] = sum(stats['complexity_scores']) / len(stats['complexity_scores'])
        
        # 基於統計資料調整AI服務參數
        optimization_config = self._generate_optimization_config(domain_stats)
        
        print("📈 優化配置:")
        for domain, config in optimization_config.items():
            print(f"  • {domain}: {config}")
        
        return optimization_config
    
    def _calculate_complexity(self, text: str) -> float:
        """計算文本複雜度"""
        words = text.split()
        sentences = text.split('.')
        
        # 基本複雜度指標
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
        
        # 特殊字符和數字的比例
        special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
        special_ratio = special_chars / len(text) if text else 0
        
        complexity = (avg_word_length * 0.3) + (avg_sentence_length * 0.5) + (special_ratio * 100 * 0.2)
        return min(complexity, 10.0)  # 限制在0-10範圍內
    
    def _generate_optimization_config(self, domain_stats: Dict) -> Dict[str, Dict]:
        """生成優化配置"""
        config = {}
        
        for domain, stats in domain_stats.items():
            # 基於統計資料調整參數
            complexity = stats.get('avg_complexity', 5.0)
            
            domain_config = {
                'temperature': max(0.1, min(1.0, complexity / 10.0)),
                'max_tokens': int(stats['avg_output_length'] * 1.2) if stats['avg_output_length'] > 0 else 500,
                'top_p': 0.9 if complexity > 6 else 0.8,
                'frequency_penalty': 0.1 if domain == 'writing' else 0.0
            }
            
            config[domain] = domain_config
        
        return config
    
    async def run_optimization(self):
        """執行完整的模型優化流程"""
        print("🚀 開始AI模型優化...")
        print("=" * 60)
        
        # 1. 載入訓練資料
        training_data = self.load_all_training_data()
        if not training_data:
            print("❌ 沒有可用的訓練資料")
            return
        
        # 2. 評估當前模型性能
        test_data = training_data[:100]  # 使用前100個作為測試集
        performance = await self.evaluate_model_performance(test_data)
        
        print(f"\n📊 模型性能評估結果:")
        print(f"  • 整體準確率: {performance['overall_accuracy']:.2%}")
        print(f"  • 評估範例數: {performance['total_evaluated']}")
        
        print(f"\n📈 各領域性能:")
        for domain, accuracy in performance['domain_accuracies'].items():
            print(f"  • {domain}: {accuracy:.2%}")
        
        # 3. 優化模型參數
        optimization_config = await self.optimize_model_parameters(training_data)
        
        # 4. 保存優化結果
        results = {
            'timestamp': asyncio.get_event_loop().time(),
            'performance': performance,
            'optimization_config': optimization_config,
            'training_data_count': len(training_data)
        }
        
        self._save_optimization_results(results)
        
        print("\n✅ 模型優化完成！")
        print("💡 建議定期重新執行優化以持續改善模型性能")
        
        return results
    
    def _save_optimization_results(self, results: Dict):
        """保存優化結果"""
        results_dir = "./optimization_results"
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"optimization_results_{int(asyncio.get_event_loop().time())}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 優化結果已保存到: {filepath}")

# 全域優化器實例
model_optimizer = ModelOptimizer()

async def main():
    """主函數"""
    await model_optimizer.run_optimization()

if __name__ == "__main__":
    asyncio.run(main())