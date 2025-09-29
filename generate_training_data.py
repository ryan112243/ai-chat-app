#!/usr/bin/env python3
"""
訓練數據生成腳本
自動生成多域大數據訓練集
"""

import os
import sys
from training_data_manager import training_manager

def main():
    """主函數 - 生成訓練數據"""
    print("🚀 開始生成AI訓練數據集...")
    print("=" * 50)
    
    # 定義要生成數據的領域
    domains = ['math', 'programming', 'writing', 'dialogue', 'mun']
    
    # 每個領域生成的範例數量
    examples_per_domain = 300
    
    try:
        # 生成全面的訓練數據集
        print(f"📊 正在為 {len(domains)} 個領域生成訓練數據...")
        print(f"📈 每個領域將生成約 {examples_per_domain} 個範例")
        print()
        
        all_training_data = training_manager.generate_comprehensive_dataset(
            domains=domains,
            examples_per_domain=examples_per_domain
        )
        
        print()
        print("=" * 50)
        print(f"✅ 總共生成了 {len(all_training_data)} 個訓練範例")
        
        # 按領域統計
        domain_stats = {}
        for example in all_training_data:
            domain = example.domain
            domain_stats[domain] = domain_stats.get(domain, 0) + 1
        
        print("\n📊 各領域數據統計:")
        for domain, count in domain_stats.items():
            print(f"  • {domain}: {count} 個範例")
        
        # 保存訓練數據
        filename = "comprehensive_training_data.json"
        training_manager.save_training_data(all_training_data, filename)
        
        # 按領域分別保存
        print("\n💾 按領域保存數據...")
        for domain in domains:
            domain_examples = [ex for ex in all_training_data if ex.domain == domain]
            domain_filename = f"{domain}_training_data.json"
            training_manager.save_training_data(domain_examples, domain_filename)
            print(f"  • {domain}: {len(domain_examples)} 個範例 → {domain_filename}")
        
        # 生成高品質數據集
        print("\n🌟 生成高品質數據集...")
        high_quality_data = training_manager.filter_by_quality(all_training_data, min_score=0.8)
        high_quality_filename = "high_quality_training_data.json"
        training_manager.save_training_data(high_quality_data, high_quality_filename)
        print(f"  • 高品質數據: {len(high_quality_data)} 個範例 → {high_quality_filename}")
        
        print("\n" + "=" * 50)
        print("🎉 訓練數據生成完成！")
        print(f"📁 數據保存位置: {training_manager.data_dir}/processed/")
        print("\n建議下一步:")
        print("1. 檢查生成的數據品質")
        print("2. 根據需要調整數據生成參數")
        print("3. 將數據整合到AI訓練流程中")
        print("4. 定期更新和擴充訓練數據")
        
    except Exception as e:
        print(f"❌ 生成訓練數據時發生錯誤: {str(e)}")
        sys.exit(1)

def generate_domain_specific_data():
    """生成特定領域的額外數據"""
    print("\n🎯 生成特定領域的額外數據...")
    
    # 為程式設計領域生成更多數據（因為需求較高）
    programming_data = training_manager.generate_synthetic_data('programming', 500)
    programming_filename = "programming_extended_data.json"
    training_manager.save_training_data(programming_data, programming_filename)
    print(f"  • 程式設計擴展數據: {len(programming_data)} 個範例")
    
    # 為數學領域生成更多數據
    math_data = training_manager.generate_synthetic_data('math', 400)
    math_filename = "math_extended_data.json"
    training_manager.save_training_data(math_data, math_filename)
    print(f"  • 數學擴展數據: {len(math_data)} 個範例")

def show_data_statistics():
    """顯示數據統計資訊"""
    print("\n📈 數據統計資訊:")
    
    processed_dir = os.path.join(training_manager.data_dir, "processed")
    if not os.path.exists(processed_dir):
        print("  尚未生成任何數據")
        return
    
    total_examples = 0
    files = os.listdir(processed_dir)
    
    for filename in files:
        if filename.endswith('.json'):
            filepath = os.path.join(processed_dir, filename)
            try:
                examples = training_manager.load_training_data(filename)
                count = len(examples)
                total_examples += count
                print(f"  • {filename}: {count} 個範例")
            except Exception as e:
                print(f"  • {filename}: 讀取失敗 ({str(e)})")
    
    print(f"\n  總計: {total_examples} 個訓練範例")

if __name__ == "__main__":
    # 檢查命令行參數
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "stats":
            show_data_statistics()
        elif command == "extended":
            generate_domain_specific_data()
        elif command == "help":
            print("訓練數據生成腳本使用說明:")
            print("  python generate_training_data.py        # 生成完整訓練數據集")
            print("  python generate_training_data.py stats  # 顯示數據統計")
            print("  python generate_training_data.py extended # 生成擴展數據")
            print("  python generate_training_data.py help   # 顯示幫助")
        else:
            print(f"未知命令: {command}")
            print("使用 'python generate_training_data.py help' 查看使用說明")
    else:
        # 默認執行完整數據生成
        main()
        
        # 詢問是否生成擴展數據
        try:
            response = input("\n是否生成擴展數據？(y/N): ").strip().lower()
            if response in ['y', 'yes', '是']:
                generate_domain_specific_data()
        except KeyboardInterrupt:
            print("\n\n👋 數據生成已完成，感謝使用！")