#!/usr/bin/env python3

import re
from collections import Counter

def analyze_data_quality():
    print("=== 分析当前训练数据质量问题 ===\n")
    
    # Analyze HQ data
    print("--- HQ (高质量) 数据分析 ---")
    hq_lengths = []
    hq_languages = {'english': 0, 'other': 0}
    
    with open('data/clean_positive.txt', 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:  # 只分析前100条
                break
            text = line.replace('__label__HQ ', '').strip()
            length = len(text)
            hq_lengths.append(length)
            
            # Simple language detection
            english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            if len(english_words) > len(text.split()) * 0.5:
                hq_languages['english'] += 1
            else:
                hq_languages['other'] += 1
                
            if i < 10:
                print(f"HQ样例 {i+1} (长度:{length}): {text[:80]}...")
    
    print(f"HQ平均长度: {sum(hq_lengths)/len(hq_lengths):.1f} 字符")
    print(f"HQ语言分布: {hq_languages}")
    print()
    
    # Analyze LQ data  
    print("--- LQ (低质量) 数据分析 ---")
    lq_lengths = []
    lq_languages = {'english': 0, 'other': 0}
    
    with open('data/clean_negative.txt', 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:  # 只分析前100条
                break
            text = line.replace('__label__LQ ', '').strip()
            length = len(text)
            lq_lengths.append(length)
            
            # Simple language detection
            english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            if len(english_words) > len(text.split()) * 0.5:
                lq_languages['english'] += 1
            else:
                lq_languages['other'] += 1
                
            if i < 10:
                print(f"LQ样例 {i+1} (长度:{length}): {text[:80]}...")
    
    print(f"LQ平均长度: {sum(lq_lengths)/len(lq_lengths):.1f} 字符")
    print(f"LQ语言分布: {lq_languages}")
    print()
    
    # Compare with test fixtures
    print("--- 测试fixtures分析 ---")
    with open('tests/fixtures/low_quality_cc.txt', 'r') as f:
        cc_text = f.read()
    with open('tests/fixtures/high_quality_wiki_reference.txt', 'r') as f:
        wiki_text = f.read()
        
    print(f"CC测试文本长度: {len(cc_text)} 字符")
    print(f"Wiki测试文本长度: {len(wiki_text)} 字符")
    print(f"CC文本预览: {cc_text[:100]}...")
    print(f"Wiki文本预览: {wiki_text[:100]}...")
    
    print("\n=== 问题分析 ===")
    print("1. HQ数据过短，质量实际很低")
    print("2. LQ数据主要是非英语内容") 
    print("3. 测试fixtures与训练数据质量模式不匹配")
    print("4. 模型学到了错误的模式: 英语=HQ, 非英语=LQ")

if __name__ == "__main__":
    analyze_data_quality()