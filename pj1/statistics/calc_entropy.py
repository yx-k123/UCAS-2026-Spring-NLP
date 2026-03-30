import json
import math
import re
from collections import Counter
import jieba

def calculate_entropy(counter):
    total_count = sum(counter.values())
    entropy = 0.0
    probabilities = {}
    for item, count in counter.items():
        prob = count / total_count
        probabilities[item] = prob
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return probabilities, entropy

def process_english(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    text = " ".join([item.get('content', '') for item in data])
    text = text.lower()
    
    # 提取英文字母 [a-z]
    letters = [char for char in text if 'a' <= char <= 'z']
    letter_counter = Counter(letters)
    
    # 提取英文单词 (由连续 [a-z] 组成的字符串)
    words = re.findall(r'[a-z]+', text)
    word_counter = Counter(words)
    
    print(f"  -> 语料统计 | 文章数: {len(data)}, 总字母数: {sum(letter_counter.values())}, 总词数(Tokens): {sum(word_counter.values())}, 独立词库大小(Types): {len(word_counter)}")

    letter_probs, letter_ent = calculate_entropy(letter_counter)
    word_probs, word_ent = calculate_entropy(word_counter)
    return (letter_probs, letter_ent), (word_probs, word_ent)

def process_chinese(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    text = " ".join([item.get('content', '') for item in data])
    
    # 获取纯中文片段，舍弃标点及其他字符
    zh_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    zh_chunks = zh_pattern.findall(text)
    clean_text = "".join(zh_chunks)
    
    # 统计汉字
    char_counter = Counter(clean_text)
    
    # 使用 jieba 进行中文分词处理
    words = []
    for chunk in zh_chunks:
        # lcut 直接返回 list
        words.extend(jieba.lcut(chunk))
    word_counter = Counter(words)
    
    print(f"  -> 语料统计 | 文章数: {len(data)}, 总汉字数: {sum(char_counter.values())}, 总词数(Tokens): {sum(word_counter.values())}, 独立词库大小(Types): {len(word_counter)}")

    char_probs, char_ent = calculate_entropy(char_counter)
    word_probs, word_ent = calculate_entropy(word_counter)
    return (char_probs, char_ent), (word_probs, word_ent)

def display_and_save_top_k(name, probs, entropy, filepath, k=20):
    print(f"[{name}]")
    print(f"信息熵 (Entropy): {entropy:.4f} bits/symbol")
    print(f"Top {k} 概率:")
    # 排序保存
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for item, prob in sorted_probs[:k]:
        print(f"  {item}: {prob:.6f}")
    print()

    # 保存所有概率分布到文件
    output_data = {
        "entropy": entropy,
        "probabilities": dict(sorted_probs)
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Calculate Entropy")
    parser.add_argument('--en_file', type=str, default="../data/bbc_data_100.json", help="Path to english json data")
    parser.add_argument('--zh_file', type=str, default="../data/sina_data_100.json", help="Path to chinese json data")
    parser.add_argument('--out_prefix', type=str, default="", help="Prefix for output JSON files")
    args = parser.parse_args()

    en_file = args.en_file
    zh_file = args.zh_file
    
    # 确保输出目录存在
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    print(f"====== 处理英文数据 (English Corpus) : {en_file} ======")
    try:
        (en_letters_probs, en_letters_ent), (en_words_probs, en_words_ent) = process_english(en_file)
        display_and_save_top_k("英文字母", en_letters_probs, en_letters_ent, f"{args.out_prefix}en_letters_dist.json")
        display_and_save_top_k("英文单词", en_words_probs, en_words_ent, f"{args.out_prefix}en_words_dist.json")
    except FileNotFoundError:
        print(f"未找到 {en_file}，请检查数据路径。")

    print(f"====== 处理中文数据 (Chinese Corpus) : {zh_file} ======")
    try:
        (zh_char_probs, zh_char_ent), (zh_words_probs, zh_words_ent) = process_chinese(zh_file)
        display_and_save_top_k("汉字", zh_char_probs, zh_char_ent, f"{args.out_prefix}zh_chars_dist.json")
        display_and_save_top_k("中文词汇", zh_words_probs, zh_words_ent, f"{args.out_prefix}zh_words_dist.json")
    except FileNotFoundError:
        print(f"未找到 {zh_file}，请检查数据路径。")
