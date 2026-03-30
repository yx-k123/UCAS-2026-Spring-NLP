import json
import re
import math
import argparse
from collections import Counter
import matplotlib.pyplot as plt

def verify_zipf(file_path, output_image):
    # 读取英文数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    text = " ".join([item.get('content', '') for item in data])
    text = text.lower()
    
    # 提取英文单词 (由连续 [a-z] 组成的字符串)
    words = re.findall(r'[a-z]+', text)
    word_counter = Counter(words)
    
    # 获取按频率降序排列的单词和频数
    sorted_words = word_counter.most_common()
    
    ranks = []
    frequencies = []
    
    for rank, (word, freq) in enumerate(sorted_words, start=1):
        ranks.append(rank)
        frequencies.append(freq)
        
    # 计算对数值 (处理 log(0) 的边界，从 1 开始没问题)
    log_ranks = [math.log10(r) for r in ranks]
    log_freqs = [math.log10(f) for f in frequencies]
    
    # 齐夫定律理想曲线: f = C / r (取 C = frequencies[0] 近似)
    C = frequencies[0]
    expected_log_freqs = [math.log10(C / r) for r in ranks]
    
    # 绘制图形
    plt.figure(figsize=(10, 6))
    
    # 实际频率点
    plt.scatter(log_ranks, log_freqs, label='Actual Data', color='blue', s=10)
    
    # 理论齐夫定律参考线 (斜率为 -1)
    plt.plot(log_ranks, expected_log_freqs, label="Zipf's Law (Slope = -1)", color='red', linestyle='--')
    
    plt.title("Zipf's Law Verification (English Corpus)")
    plt.xlabel('log(Rank)')
    plt.ylabel('log(Frequency)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图像
    plt.savefig(output_image, dpi=300)
    print(f"验证完成，图表已保存至: {output_image}")
    
    # 在控制台输出前 10 个词的 Rank*Freq (根据齐夫定律，此值应大致为一个常数)
    print("\n--- 首批单词分析 (Rank * Frequency ≈ Constant) ---")
    print(f"{'Rank':<8} | {'Word':<15} | {'Freq':<10} | {'Rank * Freq':<15}")
    for i in range(10):
        if i < len(sorted_words):
            word, freq = sorted_words[i]
            rank = i + 1
            print(f"{rank:<8} | {word:<15} | {freq:<10} | {rank * freq:<15}")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--en_file', type=str, default='../data/bbc_data_100.json')
    parser.add_argument('--out_image', type=str, default='zipf_law_en.png')
    args = parser.parse_args()

    data_path = args.en_file
    output_png = args.out_image
    
    out_dir = os.path.dirname(output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    try:
        verify_zipf(data_path, output_png)
    except FileNotFoundError:
        print(f"未能找到数据文件: {data_path}，请检查路径。")