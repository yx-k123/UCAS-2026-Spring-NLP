# -*- coding: utf-8 -*-
import random
import os

def random_sample_two_files(input_file1, input_file2, output_file1, output_file2, num_lines=50):
    """从两个文件中同步随机抽取相同行号的内容"""
    
    # 读取第一个文件的所有行
    with open(input_file1, 'r', encoding='UTF-8') as f:
        lines1 = f.readlines()
    
    # 读取第二个文件的所有行
    with open(input_file2, 'r', encoding='UTF-8') as f:
        lines2 = f.readlines()
    
    total_lines1 = len(lines1)
    total_lines2 = len(lines2)
    
    print(f"文件1总行数: {total_lines1}")
    print(f"文件2总行数: {total_lines2}")
    
    # 确保两个文件行数一致
    if total_lines1 != total_lines2:
        print("警告：两个文件的行数不一致！")
        min_lines = min(total_lines1, total_lines2)
        print(f"将使用较小的行数: {min_lines}")
    else:
        min_lines = total_lines1
    
    # 确保抽取的行数不超过总行数
    sample_size = min(num_lines, min_lines)
    print(f"抽取行数: {sample_size}")
    
    # 随机抽取行号（索引）
    random_indices = sorted(random.sample(range(min_lines), sample_size))
    print(f"\n随机抽取的行号: {[i+1 for i in random_indices[:10]]}..." if len(random_indices) > 10 else f"\n随机抽取的行号: {[i+1 for i in random_indices]}")
    
    # 根据相同的行号抽取两个文件的内容
    sampled_lines1 = [lines1[i] for i in random_indices]
    sampled_lines2 = [lines2[i] for i in random_indices]
    
    # 保存到输出文件1
    with open(output_file1, 'w', encoding='UTF-8') as f:
        f.writelines(sampled_lines1)
    
    # 保存到输出文件2
    with open(output_file2, 'w', encoding='UTF-8') as f:
        f.writelines(sampled_lines2)
    
    print(f"\n已成功从两个文件抽取 {sample_size} 行")
    print(f"文件1保存到: {output_file1}")
    print(f"文件2保存到: {output_file2}")
    
    # 打印前3行作为预览
    print("\n文件1前3行预览:")
    for i, line in enumerate(sampled_lines1[:3], 1):
        print(f"{i}. {line.strip()}")
    
    print("\n文件2前3行预览:")
    for i, line in enumerate(sampled_lines2[:3], 1):
        print(f"{i}. {line.strip()[:100]}..." if len(line.strip()) > 100 else f"{i}. {line.strip()}")

if __name__ == "__main__":
    # 输入文件
    input_file1 = "data/02_cleaned/cleaned.corpus.txt"
    input_file2 = "data/02_cleaned/sample_corpus.txt"
    
    # 输出文件
    output_file1 = "data/03_experiment/50_lines_test.txt"
    output_file2 = "data/03_experiment/50_lines_sampled.txt"
    
    # 同步随机抽取50行
    random_sample_two_files(input_file1, input_file2, output_file1, output_file2, num_lines=50)
