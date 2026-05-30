import re
import os

def clean_corpus(input_path, output_path, encoding='utf-8'):
    """
    清洗北大人民日报语料，生成“标准分词答案”文件。
    清洗规则：
    1. 去除每行开头的文章编号（如 19980101-...）
    2. 去除词性标记（/n, /v 等）
    3. 去除复合词标记（[ ]nt 等）
    4. 保留原始词语和标点，词与词之间保留空格
    """
    
    # 统计处理了多少行，方便看进度
    count = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'r', encoding=encoding) as f_in, \
         open(output_path, 'w', encoding=encoding) as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            # 按空格切分原始行
            # 原始行示例: 19980101-01-001-001/m  迈向/v  二十一/m  世纪/n
            tokens = line.split()
            
            # --- 步骤1: 处理行首编号 ---
            # 如果第一个token很长且包含日期信息，通常是编号，直接丢弃
            if tokens and '199801' in tokens[0] and '/m' in tokens[0]:
                tokens = tokens[1:]
            
            clean_tokens = []
            for token in tokens:
                # --- 步骤2: 处理复合词标记 [ ] ---
                # 示例: [中央/n -> 去掉 [ -> 中央/n
                # 示例: 电视台/n]nt -> 只要前面的部分，后面逻辑会处理
                token = token.replace('[', '')
                
                # --- 步骤3: 提取词语，去除词性 ---
                # 只要 '/' 前面的部分
                # 示例: 迈向/v -> 迈向
                # 示例: 电视台/n]nt -> 电视台
                if '/' in token:
                    word = token.split('/')[0]
                else:
                    word = token # 防止有些奇怪的数据没有标注词性
                
                # 过滤掉空字符串（防止处理产生空值）
                if word:
                    clean_tokens.append(word)
            
            # 只有当提取出内容时才写入
            if clean_tokens:
                # 直接连接，不加空格: "迈向二十一世纪"
                out_line = " ".join(clean_tokens)
                f_out.write(out_line + '\n')
                count += 1

    print(f"清洗完成！已处理 {count} 个句子。")
    print(f"结果已保存至: {output_path}")

# 执行部分
if __name__ == '__main__':
    # 请确保文件名和你上传的文件名一致
    input_file = 'data/01_raw/ChineseCorpus199801.txt' 
    output_file = 'data/02_cleaned/sample_corpus.txt'
    
    try:
        clean_corpus(input_file, output_file)
    except UnicodeDecodeError:
        # 如果utf-8报错，尝试utf-8（windows下常见的编码问题）
        print("UTF-8读取失败，尝试使用UTF-8编码读取...")
        clean_corpus(input_file, output_file, encoding='utf-8')