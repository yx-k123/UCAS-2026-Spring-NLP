# -*- coding: utf-8 -*-
import os
from openai import OpenAI

# 1. 配置
INPUT_FILE = 'data/01_raw/other.txt'
OUTPUT_FILE = 'results/task1_baseline/qwen_32b.txt'
BATCH_SIZE = 20  # 一次发给模型处理的行数 (建议 10-50 之间，取决于句子长度)

# 2. 读取 Key
try:
    with open('configs/api.txt', 'r') as f:
        api_key = f.read().strip()
except FileNotFoundError:
    print("错误：未找到 api.txt")
    exit()

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=api_key,
    default_headers={"X-Failover-Enabled": "true"},
)

def process_batch(lines_batch):

    input_text = "\n".join(lines_batch)
    

    system_prompt = (
        "You are a high-efficiency Chinese Word Segmentation tool.\n"
        "Task: Segment the provided text into words using spaces.\n"
        "Strict Constraints:\n"
        "1. The input contains multiple lines. You must process ALL lines.\n"
        "2. Output format must strictly match the line count of the input.\n"
        "3. Do NOT merge lines. Do NOT delete lines.\n"
        "4. Only output the segmented text. No header, no footer, no markdown.\n"
        "5. Use single spaces for segmentation."
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            model="Qwen3-32B",
            stream=False, 
            max_tokens=4096,
            temperature=0.1,
        )
        

        result = response.choices[0].message.content.strip()
        

        if result.startswith("```"):
            result = result.replace("```text", "").replace("```", "").strip()
            
        return result

    except Exception as e:
        print(f"\n[Batch Error]: {e}")
        return "\n".join(lines_batch) 

def main():

    with open(OUTPUT_FILE, 'w', encoding='utf-8', errors='ignore') as f:
        pass

    if not os.path.exists(INPUT_FILE):
        print(f"找不到文件: {INPUT_FILE}")
        return


    print(f"正在读取 {INPUT_FILE} ...")
    lines = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    total_lines = len(lines)
    print(f"共 {total_lines} 行有效数据。每批处理 {BATCH_SIZE} 行。")

    with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='ignore') as f_out:
        for i in range(0, total_lines, BATCH_SIZE):

            batch = lines[i : i + BATCH_SIZE]
            
            print(f"正在处理第 {i+1} - {min(i+BATCH_SIZE, total_lines)} 行...", end="", flush=True)
            

            segmented_block = process_batch(batch)
            

            if segmented_block:
                f_out.write(segmented_block + "\n")
            
            print(" 完成")

    print(f"\n处理完毕！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()