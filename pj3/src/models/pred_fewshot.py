# -*- coding: utf-8 -*-
import os
from openai import OpenAI

os.makedirs('results/task4_prompting', exist_ok=True)
# 1. 配置
INPUT_FILE = 'data/01_raw/other.txt'
OUTPUT_FILE = 'results/task4_prompting/qwen_32b_fewshot.txt' 
BATCH_SIZE = 5  

# 2. 定义 Few-Shot 示例 (In-Context Learning)
# 选取符合《人民日报》分词规范的典型例句
# 包含：普通句子、人名处理、时间日期、复合词
FEW_SHOT_PROMPT = """
Below are some examples of standard Chinese Word Segmentation:

Input: 迈向充满希望的新世纪
Output: 迈向 充满 希望 的 新 世纪

Input: 中共中央总书记江泽民
Output: 中共中央 总书记 江泽民

Input: 即使是这般光景，也不可轻言放弃。
Output: 即使 是 这般 光景 ， 也 不可 轻言 放弃 。


Input: 天有六气，降生五味。
Output: 天 有 六 气 ， 降生 五味

Now, please segment the following lines strictly following the style above.
"""

# 3. 读取 Key
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

    # 拼接当前批次的输入数据
    current_input_block = ""
    for line in lines_batch:
        current_input_block += f"Input: {line}\n"

    final_user_prompt = f"{FEW_SHOT_PROMPT}\nTarget Inputs:\n{current_input_block}\nOutput (only the segmented lines):"

    system_prompt = (
        "You are a high-efficiency Chinese Word Segmentation tool.\n"
        "Strict Constraints:\n"
        "1. Process the 'Target Inputs' line by line.\n"
        "2. Output format must strictly match the line count of the input.\n"
        "3. Output ONLY the segmented result. Do NOT repeat 'Input:' or 'Output:'.\n"
        "4. Use single spaces for segmentation.\n"
        "5. Do NOT output markdown code blocks."
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_user_prompt}
            ],
            model="Qwen3-8B",
            stream=False, 
            max_tokens=4096,
            temperature=0.1, # 低温度保证复现性
        )
        
        result = response.choices[0].message.content.strip()
        
        # 清洗数据：有时候模型会忍不住加上 "Output: " 前缀，这里做个简单清洗
        cleaned_lines = []
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith("Output:"):
                line = line.replace("Output:", "").strip()
            if line.startswith("```"): # 去除 markdown
                continue
            if line:
                cleaned_lines.append(line)
        
        # 重新组合，如果行数不对，打印警告
        if len(cleaned_lines) != len(lines_batch):
            print(f" [警告] 输入{len(lines_batch)}行 -> 输出{len(cleaned_lines)}行，可能存在对齐问题")
            
        return "\n".join(cleaned_lines)

    except Exception as e:
        print(f"\n[Batch Error]: {e}")
        return "\n".join(lines_batch)

def main():
    # 清空输出文件
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
    print(f"共 {total_lines} 行有效数据。使用 Few-Shot 模式，每批处理 {BATCH_SIZE} 行。")

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