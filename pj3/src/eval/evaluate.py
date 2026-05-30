# -*- coding: utf-8 -*-
def to_region(segmentation):
    """
    核心算法：将分词结果转换为区间集合。
    例如："我 爱 AI" -> [(0,1), (1,2), (2,4)]
    这样对比区间重合度，才能算出准确的 F1。
    """
    region = []
    start = 0
    # 按空格切分，得到词列表
    words = segmentation.strip().split()
    for word in words:
        end = start + len(word)
        region.append((start, end))
        start = end
    return set(region)

def evaluate(gold_file, pred_file):
    """
    计算 Precision, Recall, F1
    """
    with open(gold_file, 'r', encoding='utf-8') as fg, \
         open(pred_file, 'r', encoding='utf-8') as fp:
        
        gold_lines = fg.readlines()
        pred_lines = fp.readlines()

    # 确保行数一致
    if len(gold_lines) != len(pred_lines):
        print(f"警告：标准答案有 {len(gold_lines)} 行，但预测结果有 {len(pred_lines)} 行。请检查文件对应关系！")
        return

    total_hit = 0    # 分对的词总数
    total_gold = 0   # 标准答案的总词数
    total_pred = 0   # 模型输出的总词数

    for gold_line, pred_line in zip(gold_lines, pred_lines):
        # 1. 转换为区间集合
        gold_regions = to_region(gold_line)
        pred_regions = to_region(pred_line)

        # 2. 统计数量
        total_gold += len(gold_regions)
        total_pred += len(pred_regions)
        
        # 3. 计算交集（即分对的词）
        hits = len(gold_regions & pred_regions)
        total_hit += hits

    # --- 计算最终指标 (Micro-Average) ---
    # 准确率 P = 切对的 / 切出的总数
    precision = total_hit / total_pred if total_pred > 0 else 0
    
    # 召回率 R = 切对的 / 标准答案总数
    recall = total_hit / total_gold if total_gold > 0 else 0
    
    # F1值
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("=" * 30)
    print(f"【评估结果】 (基于 {len(gold_lines)} 个句子)")
    print(f"准确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1值 (F1-Score):    {f1:.4f}")
    print("=" * 30)

if __name__ == "__main__":
    # 确保这两个文件名正确
    GOLD_FILE = "data/03_experiment/50_lines_sampled.txt"  # 标准答案（带空格）
    PRED_FILE = "results/task1_baseline/qwen_32b.txt"    # 模型预测（第一步生成的）
    
    evaluate(GOLD_FILE, PRED_FILE)