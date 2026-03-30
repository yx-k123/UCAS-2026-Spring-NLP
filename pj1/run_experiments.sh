#!/bin/bash

# ===============================================================
# 全自动数据实验流水线
# 目标：对不同样本量 (Sample Sizes) 进行自动化抓取与信息统计分析。
# ===============================================================

# 定义不同数量级的样本量
SIZES=(1000 2000 )

# 确保位于工作目录 /home/kouyx/NLP/pj1
cd "$(dirname "$0")"

# 创建输出目录
mkdir -p statistics/results
mkdir -p verification/results

# 激活 Python 虚拟环境 (如果存在)
if [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi

for size in "${SIZES[@]}"; do
    echo "=========================================================="
    echo " 开始实验流水线 | 目标样本量: $size"
    echo "=========================================================="
    
    # 步骤 1: 运行爬虫抓取文本
    echo "[1/3] 正在抓取数据 (样本量: $size)..."
    bash quotes_spider/run_spider.sh bbc "$size"
    bash quotes_spider/run_spider.sh sina "$size"
    
    # 定义对应生成的文件路径
    EN_DATA="data/bbc_data_${size}.json"
    ZH_DATA="data/sina_data_${size}.json"
    
    # 步骤 2: 计算信息熵并排查频率
    echo "[2/3] 计算文本中字母/单词/汉子的概率与信息熵..."
    python statistics/calc_entropy.py \
        --en_file "$EN_DATA" \
        --zh_file "$ZH_DATA" \
        --out_prefix "statistics/results/size_${size}_"
        
    # 步骤 3: 绘制齐夫定律分布图
    echo "[3/3] 验证自然语言分布并生成齐夫定律散点图..."
    python verification/verify_zipf.py \
        --en_file "$EN_DATA" \
        --out_image "verification/results/zipf_law_en_${size}.png"
        
    echo "- 样本量 $size 分析执行完毕!"
    echo ""
done

echo "🎉 所有规模的自动实验已完成！"
echo "📊 结果已保存在: pj1/statistics/results 和 pj1/verification/results 中"
