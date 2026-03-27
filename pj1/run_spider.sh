#!/bin/bash

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "用法: $0 <爬虫名称(bbc/sina)> <抓取数量>"
    echo "示例: $0 bbc 100"
    exit 1
fi

SPIDER=$1      # 爬虫名称，应为 bbc 或 sina
COUNT=$2       # 抓取数量

# 验证爬虫名称
if [[ "$SPIDER" != "bbc" && "$SPIDER" != "sina" ]]; then
    echo "错误：爬虫名称必须是 'bbc' 或 'sina'"
    exit 1
fi

# 验证数量为正整数
if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [ "$COUNT" -le 0 ]; then
    echo "错误：抓取数量必须是正整数"
    exit 1
fi

# 项目目录
PROJECT_DIR="/home/kouyx/NLP/pj1"
# Scrapy 可执行文件路径
SCRAPY_CMD="/home/kouyx/NLP/venv/bin/scrapy"

# 输出文件名
OUTPUT_FILE="/home/kouyx/NLP/pj1/data/${SPIDER}_data_${COUNT}.json"

# 进入项目目录并执行爬虫
cd "$PROJECT_DIR" || { echo "无法进入目录 $PROJECT_DIR"; exit 1; }

echo "正在启动爬虫：$SPIDER，抓取数量：$COUNT，输出文件：$OUTPUT_FILE"

$SCRAPY_CMD crawl "$SPIDER" -o "$OUTPUT_FILE" -s CLOSESPIDER_ITEMCOUNT="$COUNT"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "爬取完成！数据已保存至 $OUTPUT_FILE"
else
    echo "爬取失败，请检查错误信息。"
    exit 1
fi