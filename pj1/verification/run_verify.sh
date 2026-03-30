#!/bin/bash

# 确保在出错时退出
set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$SCRIPT_DIR/results"

# 确保 results 目录存在
mkdir -p "$RESULTS_DIR"

# 定义日志文件路径
LOG_FILE="$RESULTS_DIR/zipf_log.txt"

# 清空之前的日志
> "$LOG_FILE"

# 定义辅助函数
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "==============================================="
log "        开始进行齐夫定律 (Zipf's Law) 验证"
log "==============================================="

# 定义需要处理的样本规模数组
SIZES=(100 200 300 500 1000 2000)

for SIZE in "${SIZES[@]}"; do
    EN_FILE="$DATA_DIR/bbc_data_${SIZE}.json"
    OUT_IMAGE="$RESULTS_DIR/zipf_law_en_${SIZE}.png"

    log ""
    log ">>> 正在验证规模为 $SIZE 的英文语料样本 ..."
    
    # 检查数据文件是否存在
    if [ ! -f "$EN_FILE" ]; then
        log "警告: 找不到英文语料 $EN_FILE，跳过。"
        continue
    fi

    # 运行 python 脚本并将输出保存到日志
    python "$SCRIPT_DIR/verify_zipf.py" \
        --en_file "$EN_FILE" \
        --out_image "$OUT_IMAGE" 2>&1 | tee -a "$LOG_FILE"
        
    log ">>> 规模 $SIZE 验证完成！图表保存在 $OUT_IMAGE"
done

log ""
log "==============================================="
log "        所有规模样本齐夫定律验证完毕！"
log "        详细输出已保存至: $LOG_FILE"
log "        结果图表已保存至: $RESULTS_DIR"
log "==============================================="
