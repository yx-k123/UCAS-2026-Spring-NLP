#!/bin/bash

# 确保在出错时退出
set -e

# 获取脚本所在目录，确保相对路径正确
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$SCRIPT_DIR/results"

# 确保 results 目录存在
mkdir -p "$RESULTS_DIR"

# 定义日志文件路径
LOG_FILE="$RESULTS_DIR/calc_log.txt"

# 清空之前的日志
> "$LOG_FILE"

# 定义一个辅助函数，用于将输出同时打印到终端并追加到日志文件
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "==============================================="
log "        开始计算不同规模样本的信息熵"
log "==============================================="

# 定义需要处理的样本规模数组 (对应于您 data 目录下的文件名)
SIZES=(100 200 300 500 1000 2000)

for SIZE in "${SIZES[@]}"; do
    EN_FILE="$DATA_DIR/bbc_data_${SIZE}.json"
    ZH_FILE="$DATA_DIR/sina_data_${SIZE}.json"
    OUT_PREFIX="$RESULTS_DIR/size_${SIZE}_"

    log ""
    log ">>> 正在处理规模为 $SIZE 的样本 ..."
    
    # 检查数据文件是否存在
    if [ ! -f "$EN_FILE" ]; then
        log "警告: 找不到英文语料 $EN_FILE，跳过。"
        continue
    fi
    
    if [ ! -f "$ZH_FILE" ]; then
        log "警告: 找不到中文语料 $ZH_FILE，跳过。"
        continue
    fi

    # 运行 python 脚本并将输出同时打印和追加到日志
    python "$SCRIPT_DIR/calc_entropy.py" \
        --en_file "$EN_FILE" \
        --zh_file "$ZH_FILE" \
        --out_prefix "$OUT_PREFIX" 2>&1 | tee -a "$LOG_FILE"
        
    log ">>> 规模 $SIZE 处理完成！结果已保存在 $RESULTS_DIR"
done

log ""
log "==============================================="
log "        所有规模样本信息熵计算完毕！"
log "        详细日志已保存至: $LOG_FILE"
log "==============================================="
