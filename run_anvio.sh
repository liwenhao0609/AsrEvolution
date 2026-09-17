#!/bin/bash

# 安装GNU Parallel（如果没有）
# conda install -c conda-forge parallel

input_dir="kegg_annotations"
output_dir="metabolism_results"
KEGG_DATA_DIR="$HOME/databases/kegg"

mkdir -p "$output_dir"

# 获取CPU核心数的一半
cpu_cores=$(nproc)
parallel_jobs=$((cpu_cores / 2))
[ $parallel_jobs -lt 1 ] && parallel_jobs=1

# 导出变量
export output_dir KEGG_DATA_DIR

echo "使用 $parallel_jobs 个并行任务"

# 定义处理函数
process_kegg() {
    kegg_file="$1"
    base_name=$(basename "$kegg_file" .kegg)
    output_prefix="$output_dir/${base_name}_metabolism"
    
    echo "[$(date +%H:%M:%S)] 处理: $base_name"
    
    anvi-estimate-metabolism \
        --enzymes-txt "$kegg_file" \
        -O "$output_prefix" \
        --output-modes modules \
        --kegg-data-dir "$KEGG_DATA_DIR" \
        --include-kos-not-in-kofam
    
    if [ $? -eq 0 ]; then
        echo "✓ 完成: $base_name"
        return 0
    else
        echo "✗ 失败: $base_name" >&2
        return 1
    fi
}

export -f process_kegg

# 并行处理
find "$input_dir" -name "*.kegg" -print0 | \
    parallel -0 -j "$parallel_jobs" --bar process_kegg

echo "所有任务完成！"