# 定义基础变量
export VLLM_ENABLE_V1_MULTIPROCESSING=0
config="vllm"
num_limit=100
max_token=3072
ntrain=0
split="test"
log_path="log/test"

# 创建日志目录
mkdir -p ${log_path}

# Declare an associative array to store model mappings
# TODO:Replace with actual model name and path
declare -A model_dict=(
    ["model_name_1"]="/path/to/model1"  
    ["model_name_2"]="/path/to/model2"
)


# Outer Loop：遍历model_dict
for exp_name in "${!model_dict[@]}"; do
    model="${model_dict[$exp_name]}"

    # Inner Loop：遍历不同的 eval_nppl 值
    for eval_nppl in 2 3 4 5 6 7 8; do
        log_file="${log_path}/${exp_name}_nppl${eval_nppl}.log" # 日志文件名包含模型名称和 eval_nppl
        echo "Starting job for model: $model, eval_nppl: $eval_nppl, logging to $log_file"

        # 启动评估任务
        CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python main_eval_instruct.py \
        --batch_size 100 \
        --model ${model} \
        --max_token ${max_token} \
        --ntrain ${ntrain} \
        --config "${config}_${exp_name}_nppl${eval_nppl}" \
        --limit ${num_limit} \
        --split ${split} \
        --temperature 0.0 \
        --top_p 1.0 \
        --seed 0 \
        --problem_type "clean" \
        --output_file "${log_path}/${exp_name}_nppl${eval_nppl}.json" \
        --eval_nppl ${eval_nppl} > "$log_file" 2>&1 
    done
done


# 等待所有后台任务完成
wait