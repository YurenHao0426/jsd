export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Declare an associative array to store model mappings
# TODO:Replace with actual model name and path
declare -A model_dict=(
    ["model_name_1"]="/path/to/model1"  
    ["model_name_2"]="/path/to/model2"
)

# 外层循环：遍历model_dict
for exp_name in "${!model_dict[@]}"; do
    model="${model_dict[$exp_name]}"
    echo "Evaluating model: $model"

    python OpenCodeEval/main.py  --model_name $model \
                    --task "MBPP" \
                    --save "test_mbpp/output_mbpp_step_${exp_name}" \
                    --num_gpus 1 \
                    --batch_size 378 \
                    --max_tokens 4096 \
                    --temperature 0.0 \
                    --seed 0 \
                    --time_out 3.0 \
                    --prompt_type "Instruction" \
                    --model_type "Chat" \
                    --prompt_prefix "" \
                    --prompt_suffix "" \
                    --trust_remote_code

done


