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
                    --task "HumanEval" \
                    --save "test/output_humaneval_${exp_name}" \
                    --num_gpus 1 \
                    --batch_size 164 \
                    --max_tokens 4096 \
                    --temperature 0.1 \
                    --seed 0 \
                    --prompt_type "Completion" \
                    --model_type "Chat" \
                    --prompt_prefix $'Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\n' \
                    --prompt_suffix $'\n```\n' \

done