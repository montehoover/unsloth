for i in $(seq 0 4);
do
    python scripts/convert_to_huggingface.py --base_path /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_base/ --lora_path /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/lora_7500/epoch_$i/ --output_path /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_sft/lora_7500/epoch_$i/
    python scripts/convert_to_huggingface.py --base_path /fs/cml-projects/guardian_models/models/Qwen2.5-3B-Instruct/huggingface_base/ --lora_path /fs/cml-projects/guardian_models/models/Qwen2.5-3B-Instruct/lora_7500/epoch_$i/ --output_path /fs/cml-projects/guardian_models/models/Qwen2.5-3B-Instruct/huggingface_sft/lora_7500/epoch_$i/
    python scripts/convert_to_huggingface.py --base_path /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_base/ --lora_path /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/lora_7500/epoch_$i/ --output_path /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_7500/epoch_$i/
done

python scripts/convert_to_huggingface.py --base_path /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_sft/lora_7500/epoch_1/ --lora_path /fs/cml-projects/guardian_models/grpo/Qwen2.5-1.5B-epoch1_1.0e-04_12_4_32_32_0.2_5/checkpoint-500/ --output_path /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/


python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples 155 --sample_size 20 &> qwen-1.5B-epoch-1-grpo.txt