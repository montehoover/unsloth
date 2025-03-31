SIZE=0.5B
EPOCH=1
GRPO_CKPT=6000
LR=5.0e-05
python convert_to_huggingface.py --base_path /fs/cml-projects/guardian_models/models/Qwen2.5-$SIZE-Instruct/huggingface_sft/lora_7500/epoch_$EPOCH/ --lora_path /fs/cml-projects/guardian_models/grpo/Qwen2.5-$SIZE-epoch$EPOCH\_$LR\_12_4_32_32_0.2_5/checkpoint-$GRPO_CKPT/ --output_path /fs/cml-projects/guardian_models/models/Qwen2.5-$SIZE-Instruct/huggingface_grpo/lora_7500/epoch_$EPOCH/
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-$SIZE-Instruct/huggingface_grpo/lora_7500/epoch_$EPOCH/ --num_examples 155 --sample_size 20 &> qwen-$SIZE-epoch-$EPOCH-grpo-$GRPO_CKPT.txt