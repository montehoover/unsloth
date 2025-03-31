#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:30:00

module load cuda/11.8.0
conda init bash
conda shell.bash activate unsloth

# declare -a sizes=("1.5B" "3B" "7B")
declare -a sizes=("0.5B")

for size in "${sizes[@]}"
do
  # python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-$size-Instruct/huggingface_base/ --num_examples 115 --sample_size 20 &> qwen-$size-base.txt
  for epoch in $(seq 0 4);
  do
    python convert_to_huggingface.py --base_path /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_base/ --lora_path /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/lora_7500/epoch_$epoch/ --output_path /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_sft/lora_7500/epoch_$epoch/
    python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-$size-Instruct/huggingface_sft/lora_7500/epoch_$epoch/ --num_examples 115 --sample_size 20 &> qwen-$size-epoch-$epoch.txt
  done
done
