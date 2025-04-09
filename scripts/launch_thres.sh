
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.001
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.01
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.1
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.2
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.3
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.4
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.5
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.6
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.7
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.8
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.9
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.99
python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_sft/lora_multirule --num_examples -1  --pass_threshold 0.999
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.001
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.01
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.1
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.2
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.3
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.4
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.5
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.6
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.7
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.8
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.9
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.99
# python eval.py --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_7500/epoch_1/ --num_examples -1  --pass_threshold 0.999

# python ~/cml-tools/launcher.py launch_thres.sh  --gpus=1 --mem=32 --cpus=4 --timelimit=24 --gpu_type=rtxa5000 --conda=unsloth --qos=scavenger --name thres-guardian