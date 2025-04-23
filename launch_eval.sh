# Run with:
# launch launch_eval.sh --classical_logfile_names --gpu_type rtxa5000 --mem 30

# Our Test Set
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model Qwen/Qwen2.5-0.5B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard  

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model Qwen/Qwen2.5-1.5B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model Qwen/Qwen2.5-3B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-3B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-3B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model Qwen/Qwen2.5-7B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model Qwen/Qwen2.5-14B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-14B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Qwen2.5-14B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model meta-llama/Llama-Guard-3-8B
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Llama-Guard-3-8B/huggingface_sft/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Llama-Guard-3-8B/huggingface_sft/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model meta-llama/meta-Llama-3.1-8B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model /fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model gpt-4o-mini
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset handcrafted --model gpt-4o


# XSTest
python scripts/eval_xstest.py --sample_size  5 --model Qwen/Qwen2.5-0.5B-Instruct
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval_xstest.py --sample_size  5 --model Qwen/Qwen2.5-1.5B-Instruct
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval_xstest.py --sample_size  5 --model Qwen/Qwen2.5-3B-Instruct
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval_xstest.py --sample_size  5 --model Qwen/Qwen2.5-7B-Instruct
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval_xstest.py --sample_size  5 --model Qwen/Qwen2.5-14B-Instruct
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval_xstest.py --sample_size  5 --model meta-llama/Llama-Guard-3-8B
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval_xstest.py --sample_size  5 --model meta-llama/meta-Llama-3.1-8B-Instruct
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval_xstest.py --sample_size  5 --model /fs/cml-projects/guardian_models/models/Qwen2.5-1.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval_xstest.py --sample_size  5 --model gpt-4o-mini
python scripts/eval_xstest.py --sample_size  5 --model gpt-4o


# WildGuard
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model Qwen/Qwen2.5-0.5B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model Qwen/Qwen2.5-1.5B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model Qwen/Qwen2.5-3B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model Qwen/Qwen2.5-7B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model Qwen/Qwen2.5-14B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model meta-llama/Llama-Guard-3-8B
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model meta-llama/meta-Llama-3.1-8B-Instruct
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_multirule_v2
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model /fs/cml-projects/guardian_models/models/Qwen2.5-0.5B-Instruct/huggingface_grpo/lora_wildguard

python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model gpt-4o-mini
python scripts/eval.py --sample_size 1 --dataset_path tomg-group-umd/compliance --subset wildguard  --model gpt-4o
