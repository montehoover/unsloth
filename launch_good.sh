# Run with:
# launch launch_good.sh --classical_logfile_names --gpu_type rtxa6000 --mem 30

# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 12 --gradient_accumulation_steps 4 --learning_rate 5e-5 --num_train_epochs 4 --model_run_name Qwen2.5-14B_7500 --model_name /fs/cml-projects/guardian_models/models/Qwen2.5-14B-Instruct/huggingface_sft/lora_7500 --gpu_memory_utilization 0.7
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 12 --gradient_accumulation_steps 4 --learning_rate 5e-5 --num_train_epochs 4 --model_run_name Meta-Llama-3.1-8B_7500 --model_name /fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_sft/7500

