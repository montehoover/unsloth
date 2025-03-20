# Run with:
# launch launch_good.sh --classical_logfile_names --gpu_type rtxa6000 --mem 30

# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6 --save_model --save_steps 500 --max_steps -1 --model_name /fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_sft/7500
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 12 --gradient_accumulation_steps 4 --learning_rate 1e-4 --num_train_epochs 4 --resume_from_checkpoint --model_name /fs/cml-projects/guardian_models/models/Qwen2-1.5B-Instruct/checkpoints/1B_lora_7500/epoch_4/huggingface_sft


