# Run with:
# launch launch_sweep.sh --classical_logfile_names --gpu_type rtxa5000 --mem 36

# python main.py

# python main.py --max_grad_norm 0.15
# python main.py --max_grad_norm 0.2
# python main.py --max_grad_norm 0.25

# python main.py --lr_scheduler_type constant_with_warmup --warmup_steps 20
# python main.py --lr_scheduler_type constant_with_warmup --warmup_steps 30
# python main.py --lr_scheduler_type constant_with_warmup --warmup_steps 40

# python main.py --num_generations 2
# python main.py --num_generations 10
# python main.py --num_generations 14

# python main.py --gradient_accumulation_steps 4
# python main.py --gradient_accumulation_steps 16
# python main.py --gradient_accumulation_steps 26

# python main.py --learning_rate 1e-4
# python main.py --learning_rate 1e-5
# python main.py --learning_rate 1e-6

# python main.py --max_grad_norm 0.1 --lr_scheduler_type cosine --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 1e-6
# python main.py --max_grad_norm 0.1 --lr_scheduler_type cosine --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 2e-4
# python main.py --max_grad_norm 0.1 --lr_scheduler_type cosine --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 1e-6
# python main.py --max_grad_norm 0.1 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 1e-6

# python main.py --max_grad_norm 0.1 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 2e-4
# python main.py --max_grad_norm 0.1 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 1e-6
# python main.py --max_grad_norm 0.1 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 2e-4
# python main.py --max_grad_norm 0.1 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 1e-6
# python main.py --max_grad_norm 0.1 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
# python main.py --max_grad_norm 0.1 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 2e-4
# python main.py --max_grad_norm 0.1 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 1e-6

# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 2e-4
# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 1e-6
# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 2e-4
# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 1e-6
# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 2e-4
# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 1e-6

# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 2e-4
# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 1e-6
# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 2e-4
# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 1e-6
# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 2e-4
# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 2e-4
# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 1e-6

# python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 10 --learning_rate 1e-5


# Learning rate
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 5e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 2e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 5e-7
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 2e-7
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-7

# Num generations
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 10 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 14 --gradient_accumulation_steps 1 --learning_rate 1e-6
# python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 18 --gradient_accumulation_steps 1 --learning_rate 1e-6 #CUDA OOM on rtxa5000

# Gradient clipping
python main.py --max_grad_norm 0.15 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.25 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6

# Gradient accumulation
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 2 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 4 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 8 --learning_rate 1e-6

# Warmup steps
python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 10 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 30 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6

# Starting model
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6 --model_name meta-llama/meta-Llama-3.1-8B-Instruct
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6 --model_name /fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/checkpoints/8B_lora_7500/huggingface

# Best guess
python main.py --max_grad_norm 0.2 --lr_scheduler_type constant_with_warmup --warmup_steps 20 --num_generations 14 --gradient_accumulation_steps 1 --learning_rate 1e-6

# Grid search over learning rate, num_generatiosn, and gradient accumulation steps
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 8 --gradient_accumulation_steps 1 --learning_rate 1e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 8 --gradient_accumulation_steps 1 --learning_rate 1e-5
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 12 --gradient_accumulation_steps 1 --learning_rate 1e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 12 --gradient_accumulation_steps 1 --learning_rate 1e-5
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 8 --gradient_accumulation_steps 4 --learning_rate 1e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 8 --gradient_accumulation_steps 4 --learning_rate 1e-5
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 12 --gradient_accumulation_steps 4 --learning_rate 1e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 12 --gradient_accumulation_steps 4 --learning_rate 1e-5

# Grid search over lora rank and alpha
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6 --lora_rank 64 --lora_alpha 64
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6 --lora_rank 64 --lora_alpha 32
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6 --lora_rank 32 --lora_alpha 64
python main.py --max_grad_norm 0.2 --lr_scheduler_type cosine --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6 --lora_rank 128 --lora_alpha 64
