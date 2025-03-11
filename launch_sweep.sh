# Run with:
# launch launch_sweep.sh --classical_logfile_names --gpu_type rtxa5000 --mem 30

python main.py --max_grad_norm 0.15
python main.py --max_grad_norm 0.2
python main.py --max_grad_norm 0.25

python main.py --lr_scheduler_type "constant_with_warmup" --warmup_steps 20
python main.py --lr_scheduler_type "constant_with_warmup" --warmup_steps 30
python main.py --lr_scheduler_type "constant_with_warmup" --warmup_steps 40

python main.py --num_generations 2
python main.py --num_generations 10
python main.py --num_generations 14

python main.py --gradient_accumulation_steps 4
python main.py --gradient_accumulation_steps 16
python main.py --gradient_accumulation_steps 26

python main.py --learning_rate 1e-4
python main.py --learning_rate 1e-5
python main.py --learning_rate 1e-6

python main.py --max_grad_norm 0.1 --lr_scheduler_type "cosine" --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.1 --lr_scheduler_type "cosine" --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 2e-4
python main.py --max_grad_norm 0.1 --lr_scheduler_type "cosine" --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 1e-6
python main.py --max_grad_norm 0.1 --lr_scheduler_type "cosine" --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 1e-6

python main.py --max_grad_norm 0.1 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 2e-4
python main.py --max_grad_norm 0.1 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.1 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 2e-4
python main.py --max_grad_norm 0.1 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 1e-6
python main.py --max_grad_norm 0.1 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.1 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 2e-4
python main.py --max_grad_norm 0.1 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 1e-6

python main.py --max_grad_norm 0.2 --lr_scheduler_type "cosine" --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 2e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type "cosine" --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type "cosine" --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 2e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type "cosine" --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type "cosine" --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type "cosine" --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 2e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type "cosine" --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 1e-6

python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 2e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 2e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 2 --gradient_accumulation_steps 26 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 2e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 1 --learning_rate 1e-6
python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 2e-4
python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 26 --learning_rate 1e-6

python main.py --max_grad_norm 0.2 --lr_scheduler_type "constant_with_warmup" --warmup_steps 20 --num_generations 6 --gradient_accumulation_steps 10 --learning_rate 1e-5

