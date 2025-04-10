# python eval.py --model together_ai/google/gemma-2b-it --num_examples 155 --sample_size 10 &> gemma2_2b.txt
# python eval.py --model together_ai/google/gemma-2-9b-it --num_examples 155 --sample_size 10 &> gemma2_9b.txt
# python eval.py --model together_ai/google/gemma-2-27b-it --num_examples 155 --sample_size 10 &> gemma2_27b.txt

python eval.py --model together_ai/mistralai/Mistral-7B-Instruct-v0.3 --num_examples 155 --sample_size 10 &> mistral_7b.txt
python eval.py --model together_ai/mistralai/Mistral-Small-24B-Instruct-2501 --num_examples 155 --sample_size 10 &> mistral_24b.txt
python eval.py --model together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1 --num_examples 155 --sample_size 10 &> mistral_47b.txt
python eval.py --model together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1 --num_examples 155 --sample_size 10 &> mistral_141b.txt

python eval.py --model together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo --num_examples 155 --sample_size 10 &> llama_3b.txt
python eval.py --model together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --num_examples 155 --sample_size 10 &> llama_8b.txt
python eval.py --model together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --num_examples 155 --sample_size 10 &> llama_70b.txt
python eval.py --model together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo --num_examples 155 --sample_size 10 &> llama_405b.txt

python eval.py --model together_ai/deepseek-ai/DeepSeek-R1 --num_examples 155 --sample_size 10 &> deepseek_r1.txt