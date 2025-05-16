# Run with:
# launch launch_eval.sh --classical_logfile_names --gpu_type rtxa5000 --mem 30

python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model gpt-4o
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model Qwen/Qwen3-0.6B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model Qwen/Qwen3-1.7B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model Qwen/Qwen3-4B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model Qwen/Qwen3-8B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model Qwen/Qwen3-14B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model meta-llama/Llama-Guard-3-8B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model nvidia/llama-3.1-nemoguard-8b-content-safety
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model tomg-group-umd/qwen3_8B_compliance_cot_mix_80k_lr1e-5_epoch1_bs128
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset compliance --split test --model tomg-group-umd/qwen3_8B_comp_cot_mix_80k_lr1e-5_epoch1_bs128_32k_mix_grpo_lr1e-6_bs48-epochs1_3k_maxlen1024

python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model gpt-4o
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model Qwen/Qwen3-0.6B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model Qwen/Qwen3-1.7B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model Qwen/Qwen3-4B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model Qwen/Qwen3-8B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model Qwen/Qwen3-14B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model meta-llama/Llama-Guard-3-8B
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model nvidia/llama-3.1-nemoguard-8b-content-safety
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model tomg-group-umd/qwen3_8B_compliance_cot_mix_80k_lr1e-5_epoch1_bs128
python scripts/eval.py --sample_size 4 --dataset_path montehoover/compliance --subset wildguard --split test_harm --model tomg-group-umd/qwen3_8B_comp_cot_mix_80k_lr1e-5_epoch1_bs128_32k_mix_grpo_lr1e-6_bs48-epochs1_3k_maxlen1024

