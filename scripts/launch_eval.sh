# launch with:
# launch launch_eval.sh --classical_logfile_names 

python eval.py --model gpt-4o      --num_examples -1  --use_batch_api --dataset_path ../data/easy_test_225.jsonl
python eval.py --model gpt-4o      --num_examples 225 --use_batch_api --dataset_path ../data/easy_train_8872.jsonl
python eval.py --model gpt-4o-mini --num_examples -1  --use_batch_api --dataset_path ../data/easy_test_225.jsonl
python eval.py --model gpt-4o-mini --num_examples 225 --use_batch_api --dataset_path ../data/easy_train_8872.jsonl 
