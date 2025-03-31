import argparse
import json
import os
import datasets
import uuid
import numpy as np

from model_wrappers import HfModelWrapper, VllmModelWrapper, ApiModelWrapper, BatchApiModelWrapper
from constants import LLAMAGUARD_TEMPLATE, SYSTEM_PROMPT, MULTIRULE_SYSTEM_PROMPT, UNSLOTH_INPUT_FIELD
from helpers import apply_llamaguard_template, confirm_model_compatibility, get_stats, confirm_dataset_compatibility, map_llamaguard_output

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import logging
logger = logging.getLogger(__name__)


def main(args):
    confirm_dataset_compatibility(args.dataset_path, args.multirule)

    # Dataset
    dataset = datasets.load_dataset("json", data_files={"placeholder": args.dataset_path})["placeholder"]
    n = args.num_examples if args.num_examples > 0 and args.num_examples < len(dataset) else len(dataset)
    # Shuffle to ensure we get a random subset. Don't shuffle if we're using the whole thing so we can keep track of indices for frequent misclassifications.
    if n < len(dataset):
        dataset.shuffle(seed=42)
    dataset = dataset.select(range(n))

    # Model 
    if "gpt" in args.model or "together_ai" in args.model:
        if args.use_batch_api:
            model = BatchApiModelWrapper(args.model, args.temperature)
        else:
            model = ApiModelWrapper(args.model, args.temperature, args.api_delay, args.retries)
    elif args.use_vllm:
        model = VllmModelWrapper(args.model, args.temperature, args.top_k, args.max_new_tokens)
    else:
        model = HfModelWrapper(args.model, args.temperature, args.top_k, args.max_new_tokens)
    
    # Generation
    sys_prompt = MULTIRULE_SYSTEM_PROMPT if args.multirule else SYSTEM_PROMPT
    if "Llama-Guard" in args.model:
        assert not args.multirule, "Llama-Guard isn't implemented for MultiRule evaluation yet"
        sys_prompt = LLAMAGUARD_TEMPLATE
        template_fn = apply_llamaguard_template
    elif args.multirule:
        sys_prompt = MULTIRULE_SYSTEM_PROMPT
        template_fn = model.apply_chat_template_cot if args.use_cot else model.apply_chat_template
    else:
        sys_prompt = SYSTEM_PROMPT
        template_fn = model.apply_chat_template


    messages = [template_fn(sys_prompt, x[UNSLOTH_INPUT_FIELD]) for x in dataset]


    accuracies = []
    false_positives = 0
    false_negatives = 0
    missing_labels = 0
    false_positive_examples = []
    false_negative_examples = []
    missing_label_examples = []
    for _ in range(args.sample_size):
        outputs = model.get_responses(messages)
        if "Llama-Guard" in args.model:
            outputs = [map_llamaguard_output(output) for output in outputs]

        # Evaluation
        stats = get_stats(outputs, dataset, multirule=args.multirule)

        accuracies.append(stats["accuracy"])

        false_positives += len(stats["false_positives"])
        false_negatives += len(stats["false_negatives"])
        missing_labels += len(stats["nulls"])

        false_positive_examples.extend(stats["false_positives"])
        false_negative_examples.extend(stats["false_negatives"])
        missing_label_examples.extend(stats["nulls"])

    logger.info(f"Raw accuracy per sample: {accuracies}")
    accuracies = np.array(accuracies)
    logger.info(f"Accuracy: {stats["accuracy"]:.2%}")
    logger.info(f"F1 Score: {stats["f1_score"]:.2%}")
    logger.info(f"Accuracy standard deviation = {accuracies.std():.2%}")
    logger.info(f"False Positives: {false_positives} ({false_positives / args.sample_size:0.2f} per sample)")
    logger.info(f"False Negatives: {false_negatives} ({false_negatives / args.sample_size:0.2f} per sample)")
    logger.info(f"Missing expected label: {missing_labels} ({missing_labels  / args.sample_size:0.2f} per sample)")
    logger.info(f"False Positive examples: {sorted(false_positive_examples)}")
    logger.info(f"False Negative examples: {sorted(false_negative_examples)}")
    logger.info(f"Missing expected label examples: {sorted(missing_label_examples)}")

    # Save outputs to disk as a log
    datasets.Dataset.from_list([{"_": _} for _ in outputs]).to_json(f"log/{args.model}_{args.dataset_path}_{uuid.uuid4()}.jsonl")
    
def configure_logging(log_level=None):
    # Determine log level: CLI argument > Environment variable > Default (INFO)
    log_level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=log_level,
        format="{name}:{levelname}: {message}",
        style="{"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    # parser.add_argument('--model', default="gpt-4o-mini", type=str, help="Model name to load")
    # parser.add_argument('--model', default="meta-llama/meta-Llama-3.1-8B-Instruct", type=str, help="Model name to load")
    parser.add_argument("--model", default="meta-llama/Llama-Guard-3-8B", type=str, help="Model name to load")
    
    # Single-rule models
    # parser.add_argument('--model', default="/fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_sft/7500", type=str, help="Model name to load")
    # parser.add_argument('--model', default="/fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_grpo/7500", type=str, help="Model name to load")
    # Multi-rule models
    # parser.add_argument('--model', default="/fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_sft/7500", type=str, help="Model name to load")
    # parser.add_argument('--model', default="/fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_grpo/7500", type=str, help="Model name to load")
    # parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", type=str, help="Model name to load")
    # parser.add_argument("--model", default="/fs/cml-projects/guardian_models/models/Qwen2.5-14B-Instruct/huggingface_sft/lora_multirule", type=str, help="Model name to load")
    
    # Single-rule datasets
    # parser.add_argument("--dataset_path", default="data/singlerule/easy_test_155.jsonl", type=str, help="Path to dataset")
    # Multi-rule datasets
    parser.add_argument("--dataset_path", default="data/multirule/multi_rule_test_98.jsonl", type=str, help="Path to dataset")
    
    parser.add_argument("--num_examples", default=5, type=int, help="Number of examples to evaluate")
    parser.add_argument("--log_level", default=None, type=str, help="Log level")
    parser.add_argument("--use_vllm", default=True, action=argparse.BooleanOptionalAction, help="Use VLLM for generation")
    parser.add_argument("--max_model_len", default=8192, type=int, help="Maximum context length for vllm. Should be based on the space of your gpu, not the model capabilities. If this is too high for the gpu, it will tell you.")
    # Generation parameters taken from gpt-fast
    parser.add_argument("--max_new_tokens", default=512, type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", default=0.6, type=float, help="Generation temperature")
    parser.add_argument("--top_k", default=300, type=int, help="Top k tokens to consider")
    # API stuff
    parser.add_argument("--api_delay", default=None, type=float, help="Minimum delay between API calls")
    parser.add_argument("--retries", default=3, type=int, help="Number of retries for API calls")
    parser.add_argument("--use_batch_api", default=False, action=argparse.BooleanOptionalAction, help="Use batch call for API models")
    # Error bands
    parser.add_argument("--sample_size", default=1, type=int, help="Number of samples used to calculate statistics.")
    parser.add_argument("--use_cot", default=False, action=argparse.BooleanOptionalAction, help="Use COT for generation")
    parser.add_argument("--multirule", default=False, action=argparse.BooleanOptionalAction, help="Use multirule evaluation")
    parser.add_argument("--llamaguard", default=False, action=argparse.BooleanOptionalAction, help="Use LlamaGuard evaluation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    main(args)