import argparse
import json
import os
import csv
import shutil
import time
import torch
import datasets
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from model_wrappers import HfModelWrapper, VllmModelWrapper, ApiModelWrapper, BatchApiModelWrapper
from constants import LLAMAGUARD_TEMPLATE, METADATA, MULTIRULE_SYSTEM_PROMPT_V2, MULTIRULE_SYSTEM_PROMPT_V2_NON_COT, MULTIRULE_SYSTEM_PROMPT_V3, MULTIRULE_SYSTEM_PROMPT_V4, NEMOGUARD_TEMPLATE, SYSTEM_PROMPT, MULTIRULE_SYSTEM_PROMPT, SYSTEM_PROMPT_EXPERIMENTAL, SYSTEM_PROMPT_EXPERIMENTAL2, SYSTEM_PROMPT_NON_COT, UNSLOTH_INPUT_FIELD
from helpers import ComplianceProjectError, apply_llamaguard_template, configure_logging, confirm_model_compatibility, get_analysis, get_stats, confirm_dataset_compatibility, map_llamaguard_output, save_results, create_enriched_outputs, save_consolidated_outputs, save_consolidated_analysis

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import logging
logger = logging.getLogger(__name__)


TEMP_PATH = f"temp_{time.time_ns()}"


def compute_f1(total_pos: int,
            false_positives: int,
            false_negatives: int) -> float:
    tp = total_pos - false_negatives
    fp = false_positives
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if total_pos == 0:
        recall = 0.0
    else:
        recall = tp / total_pos
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def add_to_csv(
    csv_filename="log/summary.csv",
    model_name="Placeholder",
    test_set="Placeholder",
    f1_score=None,
    f1_stdev=None,
    mod_f1_score=None,
    missing_labels_score=None,
):
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['model_name', 'test_set', 'f1_score', 'f1_stdev', 'missing_labels'])
        # Append the new row
        writer.writerow([model_name, test_set, f1_score, f1_stdev, missing_labels_score])

def get_hf_model(model_path, lora_path):
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    hf_model = lora_model.merge_and_unload()
    hf_model.save_pretrained(TEMP_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(TEMP_PATH)
    return TEMP_PATH


def main(args):
    configure_logging(args.log_level)

    # Dataset
    if os.path.exists(args.dataset_path):
        dataset = datasets.load_dataset("json", data_files={"test": args.dataset_path})["test"]
    else:
        dataset = datasets.load_dataset(args.dataset_path, args.subset, split=args.split)
    confirm_dataset_compatibility(dataset, args.multirule)
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
    else:
        if args.lora_path:
            model_path = get_hf_model(args.model, args.lora_path)
        else:
            model_path = args.model
        if args.use_vllm and "nemoguard" not in model_path:
            model = VllmModelWrapper(model_path, args.temperature, args.top_k, args.max_new_tokens)
        else:
            model = HfModelWrapper(model_path, args.temperature, args.top_k, args.max_new_tokens)
    
    # Generation
    if "Llama-Guard" in args.model:
        sys_prompt = LLAMAGUARD_TEMPLATE
        template_fn = apply_llamaguard_template
    elif "nemoguard" in args.model:
        sys_prompt = NEMOGUARD_TEMPLATE
        template_fn = apply_llamaguard_template
    elif args.multirule:
        sys_prompt = MULTIRULE_SYSTEM_PROMPT_V4
        template_fn = model.apply_chat_template_cot if args.use_cot else model.apply_chat_template
    else:
        sys_prompt = SYSTEM_PROMPT if args.use_cot else SYSTEM_PROMPT_NON_COT
        template_fn = model.apply_chat_template_cot if args.use_cot else model.apply_chat_template

    if "wildguard" in args.model:
        messages = [f"<s>[INST]{sys_prompt}\n{x[UNSLOTH_INPUT_FIELD]}[/INST]" for x in dataset]
    else:
        messages = [template_fn(sys_prompt, x[UNSLOTH_INPUT_FIELD]) for x in dataset]

    accuracies = []
    f1_scores = []
    false_positives = 0
    false_negatives = 0
    missing_labels = 0
    false_positive_examples = []
    false_negative_examples = []
    missing_label_examples = []
    for _ in range(args.sample_size):
        outputs = model.get_responses(messages)
        if "Llama-Guard" in args.model or "nemoguard" in args.model:
            outputs = [map_llamaguard_output(output) for output in outputs]

        if args.go_twice:
            first_outputs = []
            second_output_indices = []
            for i, output in enumerate(outputs):
                if "<answer>" not in output:
                    first_outputs.append(output)
                    second_output_indices.append(i)
            if second_outputs:
                messages = [template_fn(sys_prompt, x[UNSLOTH_INPUT_FIELD], f"{output}<answer>") for x, output in zip(dataset.select(second_output_indices), first_outputs)]
                second_outputs = model.get_responses(messages)
                outputs = [output if i not in second_output_indices else second_outputs[i] for i, output in enumerate(outputs)]

        # Evaluation
        if "GuardReasoner" in args.model:
            relaxed_parsing = True
        else:
            relaxed_parsing = args.relaxed_parsing
        stats = get_stats(outputs, dataset, multirule=args.multirule, relaxed_parsing=relaxed_parsing)

        accuracies.append(stats["accuracy"])
        f1_scores.append(stats["f1_score"])
        false_positives += len(stats["false_positives"])
        false_negatives += len(stats["false_negatives"])
        missing_labels += len(stats["nulls"])

        if args.collect_all:
            false_positive_examples.extend(stats["false_positives"])
            false_negative_examples.extend(stats["false_negatives"])
            missing_label_examples.extend(stats["nulls"])
        else: # collect last run only
            false_positive_examples = stats["false_positives"]
            false_negative_examples = stats["false_negatives"]
            missing_label_examples = stats["nulls"]

    if missing_label_examples:
        # for i in missing_label_examples:
        #     logger.notice(outputs[i])
        logger.notice(json.dumps(outputs[missing_label_examples[0]], indent=4))
    print(f"Raw accuracy per sample: {accuracies}")
    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores)
    print(f"Accuracy: {np.mean(accuracies):.2%} ")
    print(f"F1 Score: {np.mean(f1_scores):.2%}")
    print(f"Accuracy standard deviation = {accuracies.std():.2%}")
    print(f"F1 Score standard deviation = {f1_scores.std():.2%}")
    print(f"False Positives: {false_positives} ({false_positives / args.sample_size:0.2f} per sample)")
    print(f"False Negatives: {false_negatives} ({false_negatives / args.sample_size:0.2f} per sample)")
    print(f"Missing expected label: {missing_labels} ({missing_labels  / args.sample_size:0.2f} per sample)")
    print(f"False Positive examples: {sorted(false_positive_examples)}")
    print(f"False Negative examples: {sorted(false_negative_examples)}")
    print(f"Missing expected label examples: {sorted(missing_label_examples)}")
    print(f"Dataset balance: PASS: {stats["percent_pass"]:.1%} FAIL: {1 - stats["percent_pass"]:.1%}")

    # Save outputs to disk
    parts = args.model.split("/")
    model_name = f"{parts[parts.index("models") + 1]}_ours" if "models" in parts and parts.index("models") < len(parts) - 1 else args.model
    if "lora_multirule_v2" in parts:
        model_name += "_lora"
    if "lora_mix" in parts:
        model_name += "_lora_32000_mix"
    output_path = f"log/{model_name}/{time.time_ns()}"
    enriched_outputs = create_enriched_outputs(dataset, outputs, false_positive_examples, false_negative_examples, missing_label_examples)
    datasets.Dataset.from_list(enriched_outputs).to_json(f"{output_path}/outputs.jsonl")
    print(f"Outputs saved to {output_path}/outputs.jsonl")

    # Append to outputs from previous runs
    save_consolidated_outputs(
        model_name=model_name,
        enriched_outputs=enriched_outputs,
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.split,
        num_examples=len(dataset),
        f1_score=np.mean(f1_scores),
        f1_stdev=f1_scores.std(),
        missing_labels=missing_labels,
        sample_size=args.sample_size
    )

    # Append results to csv
    # if os.path.exists("log/summary.csv"):
    missing_rate = missing_labels / (args.sample_size * len(dataset))
    total_pos = int((len(dataset) * args.sample_size - missing_labels) * (1-stats["percent_pass"]))
    modifified_f1 = compute_f1(total_pos, false_positives, false_negatives)
    if args.lora_path:
        model_name = f"{model_name}_{args.lora_path.split('/')[-2]}"
    add_to_csv(
        csv_filename="log/summary.csv", 
        model_name=model_name,
        test_set=args.subset,
        f1_score=np.mean(f1_scores),
        f1_stdev=f1_scores.std(),
        missing_labels_score=missing_rate,
    ) 


    # Do analysis over length of dialogues and length of rules and stuff
    if args.handcrafted_analysis:
        wrong_predictions = false_positive_examples + false_negative_examples
        analysis_dict = get_analysis(dataset, wrong_predictions, strict=args.strict_metadata)
        
        # Save consolidated analysis for cross-model comparison
        save_consolidated_analysis(
            model_name=model_name,
            analysis_dict=analysis_dict,
            dataset_path=args.dataset_path,
            subset=args.subset,
            split=args.split,
            num_examples=len(dataset),
            f1_score=np.mean(f1_scores),
            missing_labels=missing_labels,
            sample_size=args.sample_size
        )
        
        # save_results(analysis_dict, "log", output_path, model_name, np.mean(accuracies), accuracies.std(), outputs, dataset, false_positive_examples, false_negative_examples, missing_label_examples)
        with open(f"{output_path}/analysis.json", "w") as f:
            json.dump(analysis_dict, f, indent=4)
        print(f"Analysis saved to {output_path}/analysis.json")
    
    if args.lora_path:
        # Clean up temp files
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
        logger.info(f"Temp files removed from {TEMP_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    parser.add_argument('--model', default="gpt-4o-mini", type=str, help="Model name to load")
    # parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str, help="Model name to load")
    # parser.add_argument("--model", default="/fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/huggingface_grpo/lora_mix", type=str, help="Model name to load")
    parser.add_argument("--lora_path",  default=None, type=str, help="Path to lora adapter")
    # parser.add_argument("--lora_path",  default="/fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/lora_7500/epoch_2", type=str, help="Path to lora adapter")
    
    parser.add_argument("--dataset_path", default="/Users/monte/code/system-prompt-compliance/output/formatted/compliance/test_handcrafted_v2.jsonl", type=str, help="Path to dataset")
    # parser.add_argument("--dataset_path", default="tomg-group-umd/compliance", type=str, help="Path to dataset")
    parser.add_argument("--subset", default="compliance", type=str, help="Subset of the dataset to use")
    parser.add_argument("--split", default="test_handcrafted", type=str, help="Split of the dataset to use")
    
    parser.add_argument("--num_examples", default=-1, type=int, help="Number of examples to evaluate")
    parser.add_argument("--log_level", default=None, type=str, help="Log level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "debug", "info", "warning", "error", "critical"])
    parser.add_argument("--use_vllm", default=True, action=argparse.BooleanOptionalAction, help="Use VLLM for generation")
    parser.add_argument("--max_model_len", default=8192, type=int, help="Maximum context length for vllm. Should be based on the space of your gpu, not the model capabilities. If this is too high for the gpu, it will tell you.")
    # Generation parameters taken from gpt-fast
    parser.add_argument("--max_new_tokens", default=8192, type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", default=0.6, type=float, help="Generation temperature")
    parser.add_argument("--top_k", default=300, type=int, help="Top k tokens to consider")
    # API stuff
    parser.add_argument("--api_delay", default=None, type=float, help="Minimum delay between API calls")
    parser.add_argument("--retries", default=3, type=int, help="Number of retries for API calls")
    parser.add_argument("--use_batch_api", default=False, action=argparse.BooleanOptionalAction, help="Use batch call for API models")
    # Error bands
    parser.add_argument("--sample_size", default=1, type=int, help="Number of samples used to calculate statistics.")
    parser.add_argument("--use_cot", default=True, action=argparse.BooleanOptionalAction, help="Use COT for generation")
    parser.add_argument("--multirule", default=True, action=argparse.BooleanOptionalAction, help="Use multirule evaluation")
    parser.add_argument("--handcrafted_analysis", default=True, action=argparse.BooleanOptionalAction, help="do handcrafted analysis")
    parser.add_argument("--go_twice", default=False, action=argparse.BooleanOptionalAction, help="Run the model twice to get a better accuracy")
    parser.add_argument("--relaxed_parsing", default=False, action=argparse.BooleanOptionalAction, help="Use relaxed parsing for finding PASS/FAIL between the xml tags")
    parser.add_argument("--strict_metadata", default=True, action=argparse.BooleanOptionalAction, help="Fail fast with detailed error if metadata is missing instead of skipping examples")
    parser.add_argument("--collect_all", default=False, action=argparse.BooleanOptionalAction, help="Collect all outputs from multiple runs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)