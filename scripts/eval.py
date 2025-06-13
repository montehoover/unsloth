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
from constants import LLAMAGUARD_TEMPLATE, LLAMAGUARD_TEMPLATE2, MULTIRULE_SYSTEM_PROMPT_V5, NEMOGUARD_TEMPLATE, NEMOGUARD_TEMPLATE2, UNSLOTH_INPUT_FIELD
from helpers import apply_llamaguard_template, configure_logging, extract_xml_answer, get_analysis, get_binary_classification_report, get_stats, confirm_dataset_compatibility, map_llamaguard_output, create_enriched_outputs, map_nemoguard_output, print_formatted_report, save_consolidated_outputs, save_consolidated_analysis

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
    recall=None,
    false_positive_rate=None,
    auc=None
):
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['model_name', 'test_set', 'f1_score', 'f1_stdev', 'missing_labels', 'recall', 'false_positive_rate', 'auc'])
        # Append the new row
        writer.writerow([model_name, test_set, f1_score, f1_stdev, missing_labels_score, recall, false_positive_rate, auc])

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
    if "qwen3" in args.model.lower():
        if args.use_cot:
            temperature = 0.6
            top_p = 0.95
            top_k = 20
        else:
            temperature = 0.7
            top_p = 0.8
            top_k = 20
    else:
        temperature = args.temperature
        top_p = 1.0
        top_k = args.top_k 
    if "gpt" in args.model or "together_ai" in args.model:
        if args.use_batch_api:
            model = BatchApiModelWrapper(args.model, temperature=temperature)
        else:
            model = ApiModelWrapper(args.model, temperature=temperature, api_delay=args.api_delay, retries=args.retries)
    else:
        if "nemoguard" in args.model:
            model_path = get_hf_model("meta-llama/Meta-Llama-3.1-8B-Instruct", args.model)
        elif args.lora_path:
            model_path = get_hf_model(args.model, args.lora_path)
        else:
            model_path = args.model
        if args.use_vllm: #and "nemoguard" not in model_path:
            model = VllmModelWrapper(model_path, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len)
        else:
            model = HfModelWrapper(model_path, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=args.max_new_tokens)
    
    # Generation
    if "Llama-Guard" in args.model:
        sys_prompt = LLAMAGUARD_TEMPLATE2 if "wildguard" in args.subset else LLAMAGUARD_TEMPLATE
        template_fn = apply_llamaguard_template
    elif "nemoguard" in args.model:
        sys_prompt = NEMOGUARD_TEMPLATE2 if "wildguard" in args.subset else NEMOGUARD_TEMPLATE
        template_fn = apply_llamaguard_template
    else:
        sys_prompt = MULTIRULE_SYSTEM_PROMPT_V5
        template_fn = model.apply_chat_template

    if "wildguard" in args.model:
        messages = [f"<s>[INST]{sys_prompt}\n{x[UNSLOTH_INPUT_FIELD]}[/INST]" for x in dataset]
    elif ("Llama-Guard" in args.model or "nemoguard" in args.model):
        if "wildguard" in args.subset:
            get_transcript = lambda x: extract_xml_answer(x[UNSLOTH_INPUT_FIELD], "<transcript>", "</transcript>") if "<transcript>" in x[UNSLOTH_INPUT_FIELD] else x[UNSLOTH_INPUT_FIELD]
            messages = [template_fn(sys_prompt, get_transcript(x)) for x in dataset]
        else:
            messages = [template_fn(sys_prompt, x[UNSLOTH_INPUT_FIELD]) for x in dataset]
    else:
        messages = [template_fn(sys_prompt, x[UNSLOTH_INPUT_FIELD], enable_thinking=args.use_cot) for x in dataset]

    if args.get_auc:
        non_cot_messages = [template_fn(sys_prompt, x[UNSLOTH_INPUT_FIELD], enable_thinking=False) for x in dataset]
        pos_label_probs, pos_label_logits = model.get_prediction_probs(non_cot_messages)
        report = get_binary_classification_report(dataset, pos_label_probs, args.target_fpr)
        print_formatted_report(report)

    accuracies = []
    f1_scores = []
    recalls = []
    false_positives = 0
    false_negatives = 0
    missing_labels = 0
    false_positive_examples = []
    false_negative_examples = []
    missing_label_examples = []
    for _ in range(args.sample_size):
        outputs = model.get_responses(messages, logit_bias_dict=args.logit_bias_dict)
        if "Llama-Guard" in args.model:
            original_outputs = outputs.copy()
            outputs = [map_llamaguard_output(output) for output in outputs]
        elif "nemoguard" in args.model:
            original_outputs = outputs.copy()
            outputs = [map_nemoguard_output(output) for output in outputs]

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

        # Gather stats
        if "GuardReasoner" in args.model:
            relaxed_parsing = True
        else:
            relaxed_parsing = args.relaxed_parsing
        
        stats = get_stats(outputs, dataset, multirule=args.multirule, relaxed_parsing=relaxed_parsing)
        auc = report.get("auc", None) if args.get_auc else None
        if args.get_auc:
            stats["auc"] = auc

        accuracies.append(stats["accuracy"])
        f1_scores.append(stats["f1_score"])
        recalls.append(stats["recall"])
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
    recalls = np.array(recalls)
    false_positive_rate = false_positives / (args.sample_size * len(dataset))
    print(f"Accuracy: {np.mean(accuracies):.2%} ")
    print(f"F1 Score: {np.mean(f1_scores):.2%}")
    print(f"Recall: {np.mean(recalls):.2%}")
    if auc is not None:
        print(f"AUC: {auc:.2%}")
    print(f"Accuracy standard deviation = {accuracies.std():.2%}")
    print(f"F1 Score standard deviation = {f1_scores.std():.2%}")
    print(f"False Positives: {false_positives} ({false_positives / args.sample_size:0.2f} per sample)")
    print(f"False Positive Rate: {false_positive_rate:.2%}")
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
    if args.enriched_outputs:
        output_text_data = create_enriched_outputs(dataset, outputs, false_positive_examples, false_negative_examples, missing_label_examples)
        # Append to outputs from previous runs
        save_consolidated_outputs(
            model_name=model_name,
            enriched_outputs=output_text_data,
            dataset_path=args.dataset_path,
            subset=args.subset,
            split=args.split,
            num_examples=len(dataset),
            f1_score=np.mean(f1_scores),
            f1_stdev=f1_scores.std(),
            missing_labels=missing_labels,
            sample_size=args.sample_size
        )
    else:
        output_text_data = [{"output": output, "metadata": dataset[i]} for i, output in enumerate(original_outputs)]
    datasets.Dataset.from_list(output_text_data).to_json(f"{output_path}/outputs.jsonl")
    print(f"Outputs saved to {output_path}/outputs.jsonl")

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
        recall= np.mean(recalls),
        false_positive_rate=false_positive_rate,
        auc=auc,
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
    
    if args.lora_path or "nemoguard" in args.model:
        # Clean up temp files
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
        logger.info(f"Temp files removed from {TEMP_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    # parser.add_argument('--model', default="gpt-4o-mini", type=str, help="Model name to load")
    # parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str, help="Model name to load")
    parser.add_argument("--model", default="tomg-group-umd/Qwen3-8B_train_80k_mix_sft_lr1e-5_bs128_ep1_grpo_ex3000_lr1e-6_bs48_len1024", type=str, help="Model name to load")
    parser.add_argument("--lora_path",  default=None, type=str, help="Path to lora adapter")
    # parser.add_argument("--lora_path",  default="/fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/lora_7500/epoch_2", type=str, help="Path to lora adapter")
    
    # parser.add_argument("--dataset_path", default="/Users/monte/code/system-prompt-compliance/output/formatted/compliance/test_handcrafted_v2.jsonl", type=str, help="Path to dataset")
    parser.add_argument("--dataset_path", default="tomg-group-umd/compliance", type=str, help="Path to dataset")
    parser.add_argument("--subset", default="compliance", type=str, help="Subset of the dataset to use")
    parser.add_argument("--split", default="test_handcrafted", type=str, help="Split of the dataset to use")
    
    parser.add_argument("--num_examples", default=-1, type=int, help="Number of examples to evaluate")
    parser.add_argument("--log_level", default=None, type=str, help="Log level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "debug", "info", "warning", "error", "critical"])
    parser.add_argument("--use_vllm", default=True, action=argparse.BooleanOptionalAction, help="Use VLLM for generation")
    parser.add_argument("--max_model_len", default=8192, type=int, help="Maximum context length for vllm. Should be based on the space of your gpu, not the model capabilities. If this is too high for the gpu, it will tell you.")
    # Generation parameters taken from gpt-fast
    parser.add_argument("--max_new_tokens", default=1024, type=int, help="Maximum tokens to generate")
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
    parser.add_argument("--handcrafted_analysis", default=False, action=argparse.BooleanOptionalAction, help="do handcrafted analysis")
    parser.add_argument("--go_twice", default=False, action=argparse.BooleanOptionalAction, help="Run the model twice to get a better accuracy")
    parser.add_argument("--relaxed_parsing", default=False, action=argparse.BooleanOptionalAction, help="Use relaxed parsing for finding PASS/FAIL between the xml tags")
    parser.add_argument("--strict_metadata", default=False, action=argparse.BooleanOptionalAction, help="Fail fast with detailed error if metadata is missing instead of skipping examples")
    parser.add_argument("--collect_all", default=False, action=argparse.BooleanOptionalAction, help="Collect all outputs from multiple runs")
    parser.add_argument("--enriched_outputs", default=False, action=argparse.BooleanOptionalAction, help="Enrich outputs with metadata and save to disk")
    parser.add_argument("--get_auc", default=True, action=argparse.BooleanOptionalAction, help="Calculate AUC for the model")
    parser.add_argument("--target_fpr", default=0.05, type=float, help="Target false positive rate for AUC calculation")
    parser.add_argument("--logit_bias_dict", default=None, type=json.loads, help="Logit bias dictionary for the model. Should be a dict with token ids as keys and bias values as values. If not provided, no logit bias is applied.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)