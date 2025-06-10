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
from constants import EXPLANATION_OPENING, LABEL_CLOSING, LABEL_OPENING, LLAMAGUARD_TEMPLATE, METADATA, MULTIRULE_SYSTEM_PROMPT_V2, MULTIRULE_SYSTEM_PROMPT_V2_NON_COT, MULTIRULE_SYSTEM_PROMPT_V3, MULTIRULE_SYSTEM_PROMPT_V4, MULTIRULE_SYSTEM_PROMPT_V5, SYSTEM_PROMPT, MULTIRULE_SYSTEM_PROMPT, SYSTEM_PROMPT_EXPERIMENTAL, SYSTEM_PROMPT_EXPERIMENTAL2, SYSTEM_PROMPT_NON_COT, UNSLOTH_INPUT_FIELD
from helpers import ComplianceProjectError, apply_llamaguard_template, configure_logging, confirm_model_compatibility, extract_xml_answer, get_analysis, get_stats, confirm_dataset_compatibility, map_llamaguard_output, save_results

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import logging
logger = logging.getLogger(__name__)


TEMP_PATH = f"temp_{time.time_ns()}"

def get_hf_model(model_path, lora_path, obj_only=False):
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    hf_model = lora_model.merge_and_unload()
    if obj_only:
        return hf_model
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
    
    dataset = dataset.select(range(args.example, args.example + 1))

    # Model 
    if args.lora_path:
        model_path = get_hf_model(args.model, args.lora_path)
    else:
        model_path = args.model
    
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
    model = HfModelWrapper(model_path, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=args.max_new_tokens)
    
    messages = [model.apply_chat_template(MULTIRULE_SYSTEM_PROMPT_V5, x[UNSLOTH_INPUT_FIELD], enable_thinking=args.use_cot) for x in dataset]
    outputs = model.get_responses(messages)
    for output in outputs:
        print(output)
        print()

    print("######")
    print("Adding explanation to the output.")
    print("######")
    prediction = extract_xml_answer(outputs[0], LABEL_OPENING, LABEL_CLOSING)
    assistant_content = f"{LABEL_OPENING}\n{prediction}\n{LABEL_CLOSING}\n{EXPLANATION_OPENING}"
    messages = [model.apply_chat_template(MULTIRULE_SYSTEM_PROMPT_V5, x[UNSLOTH_INPUT_FIELD], assistant_content) for x in dataset]
    outputs = model.get_responses(messages)
    for output in outputs:
        print(output)
        print()

    if args.lora_path:
        # Clean up temp files
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
        logger.info(f"Temp files removed from {TEMP_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    # parser.add_argument('--model', default="gpt-4o-mini", type=str, help="Model name to load")
    # parser.add_argument('--model', default="meta-llama/meta-Llama-3.1-8B-Instruct", type=str, help="Model name to load")
    # parser.add_argument("--model", default="meta-llama/Llama-Guard-3-8B", type=str, help="Model name to load")
    
    # Single-rule models
    # parser.add_argument('--model', default="/fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_sft/7500", type=str, help="Model name to load")
    # parser.add_argument('--model', default="/fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_grpo/7500", type=str, help="Model name to load")
    # Multi-rule models
    parser.add_argument('--model', default="tomg-group-umd/Qwen3-8B_train_80k_mix_grpo_ex9000_lr1e-6_bs48_len1024", type=str, help="Path to base model")
    # parser.add_argument('--model', default="Qwen/Qwen3-8B", type=str, help="Model name to load")
    # parser.add_argument('--model', default="/fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_grpo/7500", type=str, help="Model name to load")
    # parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str, help="Model name to load")
    # parser.add_argument("--model", default="/fs/cml-projects/guardian_models/models/Qwen2.5-14B-Instruct/huggingface_grpo/lora_multirule_v2", type=str, help="Model name to load")
    parser.add_argument("--lora_path",  default=None, type=str, help="Path to lora adapter")
    # parser.add_argument("--lora_path",  default="/fs/cml-projects/guardian_models/grpo/Qwen2.5-7B_mix_no_partial_5.0e-05_12_1_32_32_0.2_5/checkpoint-1050", type=str, help="Path to lora adapter")
    # Single-rule datasets
    # parser.add_argument("--dataset_path", default="output/handcrafted/test.jsonl", type=str, help="Path to dataset")
    # Multi-rule datasets
    # parser.add_argument("--dataset_path", default="data/multirule/multi_rule_test_98_cot.jsonl", type=str, help="Path to dataset")
    # parser.add_argument("--dataset_path", default="data/test.jsonl", type=str, help="Path to dataset")
    parser.add_argument("--dataset_path", default="tomg-group-umd/compliance", type=str, help="Path to dataset")
    parser.add_argument("--subset", default="compliance", type=str, help="Subset of the dataset to use")
    parser.add_argument("--split", default="test_handcrafted", type=str, help="Split of the dataset to use")
    
    parser.add_argument("--num_examples", default=-1, type=int, help="Number of examples to evaluate")
    parser.add_argument("--log_level", default=None, type=str, help="Log level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "debug", "info", "warning", "error", "critical"])
    parser.add_argument("--use_vllm", default=True, action=argparse.BooleanOptionalAction, help="Use VLLM for generation")
    parser.add_argument("--max_model_len", default=8192, type=int, help="Maximum context length for vllm. Should be based on the space of your gpu, not the model capabilities. If this is too high for the gpu, it will tell you.")
    # Generation parameters taken from gpt-fast
    parser.add_argument("--max_new_tokens", default=4096, type=int, help="Maximum tokens to generate")
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
    parser.add_argument("--example", default=0, type=int, help="Example to run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)