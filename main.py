import argparse
import datetime
import os
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM
from unsloth import is_bfloat16_supported
import re
from vllm import SamplingParams

import logging
logger = logging.getLogger(__name__)
os.environ["WANDB_ENTITY"] = "guardian-models"
os.environ["WANDB_PROJECT"] = "grpo-compliance"


# These constants should match the constants at the top of scripts/convert_unsloth.py
# TODO: Move these constants to a shared file
INPUT_FIELD = "question"
OUTPUT_FIELD = "answer"

COT_OPENING = "\n<reasoning>"
COT_CLOSING = "\n</reasoning>"
LABEL_OPENING = "\n<answer>"
LABEL_CLOSING = "\n</answer>"

SYSTEM_PROMPT = """
You are a guardian model evaluating the compliance of a chatbot agent to various rules. 
You will be given a rule that the agent must follow, along with a conversation between the agent and a user. 
The conversation will be in the following format:\n'User': [user query]\n'Agent': [agent response]\n
Your task is to evaluate the conversation in the context of the rule, and determine whether the rule has been violated by the agent. 
Output your response within xml tags for both the answer and reasoning supporting that answer. 
First provide the reasoning of how the conversation relates to the rule and how the chatbot either violated or did not violate the rule. 
The rule may not be applicable to the conversation, and in that case it canot possibly be violated because it does not apply.
Cite specific examples of how the rule was violated or not violated. If it was not violated, either cite evidence of the agent following the rule, or cite wording in the
rule and conversation that show by definition of terms that the rule is not applicable to the specific conversation.
Then, give the answer as either PASS for not violated or FAIL for violated. 

Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
PASS/FAIL
</answer>
"""


####################
# Reward functions
####################
# Roughly between 0.0 and 4.0, with possibility for negative rewards for length penalty.
# Breakdown:
#   Correctness: 2.5
#   Format: 1.6
#     Xml tags present: 0.1 per tag for total of 0.4
#     All 4 xml tag bonus: 0.4
#     All 4 xml tag plus newlines: 0.4
#     Labels printed correctly: 0.4
#  Length: -0.0001 per character, for maximum of -0.2048
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # Print out the first of the six rollouts for debugging.
    logger.info(f"{'-'*20} Question:\n{q}\nAnswer:\n{answer[0]}\nResponse:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}")
    return [2.5 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def label_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the label is exactly PASS or FAIL."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.4 if r in ["PASS", "FAIL"] else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion is in XML_COT_FORMAT, strictly adhering to newlines before and after every tag."""
    pattern = r"^\n<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r) for r in responses]
    return [0.4 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion is in XML_COT_FORMAT, with flexibility in newlines and whitespace."""
    pattern = r"^\s*<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r) for r in responses]
    return [0.4 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """We want to encourage xml tags to be present, so just give rewards if they are present at all. Let other functions handle extraneous stuff."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.1
    if text.count("\n</reasoning>\n") == 1:
        count += 0.1
    if text.count("\n<answer>\n") == 1:
        count += 0.1
    if text.count("\n</answer>") == 1:
        count += 0.1
    return count

def length_penalty(text) -> float:
    """The shorter the better. The maximum penalty is max_new_tokens * chars * 0.0001, so roughly 512 * 4 * 0.0001 = 0.2048."""
    return -(len(text))*0.0001

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
#######################
# End reward functions
#######################


def extract_xml_answer(text: str) -> str:
    answer = text.split(LABEL_OPENING.strip())[-1]
    answer = answer.split(LABEL_CLOSING.strip())[0]
    return answer.strip()


def get_compliance_examples(dataset_path) -> Dataset:
    data = load_dataset('json', data_files=dataset_path)['train']
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x[INPUT_FIELD]}
        ],
        OUTPUT_FIELD: extract_xml_answer(x[OUTPUT_FIELD])
    }) # type: ignore
    return data # type: ignore


def get_grpo_trainer(args, model, tokenizer, run_name):
    logger.info(f"Getting GRPO model thing...")
    PatchFastRL("GRPO", FastLanguageModel)
    
    logger.info(f"Loading gsm8k dataset...")
    dataset = get_compliance_examples(args.dataset)

    logger.info(f"Setting up GRPO trainer...")
    training_args = GRPOConfig(
        use_vllm = args.use_vllm, # use vLLM for fast inference!
        learning_rate = args.learning_rate,
        adam_beta1 = args.adam_beta1,
        adam_beta2 = args.adam_beta2,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        warmup_steps = args.warmup_steps,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optim,
        logging_steps = args.logging_steps,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps, # Increase to 4 for smoother training
        num_generations = args.num_generations, # Decrease if out of memory
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs, # Only used if max_steps = -1
        max_steps = args.max_steps,
        save_steps = args.save_steps,
        max_grad_norm = args.max_grad_norm,
        report_to = args.report_to, # Can use Weights & Biases
        resume_from_checkpoint = args.resume_from_checkpoint, # Looks in output_dir for last checkpoint
        output_dir = f"{args.output_dir}/{run_name}",
        run_name = run_name,

    )
    if args.correctness_only:
        reward_funcs = [correctness_reward_func]
    else:
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            label_reward_func,
            correctness_reward_func,
        ]
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
    )
    return trainer

def configure_logging(log_level):
    # Determine log level: CLI argument > Environment variable > Default (INFO)
    log_level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=log_level,
        format="{name}:{levelname}: {message}",
        style="{"
    )

def run(args):
    run_name = f"{args.learning_rate:.1e}_{args.num_generations}_{args.gradient_accumulation_steps}_{args.lora_rank}_{args.lora_alpha}_{args.max_grad_norm}_{args.warmup_steps}"
    if args.model_run_name:
        model_run_name = args.model_run_name
    else:
        model_run_name = args.model_name.split("/")[-1]
    run_name = f"{model_run_name}_{run_name}"

    logger.info(f"Loading model...")
    # Load model and tokenizer
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit, # Both false for LoRA 16bit
            load_in_8bit=args.load_in_8bit, # Both false for LoRA 16bit
            fast_inference=args.use_vllm, # Enable vLLM fast inference
            max_lora_rank=args.lora_rank,
            gpu_memory_utilization = args.gpu_memory_utilization, # Reduce if out of memory
        )
    except OSError as e:
        logger.error(f"Did not find model at {args.model_name}. Be sure to run scripts/convert_to_huggingface.py if you want to use a lora SFT checkpoint.")
        raise e

    logger.info(f"Turning model into PEFT...")
    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = args.lora_alpha,
        use_gradient_checkpointing = args.use_gradient_checkpointing, # Enable long context finetuning
        random_state = args.seed,
    )

    logger.info(f"Getting GRPO trainer...")
    trainer = get_grpo_trainer(args, model, tokenizer, run_name)

    # Train model
    logger.info(f"Training model...")
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info(f"Training complete")

    # Save model
    if args.save_model:
        save_path = f"{args.output_dir}/{run_name}/final_model"
        logger.info(f"Saving model...")
        # if args.quantization_method is a list, we will save the model for each quantization method
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    print(f"Saving model with quantization method: {quantization_method}")
                    model.save_pretrained_gguf(
                        save_path,
                        tokenizer,
                        quantization_method=quantization_method,
                    )
                    if args.push_model:
                        model.push_to_hub_gguf(
                            hub_path=args.hub_path,
                            hub_token=args.hub_token,
                            quantization_method=quantization_method,
                        )
            else:
                print(f"Saving model with quantization method: {args.quantization}")
                model.save_pretrained_gguf(save_path, tokenizer, quantization_method=args.quantization)
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=args.hub_path,
                        hub_token=args.hub_token,
                        quantization_method=quantization_method,
                    )
        else:
            model.save_pretrained_merged(save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(save_path, tokenizer, args.hub_token)
    else:
        print("Warning: The model is not saved!")


def parse_args():
    # Define argument parser
    parser = argparse.ArgumentParser(description="🦥 Fine-tune your llm faster using unsloth!")

    model_group = parser.add_argument_group("🤖 Model Options")
    # model_group.add_argument('--model_name', type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct", help="Model name to load")
    # model_group.add_argument('--model_name', type=str, default="/fs/cml-projects/guardian_models/models/Meta-Llama-3.1-8B-Instruct/huggingface_sft/7500", help="Model name to load")
    model_group.add_argument('--model_name', type=str, default="/fs/cml-projects/guardian_models/models/Qwen2-1.5B-Instruct/huggingface_sft/7500", help="Model name to load")
    # model_group.add_argument('--model_run_name', type=str, default=None, help="Name of the model for the run information if you don't want to use the last portion of the model path.")
    model_group.add_argument('--model_run_name', type=str, default="Qwen2-1.5B_7500", help="Name of the model for the run information if you don't want to use the last portion of the model path.")
    model_group.add_argument('--max_seq_length', type=int, default=3512, help="Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!")
    model_group.add_argument('--dtype', type=str, default=None, help="Data type for model (None for auto detection)")
    model_group.add_argument('--load_in_4bit', action=argparse.BooleanOptionalAction, default=False, help="Use 4bit quantization to reduce memory usage")
    model_group.add_argument('--load_in_8bit', action=argparse.BooleanOptionalAction, default=False, help="Use 8bit quantization to reduce memory usage")
    model_group.add_argument('--dataset', type=str, default="data/singlerule/easy_train_8428.jsonl", help="Huggingface dataset to use for training")
    model_group.add_argument('--use_vllm', action=argparse.BooleanOptionalAction, default=True, help="Use vLLM for fast inference in GRPO rollouts.")

    lora_group = parser.add_argument_group("🧠 LoRA Options", "These options are used to configure the LoRA model.")
    lora_group.add_argument('--lora_rank', type=int, default=32, help="Rank for Lora model, default is 32.  (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha parameter, default is 32. (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_dropout', type=float, default=0, help="LoRA dropout rate, default is 0.0 which is optimized.")
    lora_group.add_argument('--bias', type=str, default="none", help="Bias setting for LoRA")
    lora_group.add_argument('--use_gradient_checkpointing', type=str, default="unsloth", help="Use gradient checkpointing")
    lora_group.add_argument('--random_state', type=int, default=3407, help="Random state for reproducibility, default is 3407.")
    lora_group.add_argument('--use_rslora', action='store_true', help="Use rank stabilized LoRA")
    lora_group.add_argument('--loftq_config', type=str, default=None, help="Configuration for LoftQ")

    training_group = parser.add_argument_group("🎓 Training Options")
    training_group.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device during training, default is 1.")
    training_group.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of gradient accumulation steps, default is 1. Increase to 4 for smoother training.")
    training_group.add_argument('--warmup_steps', type=int, default=5, help="Number of warmup steps, default is 5, not used if warmup_ratio is set.")
    training_group.add_argument('--max_steps', type=int, default=-1, help="Maximum number of training steps.")
    training_group.add_argument('--save_steps', type=int, default=500, help="Save steps, default is 250.")
    training_group.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs, only used if max_steps = -1.")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate, default is 2e-4.")
    training_group.add_argument('--optim', type=str, default="paged_adamw_8bit", help="Optimizer type.")
    training_group.add_argument('--adam_beta1', type=float, default=0.9, help="Adam beta1, default is 0.9.")
    training_group.add_argument('--adam_beta2', type=float, default=0.99, help="Adam beta2, default is 0.99.")
    training_group.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay, default is 0.1.")
    training_group.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio, default is 0.1.")
    training_group.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate scheduler type, default is 'cosine'.")
    training_group.add_argument('--num_generations', type=int, default=6, help="Number of GRPO rollouts, default is 6. Decrease if out of memory.")
    training_group.add_argument('--max_prompt_length', type=int, default=3000, help="Maximum prompt length, default is 2048.")
    training_group.add_argument('--max_completion_length', type=int, default=512, help="Maximum completion length, default is 512.")
    training_group.add_argument('--max_grad_norm', type=float, default=0.1, help="Maximum gradient norm, default is 0.1.")
    # training_group.add_argument('--gpu_memory_utilization', type=float, default=0.7, help="GPU memory utilization. This dedicates memory for LoRA backprop and the rest is used for generations. 8B models with LoRA rank 64 work with 0.7 with 24GB VRAM.")
    training_group.add_argument('--gpu_memory_utilization', type=float, default=0.6, help="GPU memory utilization. This dedicates memory for LoRA backprop and the rest is used for generations. 8B models with LoRA rank 64 work with 0.7 with 24GB VRAM.")
    training_group.add_argument('--correctness_only', action=argparse.BooleanOptionalAction, default=False, help="Use correctness reward function only.")
    training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility, default is 3407.")
    training_group.add_argument('--resume_from_checkpoint', action=argparse.BooleanOptionalAction, default=False, help="Resume training from a checkpoint.")

    # Report/Logging arguments
    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument('--report_to', type=str, default="wandb",
        choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"],
        help="The list of integrations to report the results and logs to. Supported platforms are: \n\t\t 'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', 'mlflow', 'neptune', 'tensorboard', and 'wandb'. Use 'all' to report to all integrations installed, 'none' for no integrations.")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Logging steps, default is 1")
    report_group.add_argument('--log_level', type=str, 
        choices=['debug', 'info', 'warning', 'error', 'critical', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
        help="Explicitly set logging level to override environment variable. Defaults to INFO when no environment variable set nor argument passed. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    # Saving and pushing arguments
    save_group = parser.add_argument_group('💾 Save Model Options')
    save_group.add_argument('--output_dir', type=str, default="/fs/cml-projects/guardian_models/grpo", help="Output directory")
    save_group.add_argument('--save_model', action='store_true', help="Save the model after training")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Save method for the model, default is 'merged_16bit'")
    save_group.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    save_group.add_argument('--save_path', type=str, default="/fs/cml-projects/guardian_models/grpo", help="Path to save the model")
    save_group.add_argument('--quantization', type=str, default="q8_0", nargs="+",
        help="Quantization method for saving the model. common values ('f16', 'q4_k_m', 'q8_0'), Check our wiki for all quantization methods https://github.com/unslothai/unsloth/wiki#saving-to-gguf ")

    push_group = parser.add_argument_group('🚀 Push Model Options')
    push_group.add_argument('--push_model', action='store_true', help="Push the model to Hugging Face hub after training")
    push_group.add_argument('--push_gguf', action='store_true', help="Push the model as GGUF to Hugging Face hub after training")
    push_group.add_argument('--hub_path', type=str, default="hf/model", help="Path on Hugging Face hub to push the model")
    push_group.add_argument('--hub_token', type=str, help="Token for pushing the model to Hugging Face hub")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    run(args)
