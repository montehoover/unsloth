import argparse
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
os.environ["WANDB_PROJECT"] = "grpo-gsm8k"


SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

XML_COT_FORMAT = """\
    <reasoning>
    {reasoning}
    </reasoning>
    <answer>
    {answer}
    </answer>
    """

####################
# Reward functions
####################
# Roughly between 0.0 and 4.0, with possibility for negative rewards from xmlcount.
# Breakdown:
#   Correctness: 2.0
#   Format: 2.0
#     Xml tags present: 0.125 per tag for total of 0.5
#     All 4 xml tag bonus: 0.5
#     All 4 xml tag plus newlines: 0.5
#     Labels printed correctly: 0.5
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the label is exactly PASS or FAIL."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion is in XML_COT_FORMAT, strictly adhering to newlines."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion is in XML_COT_FORMAT, with flexibility in newlines."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Add rewards for each xml tag that is present, and penalize any characters that exist after </answer>. (Also ends up being a length penalty if </answer> is missing.)"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
#######################
# End reward functions
#######################


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    logger.info(data[0])
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    logger.info(data[0])
    return data # type: ignore


def get_grpo_trainer(args, model, tokenizer):
    logger.info(f"Getting GRPO model thing...")
    PatchFastRL("GRPO", FastLanguageModel)
    
    logger.info(f"Loading gsm8k dataset...")
    dataset = get_gsm8k_questions()

    logger.info(f"Setting up GRPO trainer...")
    training_args = GRPOConfig(
        use_vllm = args.use_vllm, # use vLLM for fast inference!
        learning_rate = args.learning_rate,
        adam_beta1 = args.adam_beta1,
        adam_beta2 = args.adam_beta2,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
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
        output_dir = args.output_dir,
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
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
    logger.info(f"Loading model...")
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        oad_in_4bit=args.load_in_4bit, # False for LoRA 16bit
        fast_inference=args.use_vllm, # Enable vLLM fast inference
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization = args.gpu_memory_utilization, # Reduce if out of memory
    )

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
    trainer = get_grpo_trainer(args, model, tokenizer)

    # Train model
    logger.info(f"Training model...")
    trainer_stats = trainer.train()
    logger.info(f"Training complete")

    # Save model
    if args.save_model:
        logger.info(f"Saving model...")
        # if args.quantization_method is a list, we will save the model for each quantization method
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    print(f"Saving model with quantization method: {quantization_method}")
                    model.save_pretrained_gguf(
                        args.save_path,
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
                model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method=args.quantization)
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=args.hub_path,
                        hub_token=args.hub_token,
                        quantization_method=quantization_method,
                    )
        else:
            model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)
    else:
        print("Warning: The model is not saved!")

    # Test generation
    level = logging.getLevelName(logger.getEffectiveLevel())
    if level == "INFO" or level == "DEBUG":
        text = tokenizer.apply_chat_template([
            {"role" : "user", "content" : "Calculate pi."},
        ], tokenize = False, add_generation_prompt = True)
        sampling_params = SamplingParams(
            temperature = 0.8,
            top_p = 0.95,
            max_tokens = 1024,
        )

        # Before training, without the LoRA adaptor
        output = model.fast_generate(
            [text],
            sampling_params = sampling_params,
            lora_request = None,
        )[0].outputs[0].text
        logger.info(f"""
                    
#################
Before training:
#################
                    
{output}""")

        # After training, with the LoRA adaptor
        model.save_lora("outputs/grpo_saved_lora")
        output = model.fast_generate(
            text,
            sampling_params = sampling_params,
            lora_request = model.load_lora("outputs/grpo_saved_lora"),
        )[0].outputs[0].text
        logger.info(f"""
#################
After training:
#################
                    
{output}""")

        # Also test huggingface style generation
        # model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
        # model = AutoModelForCausalLM.from_pretrained(args.save_path, device_map="auto").eval()
        # input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        # output = model.generate(
        #     input_ids,
        #     max_new_tokens=1024,
        #     temperature=0.8,
        #     top_p=0.95,
        #     return_dict_in_generate=True,
        # )
        # output = tokenizer.decode(output.sequences[0])
        # logger.info(f"Huggingface generation: \n{output}")


def parse_args():
    # Define argument parser
    parser = argparse.ArgumentParser(description="🦥 Fine-tune your llm faster using unsloth!")

    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument('--model_name', type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct", help="Model name to load")
    # model_group.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name to load")
    model_group.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length, default is 512. We auto support RoPE Scaling internally!")
    model_group.add_argument('--dtype', type=str, default=None, help="Data type for model (None for auto detection)")
    model_group.add_argument('--load_in_4bit', action=argparse.BooleanOptionalAction, default=True, help="Use 4bit quantization to reduce memory usage")
    model_group.add_argument('--dataset', type=str, default="yahma/alpaca-cleaned", help="Huggingface dataset to use for training")
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
    training_group.add_argument('--max_steps', type=int, default=3, help="Maximum number of training steps.")
    training_group.add_argument('--save_steps', type=int, default=250, help="Save steps, default is 250.")
    training_group.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs, only used if max_steps = -1.")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate, default is 2e-4.")
    training_group.add_argument('--optim', type=str, default="paged_adamw_8bit", help="Optimizer type.")
    training_group.add_argument('--adam_beta1', type=float, default=0.9, help="Adam beta1, default is 0.9.")
    training_group.add_argument('--adam_beta2', type=float, default=0.99, help="Adam beta2, default is 0.99.")
    training_group.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay, default is 0.1.")
    training_group.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio, default is 0.1.")
    training_group.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate scheduler type, default is 'cosine'.")
    training_group.add_argument('--num_generations', type=int, default=6, help="Number of GRPO rollouts, default is 6. Decrease if out of memory.")
    training_group.add_argument('--max_prompt_length', type=int, default=256, help="Maximum prompt length, default is 256.")
    training_group.add_argument('--max_completion_length', type=int, default=200, help="Maximum completion length, default is 200.")
    training_group.add_argument('--max_grad_norm', type=float, default=0.1, help="Maximum gradient norm, default is 0.1.")
    training_group.add_argument('--gpu_memory_utilization', type=float, default=0.6, help="GPU memory utilization, default is 0.6. Reduce if out of memory.")
    training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility, default is 3407.")
    
    # Report/Logging arguments
    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument('--report_to', type=str, default="none",
        choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"],
        help="The list of integrations to report the results and logs to. Supported platforms are: \n\t\t 'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', 'mlflow', 'neptune', 'tensorboard', and 'wandb'. Use 'all' to report to all integrations installed, 'none' for no integrations.")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Logging steps, default is 1")
    report_group.add_argument('--log_level', type=str, 
        choices=['debug', 'info', 'warning', 'error', 'critical', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
        help="Explicitly set logging level to override environment variable. Defaults to INFO when no environment variable set nor argument passed. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    # Saving and pushing arguments
    save_group = parser.add_argument_group('💾 Save Model Options')
    save_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    save_group.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True, help="Save the model after training")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Save method for the model, default is 'merged_16bit'")
    save_group.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    save_group.add_argument('--save_path', type=str, default="model", help="Path to save the model")
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
