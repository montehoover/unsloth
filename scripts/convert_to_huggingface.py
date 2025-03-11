import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def main(args):
    print("This requires 2x the model size in cpu memory, so 32G for an 8B model.")

    root = "/fs/cml-projects/guardian_models/models"
    if args.model_size == "1B":
        model_name = "Meta-Llama-3.2-1B-Instruct"
    elif args.model_size == "3B":
        model_name = "Meta-Llama-3.2-3B-Instruct"
    elif args.model_size == "8B":
        model_name = "Meta-Llama-3.1-8B-Instruct"

    base_path = f"{root}/{model_name}"
    lora_path = f"{base_path}/checkpoints/{args.model_size}_lora_{args.train_count}"
    hf_directory = f"{lora_path}/huggingface"

    print(f"loading model from {base_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(base_path)
    
    print(f"loading lora adapters from {lora_path}...")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    
    print(f"Converting to HuggingFace format and saving to {hf_directory}...")
    hf_model = lora_model.merge_and_unload()
    hf_model.save_pretrained(hf_directory)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.save_pretrained(hf_directory)


    if args.test_output:
        print("Testing output of converted model...")
        hf_model
        inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")
        outputs = hf_model.generate(**inputs, max_new_tokens=10, num_return_sequences=1)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))



def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    parser.add_argument("--model_size", default="8B", type=str, help="Size of the model", choices=["1B", "3B", "8B"])
    parser.add_argument("--train_count", default=2500, type=int, help="Number of examples", choices=[2500, 5000, 7500])
    parser.add_argument("--test_output", action="store_true", help="Test output of converted model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)