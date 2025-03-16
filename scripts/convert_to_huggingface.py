import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def main(args):
    print(f"loading model from {args.base_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_path, torch_dtype=torch.bfloat16)
    
    print(f"loading lora adapters from {args.lora_path}...")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_path)
    
    print(f"Converting to HuggingFace format and saving to {args.output_path}...")
    hf_model = lora_model.merge_and_unload()
    hf_model.save_pretrained(args.output_path)
    # Downstream applications need the tokenizer in the same directory as the model
    tokenizer = AutoTokenizer.from_pretrained(args.base_path)
    tokenizer.save_pretrained(args.output_path)

    if args.test_output:
        print("Testing output of converted model...")
        inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")
        outputs = hf_model.generate(**inputs, max_new_tokens=10, num_return_sequences=1)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    parser.add_argument("--base_path",   default="/fs/cml-projects/guardian_models/models_xml/Meta-Llama-3.1-8B-Instruct/huggingface_base", type=str, help="Path to base model")
    parser.add_argument("--lora_path",   default="/fs/cml-projects/guardian_models/models_xml/Meta-Llama-3.1-8B-Instruct/sft_output/7500/epoch_4", type=str, help="Path to lora adapter")
    parser.add_argument("--output_path", default="/fs/cml-projects/guardian_models/models_xml/Meta-Llama-3.1-8B-Instruct/huggingface_sft/7500", type=str, help="Path to save converted model")
    parser.add_argument("--test_output", action="store_true", help="Test output of converted model")
    parser.add_argument("--output_dir", default="huggingface", type=str, help="Output directory for HuggingFace model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)