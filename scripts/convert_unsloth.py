import argparse
import datasets
from tqdm import tqdm

from constants import (
    INPUT_FIELD,
    OUTPUT_FIELD,
    NUM_RULES_METADATA,
)
from helpers import get_cleaned_fields, get_multirule_input, get_multirule_output, get_singlerule_examples, print_stats

"""
Convert a Compliance dataset in the format:
{
    rules: List[str],        # numbered 
    dialogue: str, 
    discussions: List[str],  # numbered
    explanations: List[str], # numbered 
    labels: List[str]}       # numbered
}

to an eval-friendly dataset in the format:
{
    question: str,           # total = num_original_examples
    answer: str,             # XML tagged as below
    num_rules: int,          # Optional
}
"""

def preprocess_dataset(dataset_path, subset=None, split=None, size=None, local=False, data_dir="data"):
    if local:
        dataset = datasets.load_dataset("json", data_files={"placeholder": dataset_path})["placeholder"]
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    print(f"Examples in {subset} {split}: {len(dataset)}")

    examples = []
    for row_index, row in tqdm(enumerate(dataset)):
        rules, labels, explanations, discussions, dialogue, dialogue_turns, num_rules, num_turns = get_cleaned_fields(row, row_index)
        if args.multirule:
            example = {}
            example[INPUT_FIELD] = get_multirule_input(rules, labels, explanations, dialogue)
            example[OUTPUT_FIELD] = get_multirule_output(labels, explanations, dialogue_turns, num_rules, num_turns)
            example[NUM_RULES_METADATA] = num_rules
            new_examples = [example]
        else:
            new_examples = get_singlerule_examples(
                rules,
                labels,
                explanations,
                discussions,
                dialogue_turns,
                num_rules,
                num_turns,
            )
        examples.extend(new_examples)
    processed_dataset = datasets.Dataset.from_list(examples)

    if size is not None and len(processed_dataset) > size:
        processed_dataset = processed_dataset.shuffle(seed=42)
        processed_dataset = processed_dataset.select(range(size))
    type = "multirule" if args.multirule else "singlerule"
    file_path = f"{data_dir}/{type}/{subset}_{split}_{len(processed_dataset)}.jsonl"
    processed_dataset.to_json(file_path)
    print(f"Saved to {file_path}")
    return file_path


def main(args):
    huggingface_dataset = "tomg-group-umd/compliance"
    subsets = args.subsets
    splits = args.splits
    file_paths = {}
    for subset in subsets:
        for split in splits:
            file_path = preprocess_dataset(huggingface_dataset, subset, split, size=args.train_size, data_dir=args.data_dir)
            file_paths[f"{subset}_{split}"] = file_path
            if args.multirule:
                print_stats(file_path)
    
    if args.extra_examples:
        train_file_path = file_paths["easy_train"]
        val_file_path = file_paths["easy_validation"]
        
        train_dataset = datasets.load_dataset("json", data_files={"placeholder": train_file_path})["placeholder"]
        val_dataset = datasets.load_dataset("json", data_files={"placeholder": val_file_path})["placeholder"]
        
        combined_len = len(train_dataset) + len(val_dataset)
        combined_file_path = f"{args.data_dir}/easy_train_{combined_len}.jsonl"
        
        combined_dataset = datasets.concatenate_datasets([train_dataset, val_dataset])
        combined_dataset = combined_dataset.shuffle(seed=42)
        combined_dataset.to_json(combined_file_path)
        print(f"Combined easy train and validation datasets saved to {combined_file_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--train_size", default=10000, type=int)
    # parser.add_argument("--subsets", type=list, default=["easy", "hard"])
    # parser.add_argument("--splits", type=list, default=["train", "validation", "test"])
    parser.add_argument("--subsets", type=list, default=["multi_rule"])
    parser.add_argument("--splits", type=list, default=["train", "test"])
    parser.add_argument("--extra_examples", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--multirule", default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)