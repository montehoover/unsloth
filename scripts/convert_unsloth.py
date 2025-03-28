import argparse
import os
import random
import datasets
from tqdm import tqdm
import spacy

from constants import (
    UNSLOTH_INPUT_FIELD,
    UNSLOTH_OUTPUT_FIELD,
    TORCHTUNE_INPUT_FIELD,
    TORCHTUNE_OUTPUT_FIELD,
    NUM_RULES_METADATA,
)
from helpers import combine_datasets, get_cleaned_fields, get_multirule_input, get_multirule_output, get_singlerule_examples, print_stats

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

def preprocess_dataset(
    dataset_path, 
    subset=None, 
    split=None, 
    size=None, 
    local=False, 
    data_dir="data", 
    add_cot=False, 
    skip_explanations=False, 
    torchtune=False, 
    torchtune_root=None
):
    if torchtune:
        input_field = TORCHTUNE_INPUT_FIELD
        output_field = TORCHTUNE_OUTPUT_FIELD
    else:
        input_field = UNSLOTH_INPUT_FIELD
        output_field = UNSLOTH_OUTPUT_FIELD
    if local:
        dataset = datasets.load_dataset("json", data_files={"placeholder": dataset_path})["placeholder"]
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    print(f"Examples in {subset} {split}: {len(dataset)}")

    if add_cot:
        print(f"Doing sentence processing for the COTs which is slow. Should take {len(dataset)//1000+1} minutes to process {len(dataset)} examples...")
        # Setup nlp processor
        spacy_model = "en_core_web_sm"
        if not spacy.util.is_package(spacy_model):
            spacy.cli.download(spacy_model)
        nlp_processor = spacy.load(spacy_model)
    else:
        nlp_processor = None

    examples = []
    for row_index, row in tqdm(enumerate(dataset)):
        rules, labels, explanations, discussions, dialogue, dialogue_turns, num_rules, num_turns = get_cleaned_fields(row, row_index)
        if args.multirule:
            example = {}

            # Shuffle all of the lists (rules, labels, explanations) so there is no bias in the rule order
            zipped = list(zip(rules, labels, explanations, discussions))
            random.shuffle(zipped)
            rules, labels, explanations, discussions = zip(*zipped)

            example[input_field] = get_multirule_input(rules, labels, explanations, dialogue)
            example[output_field] = get_multirule_output(labels, explanations, discussions, dialogue_turns, num_rules, num_turns, add_cot, nlp_processor, skip_explanations)
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
                input_field,
                output_field,
            )
        examples.extend(new_examples)
    processed_dataset = datasets.Dataset.from_list(examples)

    if size is not None and len(processed_dataset) > size:
        processed_dataset = processed_dataset.shuffle(seed=42)
        processed_dataset = processed_dataset.select(range(size))
    
    # Construct filename
    type = "multirule" if args.multirule else "singlerule"
    file_path = f"{data_dir}/{type}/{subset}_{split}_{len(processed_dataset)}.jsonl"
    if torchtune:
        root = os.path.abspath(torchtune_root)
        assert os.path.exists(root), f"Did not find the torchtune repo at {root}"
        file_path = f"{root}/{file_path}"
    if add_cot:
        file_path = file_path.replace(".jsonl", "_cot.jsonl")
    
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
            kwargs = {
                "dataset_path": huggingface_dataset,
                "subset": subset,
                "split": split,
                "size": args.train_size,
                "data_dir": args.data_dir,
                "add_cot": args.add_cot,
                "skip_explanations": args.skip_explanations,
                "torchtune": args.torchtune,
                "torchtune_root": args.torchtune_root,
            }
            if args.combine:
                non_cot_filepath = preprocess_dataset(**kwargs, add_cot=False)
                cot_filepath = preprocess_dataset(**kwargs, add_cot=True)
                file_path = combine_datasets(non_cot_filepath, cot_filepath)
            else:
                file_path = preprocess_dataset(**kwargs)

            file_paths[f"{subset}_{split}"] = file_path
            if args.multirule:
                print_stats(file_path, torchtune=args.torchtune)
    
    if args.extra_examples:
        assert "easy_train" in file_paths and "easy_validation" in file_paths, "Didn't get the expected subset of easy and the train + val splits. Check your --subsets and --splits args."
        train_file_path = file_paths["easy_train"]
        val_file_path = file_paths["easy_validation"]
        
        train_dataset = datasets.load_dataset("json", data_files={"placeholder": train_file_path})["placeholder"]
        val_dataset = datasets.load_dataset("json", data_files={"placeholder": val_file_path})["placeholder"]
        
        combined_len = len(train_dataset) + len(val_dataset)
        combined_file_path = train_file_path.replace(f"_{len(train_dataset)}", f"_{combined_len}")
        combined_dataset = datasets.concatenate_datasets([train_dataset, val_dataset])
        combined_dataset = combined_dataset.shuffle(seed=42)
        combined_dataset.to_json(combined_file_path)
        print(f"Combined easy train and validation datasets saved to {combined_file_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--train_size", default=10000, type=int)
    parser.add_argument("--subsets", type=list, default=["easy"])
    parser.add_argument("--splits", type=list, default=["train", "validation", "test"])
    # parser.add_argument("--subsets", type=list, default=["multi_rule"])
    # parser.add_argument("--splits", type=list, default=["train", "test"])
    parser.add_argument("--extra_examples", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--multirule", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--add_cot", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_explanations", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--torchtune", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--torchtune_root", default="../torchtune", type=str)
    parser.add_argument("--combine", default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)