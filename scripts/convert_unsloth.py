import argparse
import os
import random
import datasets
from tqdm import tqdm
import spacy

from constants import (
    NUM_TOKENS_METADATA,
    NUM_TURNS_METADATA,
    UNSLOTH_INPUT_FIELD,
    UNSLOTH_OUTPUT_FIELD,
    TORCHTUNE_INPUT_FIELD,
    TORCHTUNE_OUTPUT_FIELD,
    NUM_RULES_METADATA,
)
from helpers import combine_datasets, configure_logging, get_cleaned_fields, get_multirule_input, get_multirule_output, get_singlerule_examples, print_stats, get_token_count

import logging
logger = logging.getLogger(__name__)

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
    subset="compliance", 
    split="train", 
    size=-1, 
    data_dir="data", 
    add_cot=False, 
    add_explanations=False,
    rules_first=False, 
    torchtune=False, 
    torchtune_root=None,
    count_tokens=False,
):
    if torchtune:
        input_field = TORCHTUNE_INPUT_FIELD
        output_field = TORCHTUNE_OUTPUT_FIELD
    else:
        input_field = UNSLOTH_INPUT_FIELD
        output_field = UNSLOTH_OUTPUT_FIELD
    # Check if dataset_path is a file that exists locally
    if os.path.exists(dataset_path):
        dataset = datasets.load_dataset("json", data_files={"_": dataset_path})["_"]
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)

    logger.notice(f"Examples in {subset} {split}: {len(dataset)}. Processing {size if size != -1 else len(dataset)} examples.")
    if count_tokens:
        logger.notice("Counting tokens set to true and this is very slow for large datasets. Will take an hour to process 10k examples.")
        
    if size !=-1 and len(dataset) > size:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(size))

    if add_cot:
        logger.notice(f"Using Spacy to extract the first 4 sencences from reach rule discussion for COT. Should take {len(dataset)//1000+1} minutes to process {len(dataset)} examples...")
        # Setup nlp processor
        spacy_model = "en_core_web_sm"
        if not spacy.util.is_package(spacy_model):
            spacy.cli.download(spacy_model)
        nlp_processor = spacy.load(spacy_model)
    else:
        nlp_processor = None


    examples = []
    for row_index, row in tqdm(enumerate(dataset), total=len(dataset)):
        rules, labels, explanations, discussions, dialogue, dialogue_turns, num_rules, num_labeled_turns = get_cleaned_fields(row, row_index)
        if args.multirule:
            example = {}

            # Shuffle all of the lists (rules, labels, explanations) so there is no bias in the rule order
            zipped = list(zip(rules, labels, explanations, discussions))
            random.shuffle(zipped)
            rules, labels, explanations, discussions = zip(*zipped)

            example[input_field] = get_multirule_input(rules, dialogue)
            example[output_field] = get_multirule_output(
                labels=labels,
                explanations=explanations,
                discussions=discussions,
                dialogue_turns=dialogue_turns,
                num_rules=num_rules,
                num_labeled_turns=num_labeled_turns,
                add_cot=add_cot,
                add_explanations=add_explanations,
                nlp_processor=nlp_processor,
                rules_first=rules_first,
            )

            example[NUM_RULES_METADATA] = num_rules
            example[NUM_TURNS_METADATA] = len(dialogue_turns)
            if count_tokens:
                example[NUM_TOKENS_METADATA] = get_token_count(example[input_field])
            new_examples = [example]
        else:
            new_examples = get_singlerule_examples(
                rules,
                labels,
                explanations,
                discussions,
                dialogue_turns,
                num_rules,
                num_labeled_turns,
                input_field,
                output_field,
            )
        examples.extend(new_examples)
    processed_dataset = datasets.Dataset.from_list(examples)

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
    logger.notice(f"Saved to {file_path}")
    return file_path


def main(args):
    configure_logging(args.log_level)
    subsets = args.subsets
    splits = args.splits
    # file_paths = {}
    kwargs = {
        "dataset_path": args.dataset,
        "size": args.train_size,
        "data_dir": args.data_dir,
        "add_cot": args.add_cot,
        "add_explanations": args.add_explanations,
        "rules_first": args.rules_first,
        "torchtune": args.torchtune,
        "torchtune_root": args.torchtune_root,
        "count_tokens": args.count_tokens,
    }
    if subsets == []:
        file_path = preprocess_dataset(**kwargs)
        print_stats(file_path, torchtune=args.torchtune)
    else:
        for subset in subsets:
            assert splits != [], "If you are using subsets, you must also specify one or more splits."
            for split in splits:
                kwargs["subset"] = subset
                kwargs["split"] = split
                file_path = preprocess_dataset(**kwargs)
                print_stats(file_path, torchtune=args.torchtune)


            # if args.combine:
            #     kwargs["add_cot"] = False
            #     non_cot_filepath = preprocess_dataset(**kwargs)
            #     kwargs["add_cot"] = True
            #     cot_filepath = preprocess_dataset(**kwargs)
            #     file_path = combine_datasets(non_cot_filepath, cot_filepath)

            # file_paths[f"{subset}_{split}"] = file_path
    # if args.extra_examples:
    #     assert "easy_train" in file_paths and "easy_validation" in file_paths, "Didn't get the expected subset of easy and the train + val splits. Check your --subsets and --splits args."
    #     train_file_path = file_paths["easy_train"]
    #     val_file_path = file_paths["easy_validation"]
        
    #     train_dataset = datasets.load_dataset("json", data_files={"placeholder": train_file_path})["placeholder"]
    #     val_dataset = datasets.load_dataset("json", data_files={"placeholder": val_file_path})["placeholder"]
        
    #     combined_len = len(train_dataset) + len(val_dataset)
    #     combined_file_path = train_file_path.replace(f"_{len(train_dataset)}", f"_{combined_len}")
    #     combined_dataset = datasets.concatenate_datasets([train_dataset, val_dataset])
    #     combined_dataset = combined_dataset.shuffle(seed=42)
    #     combined_dataset.to_json(combined_file_path)
    #     print(f"Combined easy train and validation datasets saved to {combined_file_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", default=None, type=str, help="Log level for the script Default is NOTICE (between INFO and DEBUG).")
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--train_size", default=-1, type=int)
    # parser.add_argument("--dataset", default="tomg-group-umd/compliance", type=str, help="Dataset name or path to local dataset")
    parser.add_argument("--dataset", default="/Users/monte/code/system-prompt-compliance/output/multi_rule/train_combined.jsonl", type=str, help="Dataset name or path to local dataset")
    parser.add_argument("--subsets", default=[], type=list, help="List of subsets to process.")
    parser.add_argument("--splits", default=[], type=list, help="List of splits to process.")
    # parser.add_argument("--subsets", type=list, default=["easy"])
    # parser.add_argument("--splits", type=list, default=["train", "validation", "test"])
    # parser.add_argument("--subsets", type=list, default=["multi_rule"])
    # parser.add_argument("--splits", type=list, default=["train", "test"])
    # parser.add_argument("--extra_examples", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--multirule", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--add_cot", default=False, action=argparse.BooleanOptionalAction, help="Add COTs to the beginning of output in the dataset")
    parser.add_argument("--add_explanations", default=False, action=argparse.BooleanOptionalAction, help="Skip the explanation section at the end of the output")
    parser.add_argument("--rules_first", default=False, action=argparse.BooleanOptionalAction, help="Put the rules first in the output")
    parser.add_argument("--torchtune", default=False, action=argparse.BooleanOptionalAction, help="Use the torchtune format for expected column names in the dataset.")
    parser.add_argument("--torchtune_root", default="../torchtune", type=str, help="Path to the torchtune repo for saving the dataset. Only used if --torchtune is set.")
    parser.add_argument("--count_tokens", default=False, action=argparse.BooleanOptionalAction, help="Add a token count to the dataset.")
    # parser.add_argument("--combine", default=False, action=argparse.BooleanOptionalAction, help="Combine the COT and non-COT datasets")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)