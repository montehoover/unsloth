import argparse
import ast
import json
import random
import datasets
from tqdm import tqdm
import re

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
    input: str,
    allpass_label: str,
    rules_violated: List[int],
    violation_lines: List[str],
    explanations: List[str]
    num_rules: int,
}
"""

# These constants should match the constants at the top of main.py
# TODO: Move these constants to a shared file
INPUT_FIELD = "question"
OUTPUT_LABEL_FIELD = "answer"
OUTPUT_RULES_FIELD = "rules_violated"
OUTPUT_VIOLATION_LINES_FIELD = "violation_lines"
OUTPUT_EXPLANATIONS_FIELD = "explanations"
NUM_RULES_METADATA = "num_rules"

COT_OPENING = "\n<reasoning>"
COT_CLOSING = "\n</reasoning>"
LABEL_OPENING = "\n<answer>"
LABEL_CLOSING = "\n</answer>"


class ComplianceProjectError(ValueError):
    pass

def print_stats(dataset_path, local=True, obj=False):
    if obj:
        dataset = dataset_path
    elif local:
        dataset = datasets.load_dataset("json", data_files={"_": dataset_path}, split="_")
    else:
        dataset = datasets.load_dataset(dataset_path)
    min_rules = float("inf")
    max_rules = 0
    num_pass = 0
    num_fail = 0
    total_rules = 0
    for i, example in enumerate(dataset):
        if example[OUTPUT_LABEL_FIELD] == "PASS":
            num_pass += 1
        elif example[OUTPUT_LABEL_FIELD] == "FAIL":
            num_fail += 1
        else:
            raise ComplianceProjectError(f"Invalid label for example {i}: {example[OUTPUT_LABEL_FIELD]}")
        if example[NUM_RULES_METADATA] < min_rules:
            min_rules = example[NUM_RULES_METADATA]
        if example[NUM_RULES_METADATA] > max_rules:
            max_rules = example[NUM_RULES_METADATA]
        total_rules += example[NUM_RULES_METADATA]
    mean_rules = total_rules / len(dataset)
    pass_rate = num_pass / len(dataset)
    print(f"""Number of examples: {len(dataset)}
Number of PASS examples: {num_pass}
Number of FAIL examples: {num_fail}
Pass rate: {pass_rate:.1%}
Min rules: {min_rules}
Max rules: {max_rules}
Mean rules: {mean_rules:.1f}
""")


def clean_rule(rule):
    # Use regex to remove any whitespace followed by a number, a period, and a space at the beginning of the string
    rule = re.sub(r"^\s*\d+\.\s", "", rule).strip()
    return rule

def clean_explanation(explanation):
    # Looking for "Turn x: "
    explanation = explanation.split(": ", 1)[1].strip()
    return explanation

def parse_string_list_problem(string_list):
    # Format: "1. ['Turn 1. ...', 'Turn 2. ...', ...]\n"
    string_list = string_list.split(". ", 1)[1].strip()
    turn_count = string_list.count("Turn ")
    native_list = ast.literal_eval(string_list)
    if len(native_list) > turn_count:
        native_list = string_list.split("Turn ")[1:]
    return native_list

def parse_string_list(string_list):
    # Format: "1. ['PASS', 'PASS', 'PASS']\n"
    string_list = string_list.split(". ", 1)[1].strip()
    native_list = ast.literal_eval(string_list)
    return native_list

def get_dialogue_turns(dialogue, expected_turns, example_index=-1):
    delimiters = ["'User':", """"User":"""]
    dialogue_turns = []
    for delimiter in delimiters:
        if delimiter in dialogue:
            dialogue_preamble = dialogue.split(delimiter, 1)[0]
            main_dialogue = dialogue.split(delimiter, 1)[1]
            dialogue_turns = [f"{delimiter}{item}" for item in main_dialogue.split(delimiter) if item]
            dialogue_turns[0] = dialogue_preamble + dialogue_turns[0]
            break
    if len(dialogue_turns) != expected_turns:
        raise ComplianceProjectError(f"""
            Example {example_index}: Number of dialogue turns ({len(dialogue_turns)}) does not match number of turns in labels ({expected_turns}).
            Delimiters: {delimiters}
            Dialogue: {json.dumps(dialogue_turns, indent=4)}
            """)
    return dialogue_turns

def preprocess_dataset(dataset_path, subset=None, split=None, size=None, local=False, data_dir="data"):
    if local:
        dataset = datasets.load_dataset("json", data_files={"placeholder": dataset_path})["placeholder"]
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    print(f"Examples in {subset} {split}: {len(dataset)}")

    examples = []
    for row_index, row in tqdm(enumerate(dataset)):
        ##################
        # Setup
        ##################
        example ={}
        rules = row["rules"]
        dialogue = row["dialogue"]
        labels = row["labels"]
        explanations = row["explanations"]

        cleaned_rules = []
        cleaned_labels = []
        cleaned_explanations = []
        for i in range(len(rules)):
            cleaned_rules.append(clean_rule(rules[i]))
            cleaned_labels.append(parse_string_list(labels[i]))
            cleaned_explanations.append([clean_explanation(explanation) for explanation in parse_string_list(explanations[i])])

        num_rules = len(cleaned_rules)
        num_turns = len(cleaned_labels[0])
        dialogue_turns = get_dialogue_turns(dialogue, expected_turns=num_turns, example_index=row_index)
        
        ##################
        # Input
        ##################
        # Shuffle all of the lists (rules, labels, explanations) so there is no bias in the rule order
        zipped = list(zip(cleaned_rules, cleaned_labels, cleaned_explanations))
        random.shuffle(zipped)
        cleaned_rules, cleaned_labels, cleaned_explanations = zip(*zipped)

        enumerated_rules = '\n'.join(f"{i+1}. {rule}" for i, rule in enumerate(cleaned_rules))
        example[INPUT_FIELD] = f'''
Rules agent must follow:
{enumerated_rules}

Transcript:
{dialogue}
'''     
        ##################
        # Output        
        ##################
        allpass_label = "PASS"
        violated_rules = []
        violation_lines = []
        violation_explanations = []

        for i in range(num_rules):
            for j in range(num_turns):
                if cleaned_labels[i][j] == "FAIL":
                    allpass_label = "FAIL"
                    violated_rules.append(i+1)
                    violation_lines.append(dialogue_turns[j])
                    violation_explanations.append(cleaned_explanations[i][j])
                    break # We capture the first violation of a given rule and then move to the next rule
        
        example[OUTPUT_LABEL_FIELD] = allpass_label
        example[OUTPUT_RULES_FIELD] = violated_rules
        example[OUTPUT_VIOLATION_LINES_FIELD] = violation_lines
        example[OUTPUT_EXPLANATIONS_FIELD] = violation_explanations
        example[NUM_RULES_METADATA] = num_rules
        examples.append(example)

    processed_dataset = datasets.Dataset.from_list(examples)

    if size is not None and len(processed_dataset) > size:
        processed_dataset = processed_dataset.shuffle(seed=42)
        processed_dataset = processed_dataset.select(range(size))

    file_path = f"{data_dir}/{subset}_{split}_{len(processed_dataset)}.jsonl"

    processed_dataset.to_json(file_path)
    print_stats(file_path)
    print(f"Saved to {file_path}")
    return file_path


def main(args):
    # Local:
    # preprocess_dataset("test.jsonl", local=True)

    huggingface_dataset = "tomg-group-umd/compliance"
    # Subset choices are "easy" or "hard"
    # Easy: 9007 train, 1793 val, 67 test
    # Hard: 1670 train, 313 val, 44 test
    subsets = ["easy", "hard"]
    splits = ["train", "validation", "test"]
    file_paths = {}
    for subset in subsets:
        for split in splits:
            file_path = preprocess_dataset(huggingface_dataset, subset, split, size=args.train_size, data_dir=args.data_dir)
            file_paths[f"{subset}_{split}"] = file_path

    if args.extra_examples:
        train_file_path = file_paths["easy_train"]
        val_file_path = file_paths["easy_validation"]
        
        train_dataset = datasets.load_dataset("json", data_files={"_": train_file_path}, split="_")
        val_dataset = datasets.load_dataset("json", data_files={"_": val_file_path}, split="_")
        
        combined_len = len(train_dataset) + len(val_dataset)
        combined_file_path = f"{args.data_dir}/easy_train_{combined_len}.jsonl"
        
        combined_dataset = datasets.concatenate_datasets([train_dataset, val_dataset])
        combined_dataset = combined_dataset.shuffle(seed=42)
        combined_dataset.to_json(combined_file_path)
        print(f"Combined easy train and validation datasets saved to {combined_file_path}")
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/multi_rule", type=str)
    parser.add_argument("--train_size", default=10000, type=int)
    parser.add_argument("--extra_examples", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)