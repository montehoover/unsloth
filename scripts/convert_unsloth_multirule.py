import argparse
import ast
import json
import datasets
from tqdm import tqdm

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
}
"""

# These constants should match the constants at the top of main.py
# TODO: Move these constants to a shared file
INPUT_FIELD = "question"
OUTPUT_LABEL_FIELD = "answer"
OUTPUT_RULES_FIELD = "rules_violated"
OUTPUT_VIOLATION_LINES_FIELD = "violation_lines"
OUTPUT_EXPLANATIONS_FIELD = "explanations"

COT_OPENING = "\n<reasoning>"
COT_CLOSING = "\n</reasoning>"
LABEL_OPENING = "\n<answer>"
LABEL_CLOSING = "\n</answer>"


class ComplianceProjectError(ValueError):
    pass

def clean_rule(rule):
    # Looking for 1. or 2. etc.
    splits = rule.split(". ", 1)
    if len(splits) > 1:
        rule = splits[1].strip()
    else:
        rule = rule.strip()
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

        # labels = row["labels"]
        # print(labels)
        # print(type(labels))
        # print(type(labels[0]))
        # print(len(labels))
        # print(len(labels[0]))

        cleaned_rules = []
        cleaned_labels = []
        cleaned_explanations = []
        for i in range(len(rules)):
            cleaned_rules.append(clean_rule(rules[i]))
            cleaned_labels.append(parse_string_list(labels[i]))
            cleaned_explanations.append([clean_explanation(explanation) for explanation in parse_string_list(explanations[i])])

        delimiters = ["'User':", """"User":"""]
        for delimiter in delimiters:
            if delimiter in dialogue:
                dialogue_preamble = dialogue.split(delimiter, 1)[0]
                main_dialogue = dialogue.split(delimiter, 1)[1]
                dialogue_turns = [f"{delimiter}{item}" for item in main_dialogue.split(delimiter) if item]
                dialogue_turns[0] = dialogue_preamble + dialogue_turns[0]
                break

        num_rules = len(cleaned_rules)
        num_turns = len(dialogue_turns)
        if num_turns != len(cleaned_labels[0]):
            raise ComplianceProjectError(f"Example {row_index}: Number of dialogue turns ({len(dialogue_turns)}) does not match number of turns in labels ({len(cleaned_labels[0])}) \nDialogue: {json.dumps(dialogue_turns, indent=4)} \nLabels: {cleaned_labels[0]}")
        
        ##################
        # Input
        ##################
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
        examples.append(example)

    processed_dataset = datasets.Dataset.from_list(examples)
    processed_dataset = processed_dataset.shuffle(seed=42)

    if size is None or len(processed_dataset) < size:
        size = len(processed_dataset)
    processed_dataset = processed_dataset.select(range(size))

    file_path = f"{data_dir}/{subset}_{split}_{size}.jsonl"

    processed_dataset.to_json(file_path)
    print(f"Examples in processed dataset: {size}")
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