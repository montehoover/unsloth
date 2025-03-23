import argparse
import ast
import datasets
import json


# These constants should match the constants at the top of main.py
# TODO: Move these constants to a shared file
INPUT_FIELD = "question"
OUTPUT_FIELD = "answer"

# These constants should match with the system prompt in the config file and with the GRPO constants in Unsloth
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
        dataset = datasets.load_dataset('json', data_files=dataset_path)['train']
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    print(f"Examples in {subset} {split}: {len(dataset)}")

    examples = []
    for row_index, row in enumerate(dataset):
                ##################
        # Setup
        ##################
        example ={}
        rules = row["rules"]
        dialogue = row["dialogue"]
        labels = row["labels"]
        explanations = row["explanations"]
        discussions = row["discussions"]

        cleaned_rules = []
        cleaned_labels = []
        cleaned_explanations = []
        cleaned_discussions = []
        for i in range(len(rules)):
            cleaned_rules.append(clean_rule(rules[i]))
            cleaned_labels.append(parse_string_list(labels[i]))
            cleaned_explanations.append([clean_explanation(explanation) for explanation in parse_string_list(explanations[i])])
            cleaned_discussions.append([clean_explanation(discussion) for discussion in parse_string_list(discussions[i])])

        num_rules = len(cleaned_rules)
        num_turns = len(cleaned_labels[0])
        dialogue_turns = get_dialogue_turns(dialogue, expected_turns=num_turns, example_index=row_index)

        ##################
        # Input
        ##################
        for i, rule in enumerate(rules):
            for j in range(len(dialogue_turns)):
                example = {}
                dialogue_subset = "".join(dialogue_turns[:j+1])
                # Construct input
                try:
                    rule = clean_rule(rule)
                except Exception as e:
                    print(f"BAD RULE in example {row_index}: {rule}")
                    raise e
                example[INPUT_FIELD] = f'''
Rule Agent must follow:
{rule}

Conversation:
{dialogue_subset}
'''
                # Construct ouput
                try:
                    if subset == "hard" and split == "test":
                        discussion = parse_string_list_problem(discussions[i])[j]
                    else:
                        discussion = parse_string_list(discussions[i])[j] # Starts with "Turn x: "
                    discussion = clean_explanation(discussion)
                except Exception as e:
                    print(f"BAD DISCUSSION in example {row_index}: {discussions}")
                    raise e
                
                try:
                    if subset == "hard" and split == "test":
                        explanation = parse_string_list_problem(explanations[i])[j]
                    else:
                        explanation = parse_string_list(explanations[i])[j] # Starts with "Turn x: "
                    explanation = clean_explanation(explanation)
                except Exception as e:
                    print(f"BAD EXPLANATION in example {row_index}: {explanations}")
                    raise e
                
                label = parse_string_list(labels[i])[j]
                example[OUTPUT_FIELD] = f"{COT_OPENING} {discussion} {explanation} {COT_CLOSING} {LABEL_OPENING} {label} {LABEL_CLOSING}"
                examples.append(example)

    torchtune_dataset = datasets.Dataset.from_list(examples)
    torchtune_dataset = torchtune_dataset.shuffle(seed=42)

    if size is None or len(torchtune_dataset) < size:
        size = len(torchtune_dataset)
    torchtune_dataset = torchtune_dataset.select(range(size))

    file_path = f"{data_dir}/{subset}_{split}_{size}.jsonl"

    torchtune_dataset.to_json(file_path, orient='records', lines=True, indent=None)
    print(f"Examples in dataset preprocessed for TorchTune: {size}")
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
    parser.add_argument("--extra_examples", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
