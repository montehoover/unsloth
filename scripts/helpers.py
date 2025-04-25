import ast
import json
import os
import random
import re
import uuid
import datasets
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from constants import (
    COT_CLOSING,
    COT_OPENING,
    EXPLANATION_CLOSING,
    EXPLANATION_OPENING,
    METADATA,
    TORCHTUNE_INPUT_FIELD,
    TORCHTUNE_OUTPUT_FIELD,
    UNSLOTH_INPUT_FIELD,
    UNSLOTH_OUTPUT_FIELD,
    LINE_CLOSING,
    LINE_OPENING,
    NUM_RULES_METADATA,
    LABEL_OPENING,
    LABEL_CLOSING,
    RULE_NUMBER_CLOSING,
    RULE_NUMBER_OPENING,
    RULES_OPENING,
    RULES_CLOSING,
    LABEL_CLOSING,
    LABEL_OPENING,
    RULE_START,
    CONVERSATION_START,
)
import logging
import numpy as np
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)


class ComplianceProjectError(ValueError):
    pass

def extract_rules_violated(text):
    rules = text.split(RULES_OPENING)[-1]
    rules = rules.split(RULES_CLOSING)[0]
    rules = rules.split(",")
    rules_violated = []
    for rule in rules:
        if rule.strip().isdigit():
            rules_violated.append(int(rule))
    return rules_violated

def update_rule_violations(ground_truth_text, output_text, rule_violations):
    # There could be rules it missed, and there could be extra rules that it thought were violated but weren't, so we report those.
    ground_truth_rules = extract_rules_violated(ground_truth_text)
    predicted_rules = extract_rules_violated(output_text)
    num_missed = len(set(ground_truth_rules) - set(predicted_rules))
    num_extra = len(set(predicted_rules) - set(ground_truth_rules))
    if num_missed > 0 or num_extra > 0:
        logger.debug(f"Missed rules: {num_missed}, Extra rules: {num_extra}")
    rule_violations["missed"] += num_missed
    rule_violations["extra"] += num_extra

def extract_xml_answer(text, opening_tag, closing_tag):
    answer = text.split(opening_tag.strip())[-1]
    answer = answer.split(closing_tag.strip())[0]
    return answer.strip()

def filter_nulls(ground_truth_labels, predicted_labels):
    nulls = []
    for i, label in enumerate(predicted_labels):
        if label not in ["PASS", "FAIL"]:
            nulls.append(i)
            # Guarantee that we get the wrong answer if we don't have a prediction.
            predicted_labels[i] = "FAIL" if ground_truth_labels[i] == "PASS" else "PASS"
    return predicted_labels, nulls

def get_stats(outputs, dataset, multirule=False):
    ground_truth_labels = []
    predicted_labels = []
    false_negatives = []  
    false_positives = []  # The transcript was fine but we flagged a violation.
    rule_violations = {"missed": 0, "extra": 0}
    for i, (example, output_text) in enumerate(zip(dataset, outputs)):
        ground_truth_text = example[UNSLOTH_OUTPUT_FIELD]
        ground_truth_label = extract_xml_answer(ground_truth_text, LABEL_OPENING, LABEL_CLOSING)
        predicted_label = extract_xml_answer(output_text, LABEL_OPENING, LABEL_CLOSING)
        
        # Thing for GuardReasoner
        # if "PASS" in output_text:
        #     predicted_label = "PASS"
        # else:
        #     predicted_label = "FAIL"

        ground_truth_labels.append(ground_truth_label)
        predicted_labels.append(predicted_label)
        
        if multirule:
            # When it gets it right that some rules were violated, check to see if it marked the right rules.
            if predicted_label == "FAIL" and ground_truth_label == "FAIL":
                update_rule_violations(ground_truth_label, output_text, rule_violations)
            
        if predicted_label == "PASS" and ground_truth_label == "FAIL":
            false_negatives.append(i)
        if predicted_label == "FAIL" and ground_truth_label == "PASS":
            false_positives.append(i)

    percent_pass = ground_truth_labels.count("PASS") / len(ground_truth_labels)
    predicted_labels, nulls = filter_nulls(ground_truth_labels, predicted_labels)
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    try:
        f1 = f1_score(ground_truth_labels, predicted_labels, pos_label="FAIL")
    except ValueError as e:
        if "Target is multiclass" in str(e):
            raise ComplianceProjectError(f""""
                Something unexpected happened with the labels.
                If ground_truth_labels are not all PASS/FAIL, then there was a mismatch between the dataset and expected xml tags.
                If predicted_labels are not all PASS/FAIL, then something went wrong in filter_nulls().
                Multi-rule eval: {multirule}
                Expected xml tags: {LABEL_OPENING} {LABEL_CLOSING}
                ground_truth_labels: {ground_truth_labels}
                predicted_labels: {predicted_labels}
                """) from None
    stats = {
        "accuracy": accuracy,
        "f1_score": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "nulls": nulls,
        "missed_rules": rule_violations["missed"],
        "extra_rules": rule_violations["extra"],
        "percent_pass": percent_pass,
    }
    return stats

def confirm_model_compatibility(model_name, use_llamaguard):
    if use_llamaguard and "Llama-Guard" not in model_name:
        raise ComplianceProjectError(f"Selected Llama-Guard evaluation but gave a non-Llama-Guard model: {model_name}")
    if not use_llamaguard and "Llama-Guard" in model_name:
        raise ComplianceProjectError(f"Gave a Llama-Guard model but didn't select llamaguard mode with --llamaguard: {model_name}")

def confirm_dataset_compatibility(dataset, use_multirule):
    output_text = dataset[0][UNSLOTH_OUTPUT_FIELD]
    if use_multirule:
        required_tag = LABEL_OPENING
    else:
        required_tag = LABEL_OPENING
    if required_tag not in output_text:
        type = "multi-rule" if use_multirule else "single-rule"
        raise ComplianceProjectError(
            f"Selected {type} evaluation but gave a dataset which does not include the expected label tag of {required_tag} needed for this type of evaluation."
            "\nTry looking for a dataset with 'multi_rule' in the path if you wanted multi-rule."
            )

def apply_llamaguard_template(system_content, user_content):
    assert "rule" in system_content and "conversation" in system_content, f"Expected a llamaguard system template but got {system_content}"
    
    rule_idx = user_content.find(RULE_START)
    conversation_idx = user_content.find(CONVERSATION_START)

    rule = user_content[rule_idx + len(RULE_START):conversation_idx].strip()
    conversation = user_content[conversation_idx + len(CONVERSATION_START):].strip()
    conversation = conversation.replace('\'User\'', 'User').replace('\'Agent\'', 'Agent')

    return system_content.format(rule=rule, conversation=conversation)

def map_llamaguard_output(output):
    if 'unsafe' in output:
        return '<answer>FAIL</answer>' 
    elif 'safe' in output:
        return '<answer>PASS</answer>'
    else:
        return 'null'

def get_cot(discussions, explanations, nlp_processor, num_sentences=4):
    # There is a discussion for every rule, and within that a discussion for every turn. Get only the discussion from the last turn for the COT.
    last_turn_discussions = [turn_discussions[-1] for turn_discussions in discussions]
    last_turn_explanations = [explanations[-1] for explanations in explanations]

    short_discussions = []
    # This whole thing is slow so we're trying to speed it up with the pipeline version of Spacy's nlp processor
    nlp_pipeline = nlp_processor.pipe(last_turn_discussions, disable=["ner", "tagger"])
    for processed_discussion in nlp_pipeline:
        sentences = [sent.text.strip() for sent in processed_discussion.sents]
        first_few_sentences = sentences[:num_sentences]
        short_discussion = ' '.join(first_few_sentences)
        short_discussions.append(short_discussion)

    # Combine the short discussions with the explanations into a single COT for each rule
    cot_by_rule = [f"{short_discussion} {explanation}" for short_discussion, explanation in zip(short_discussions, last_turn_explanations)]
    enumerated_cot = '\n'.join(f"Rule {i+1}. {cot}" for i, cot in enumerate(cot_by_rule))
    return enumerated_cot


def get_multirule_input(rules, dialogue):
    enumerated_rules = '\n'.join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
    input = f"{RULE_START}\n{enumerated_rules}\n\n{CONVERSATION_START}\n{dialogue}"
    return input

def get_multirule_output(
        labels, 
        explanations, 
        discussions, 
        dialogue_turns, 
        num_rules, 
        num_labeled_turns, 
        add_cot=False,
        add_explanations=False, 
        nlp_processor=None, 
        num_sentences=4,
        rules_first=False,
    ):
    if add_cot and add_explanations:
        raise ComplianceProjectError("Cannot set both add_cot and add_explanations to True. Choose one method for displaying reasoning.")
    
    # Initialize variables for output
    allpass_label = "PASS"
    violated_rules = []
    # violation_lines = []
    violation_explanations = []

    logger.info(f"Checking for rule violations in {num_rules} rules and {num_labeled_turns} turns...")
    for i in range(num_rules):
        for j in range(num_labeled_turns):
            if labels[i][j] == "FAIL":
                allpass_label = "FAIL"
                violated_rules.append(i+1)
                # violation_lines.append(dialogue_turns[j])
                violation_explanations.append(explanations[i][j])
                break # We capture the first violation of a given rule and then move to the next rule
    
    # Formatting
    violated_rules = ",".join(map(str, violated_rules))
    violation_explanations = "\n".join(violation_explanations)
    if add_cot:
        logger.info(f"Using Spacy to extract the first {num_sentences} sentences from each rule discussion for COT.")
        cot = get_cot(discussions, explanations, nlp_processor, num_sentences=num_sentences)

    # Format in xml tags
    # Note that cot_block and explanation block use the same tags, but the explanations have shorter text inside
    cot_block = f"{COT_OPENING}\n{cot}\n{COT_CLOSING}\n" if add_cot else ""
    label_block = f"{LABEL_OPENING}\n{allpass_label}\n{LABEL_CLOSING}\n"
    rules_block = f"{RULES_OPENING}\n{violated_rules or "None"}\n{RULES_CLOSING}\n"
    explanation_block = f"{COT_OPENING}\n{violation_explanations}\n{COT_CLOSING}\n" if violated_rules and add_explanations else ""
    # Below is our older, more elaborate way of doing it:
    # explanation_block = ""
    # for i in range(len(violated_rules)):
    #     rule_number = violated_rules[i]
    #     line_in_transcript = violation_lines[i]
    #     explanation = violation_explanations[i]
    #     explanation_block += (
    #         f"{RULE_NUMBER_OPENING}\n{rule_number}\n{RULE_NUMBER_CLOSING}\n"
    #         f"{LINE_OPENING}\n{line_in_transcript}\n{LINE_CLOSING}\n"
    #         f"{EXPLANATION_OPENING}\n{explanation}\n{EXPLANATION_CLOSING}\n"
    #     )

    # The default looks something like this: 
    #[<reasoning>...</reasoning>]  <answer>...</answer>  <rules_violated>...</rules_violated>  [<reasoning>...</reasoning>]
    if rules_first:
        output = f"{cot_block}{rules_block}{label_block}{explanation_block}"
    else:
         output = f"{cot_block}{label_block}{rules_block}{explanation_block}"

    return output

def get_singlerule_examples(rules, labels, explanations, discussions, dialogue_turns, num_rules, num_turns, input_field, output_field):
    examples = []
    for i in range(num_rules):
        for j in range(num_turns):
            example = {}
            rule = rules[i]
            discussion = discussions[i][j]
            explanation = explanations[i][j]
            label = labels[i][j]
            dialogue_subset = "".join(dialogue_turns[:j+1])
            example[input_field] = f"{RULE_START}\n{rule}\n\n{CONVERSATION_START}\n{dialogue_subset}"
            example[output_field] = f"{COT_OPENING}{discussion} {explanation}{COT_CLOSING}{LABEL_OPENING}{label}{LABEL_CLOSING}"
            examples.append(example)
    return examples

def get_cleaned_fields(example, example_index):
        rules = example["rules"]
        dialogue = example["dialogue"]
        labels = example["labels"]
        explanations = example["explanations"]
        discussions = example["discussions"]
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
        num_labeled_turns = len(cleaned_labels[0])
        dialogue_turns = get_dialogue_turns(dialogue, num_labeled_turns=num_labeled_turns, example_index=example_index, strict=False)
        return cleaned_rules, cleaned_labels, cleaned_explanations, cleaned_discussions, dialogue, dialogue_turns, num_rules, num_labeled_turns

def get_token_count(text):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    tokens = tokenizer.encode(text)
    return len(tokens)

def print_stats(dataset_path, local=True, obj=False, torchtune=False):
    if obj:
        dataset = dataset_path
    elif local:
        dataset = datasets.load_dataset("json", data_files={"_": dataset_path}, split="_")
    else:
        dataset = datasets.load_dataset(dataset_path)
    # Stats for rules
    min_rules = float("inf")
    max_rules = 0
    total_rules = 0
    
    # Stats for turns
    min_turns = float("inf")
    max_turns = 0
    total_turns = 0
    
    # Stats for tokens
    min_tokens = float("inf")
    max_tokens = 0
    total_tokens = 0
    
    num_pass = 0
    num_fail = 0
    
    for i, example in enumerate(dataset):
        output_field = TORCHTUNE_OUTPUT_FIELD if torchtune else UNSLOTH_OUTPUT_FIELD
        label = extract_xml_answer(example[output_field], LABEL_OPENING, LABEL_CLOSING)
        if label == "PASS":
            num_pass += 1
        elif label == "FAIL":
            num_fail += 1
        else:
            raise ComplianceProjectError(f"Invalid label for example {i}: {example[output_field]}")
            
        # Rules stats
        if example[NUM_RULES_METADATA] < min_rules:
            min_rules = example[NUM_RULES_METADATA]
        if example[NUM_RULES_METADATA] > max_rules:
            max_rules = example[NUM_RULES_METADATA]
        total_rules += example[NUM_RULES_METADATA]
        
        # Turns stats
        if "num_turns" in example:
            if example["num_turns"] < min_turns:
                min_turns = example["num_turns"]
            if example["num_turns"] > max_turns:
                max_turns = example["num_turns"]
            total_turns += example["num_turns"]
        
        # Tokens stats
        if "num_tokens" in example:
            if example["num_tokens"] < min_tokens:
                min_tokens = example["num_tokens"]
            if example["num_tokens"] > max_tokens:
                max_tokens = example["num_tokens"]
            total_tokens += example["num_tokens"]
            
    mean_rules = total_rules / len(dataset)
    pass_rate = num_pass / len(dataset)
    
    print(f"""Number of examples: {len(dataset)}
Number of PASS examples: {num_pass}
Number of FAIL examples: {num_fail}
Pass rate: {pass_rate:.1%}
Min rules: {min_rules}
Max rules: {max_rules}
Mean rules: {mean_rules:.1f}""")
    
    # Print turns stats if available
    if total_turns > 0:
        mean_turns = total_turns / len(dataset)
        print(f"""Min turns: {min_turns}
Max turns: {max_turns}
Mean turns: {mean_turns:.1f}""")
    
    # Print tokens stats if available
    if total_tokens > 0:
        mean_tokens = total_tokens / len(dataset)
        print(f"""Min tokens: {min_tokens}
Max tokens: {max_tokens}
Mean tokens: {mean_tokens:.1f}
""")

def clean_rule(rule):
    # Use regex to remove any whitespace followed by a number, a period, and a space at the beginning of the string
    rule = re.sub(r"^\s*\d+\.\s", "", rule).strip()
    return rule

def clean_explanation(explanation):
    # Looking for "Turn x: "
    explanation = explanation.split(": ", 1)[1].strip()
    return explanation

def parse_string_list(string_list):
    # Format: "1. ['PASS', 'PASS', 'PASS']\n"
    string_list = string_list.split(". ", 1)[1].strip()
    native_list = ast.literal_eval(string_list)
    return native_list

def get_dialogue_turns(dialogue, num_labeled_turns, example_index=-1, strict=False):
    delimiters = ["'User':", """"User":"""]
    dialogue_turns = []
    for delimiter in delimiters:
        if delimiter in dialogue:
            dialogue_preamble = dialogue.split(delimiter, 1)[0]
            main_dialogue = dialogue.split(delimiter, 1)[1]
            dialogue_turns = [f"{delimiter}{item}" for item in main_dialogue.split(delimiter) if item]
            dialogue_turns[0] = dialogue_preamble + dialogue_turns[0]
            break
    if strict and len(dialogue_turns) != num_labeled_turns:
        raise ComplianceProjectError(f"""
            Example {example_index}: Number of dialogue turns ({len(dialogue_turns)}) does not match number of turns in labels ({num_labeled_turns}).
            Delimiters: {delimiters}
            Dialogue: {json.dumps(dialogue_turns, indent=4)}
            """)
    return dialogue_turns

def combine_datasets(non_cot_filepath, cot_filepath):
    non_cot_dataset = datasets.load_dataset("json", data_files={"_": non_cot_filepath}, split="_")
    cot_dataset = datasets.load_dataset("json", data_files={"_": cot_filepath}, split="_")
    combined_dataset = datasets.concatenate_datasets([non_cot_dataset, cot_dataset])
    combined_dataset = combined_dataset.shuffle(seed=42)
    orig_size = len(cot_dataset)
    new_size = len(combined_dataset)
    new_path = cot_filepath.replace(str(orig_size), str(new_size)).replace("_cot", "_combined")
    combined_dataset.to_json(new_path)
    print(f"Saved combined dataset to {new_path}")
    return new_path


def get_analysis(dataset, wrong_predictions):
    assert METADATA in dataset.column_names, f"Dataset {dataset} does not have {METADATA} field"
    counts = {}
    for i, example in enumerate(dataset):
        metadata = example[METADATA]
        if i in wrong_predictions:
            counts["business_impact_categories"] = counts.get("business_impact_categories", set()) | {metadata["business_impact"]}
            counts["failure_mode_categories"] = counts.get("failure_mode_categories", set()) | {metadata["failure_mode"]}
            counts[metadata["business_impact"]] = counts.get(metadata["business_impact"], 0) + 1
            counts[metadata["failure_mode"]] = counts.get(metadata["failure_mode"], 0) + 1
            if metadata["extra_failure_modes"] != "":
                counts["failure_mode_categories"] = counts.get("failure_mode_categories", set()) | {metadata["extra_failure_modes"]}
                counts[metadata["extra_failure_modes"]] = counts.get(metadata["extra_failure_modes"], 0) + 1
            if metadata["num_counts"] != -1:
                counts["num_counts"] = counts.get("num_counts", []) + [metadata["num_counts"]]
            if metadata["num_hops"] != -1:
                counts["num_hops"] = counts.get("num_counts", []) + [metadata["num_hops"]]
            if metadata["num_turns"] != -1:
                counts["num_turns"] = counts.get("num_counts", []) + [metadata["num_turns"]]
            if metadata["num_rules"] != -1:
                counts["num_rules"] = counts.get("num_counts", []) + [metadata["num_rules"]]
            if metadata["rule_len"] != -1:
                counts["rule_len"] = counts.get("num_counts", []) + [metadata["rule_len"]]
    for key in ["num_counts", "num_hops", "num_turns", "num_rules", "rule_len"]:
        if key in counts and isinstance(counts[key], list) and counts[key]:
            counts[f"{key}_median"] = np.median(counts[key])
    return counts

class JsonSetEncoder(json.JSONEncoder):
    """Allows json.dump to handle dictionaries that contain sets."""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def save_results(analysis_dict, output_root, output_path, model_name, total_accuracy, stdev, outputs):    
    # --- JSONs for generation outputs and analysis_dict ---
    datasets.Dataset.from_list([{"_": _} for _ in outputs]).to_json(f"{output_path}/outputs.jsonl")
    with open(f"{output_path}/analysis.json", "w") as f:
        json.dump(analysis_dict, f, indent=4, cls=JsonSetEncoder)

    # --- Matplotlib configuration ---
    mpl.rcParams.update({
        # 'font.family': 'serif',
        # 'font.serif': ['Times New Roman'],  # Academic standard font
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,       # Higher resolution for publication-quality images
        'savefig.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1,
        'grid.color': '0.8',     # Light gray grid lines for subtlety
        'grid.linestyle': '--'
    })
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    mpl.rc('text', usetex=True)
    
    # --- Pie Chart for Business Impact Categories ---
    if "business_impact_categories" in analysis_dict and analysis_dict["business_impact_categories"]:
        business_categories = list(analysis_dict["business_impact_categories"])
        business_counts = [analysis_dict.get(cat, 0) for cat in business_categories]

        fig, ax = plt.subplots()
        ax.pie(business_counts, labels=business_categories, autopct='%1.1f%%', startangle=90)
        ax.set_title("Distribution of Business Impact Categories")
        file_path = os.path.join(output_path, "business_impact_categories.png")
        plt.savefig(file_path)
        plt.close(fig)
    else:
        print("No business impact categories data available.")

    # --- Pie Chart for Failure Mode Categories ---
    if "failure_mode_categories" in analysis_dict and analysis_dict["failure_mode_categories"]:
        failure_categories = list(analysis_dict["failure_mode_categories"])
        failure_counts = [analysis_dict.get(cat, 0) for cat in failure_categories]
        fig, ax = plt.subplots()
        ax.pie(failure_counts, labels=failure_categories, autopct='%1.1f%%', startangle=90)
        ax.set_title("Distribution of Failure Mode Categories")
        file_path = os.path.join(output_path, "failure_mode_categories.png")
        plt.savefig(file_path)
        plt.close(fig)
    else:
        print("No failure mode categories data available.")

    # --- Histograms for Numerical Metrics ---
    numeric_keys = ['num_counts', 'num_hops', 'num_turns', 'num_rules', 'rule_len']
    for key in numeric_keys:
        if key in analysis_dict and isinstance(analysis_dict[key], list) and analysis_dict[key]:
            plt.figure()
            plt.hist(analysis_dict[key], bins=10, edgecolor='black')
            plt.title(f"Histogram of {key}")
            plt.xlabel(key)
            plt.ylabel("Frequency")
            file_path = os.path.join(output_path, f"{key}_histogram.png")
            plt.savefig(file_path)
            plt.close()  # Closes the current figure
        else:
            print(f"No numerical data available for {key}")

    # --- Update results CSV ---
    results = {}
    medians = {key.replace('_median', ''): value 
                     for key, value in analysis_dict.items() if key.endswith('_median')}
    results["total_accuracy"] = total_accuracy
    results["accuracy_std"] = stdev
    
    # If CSV exists, load it and append the new row. Otherwise, create a new DataFrame.
    csv_filename = os.path.join(output_root, "results.csv")
    new_row = pd.DataFrame([results], index=[model_name])
    new_row.index.name = "model_name"
    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename, index_col=0)
        df = pd.concat([existing_df, new_row], axis=0)
    else:
        df = new_row
    df.to_csv(csv_filename, index=True)
    
    # --- Create LaTeX-style table as a PNG using the updated CSV DataFrame ---
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns)), 0.5 * len(df) + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    png_filename = os.path.join(output_root, "results_table.png")
    plt.savefig(png_filename, bbox_inches='tight')
    plt.close(fig)
    
    # --- Generate LaTeX code for the updated table ---
    latex_str = df.to_latex(header=True, index=True,
                            caption="Median values from analysis",
                            label="tab:medians")
    latex_filename = os.path.join(output_root, "results.tex")
    with open(latex_filename, "w") as f:
        f.write(latex_str)

    # --- Bar Chart for results ---
    plt.figure(figsize=(8, 6))
    plt.bar(df.index, df['total_accuracy'],
        yerr=df['accuracy_std'],     # Using standard deviation as error bars
        capsize=5,                   # Add caps to the error bars
        color='#4D4D4D',             # Muted dark gray fill color
        edgecolor='black',           # Black outline for the bars
        linewidth=1.2)
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f"{output_root}/results_bar_chart.png", format='png')

def configure_logging(log_level=None, ext_level_bump=1):
    # Create a custom level that is between INFO and WARNING
    logging.addLevelName(25, "NOTICE")
    notice = lambda self, message, *args, **kwargs: self._log(25, message, args, **kwargs)
    logging.Logger.notice = notice

    # Determine log level: CLI argument > Environment variable > Default (NOTICE)
    log_level = (log_level or os.getenv("LOG_LEVEL", "NOTICE")).upper()
    logging.basicConfig(
        level=log_level,
        format="{name}:{levelname}: {message}",
        style="{"
    )