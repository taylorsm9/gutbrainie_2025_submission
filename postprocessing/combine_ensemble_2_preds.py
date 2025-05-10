from combine_ensemble_1_preds import *
import json
import argparse

if __name__ == "__main__":
    # Args for input/output
    parser = argparse.ArgumentParser(description='Process NER predictions with thresholds')
    parser.add_argument('--model_4_preds', help='Path to eval format preds JSON file')
    parser.add_argument('--model_5_preds', help='Path to eval format preds JSON file')
    parser.add_argument('--output', help='Path to output JSON file')

    args = parser.parse_args()

    # Load raw gliner preds
    with open(args.model_4_preds, 'r', encoding='utf-8') as f:
        model_4_preds = json.load(f)
    with open(args.model_5_preds, 'r', encoding='utf-8') as f:
        model_5_preds = json.load(f)

    # Remove all preds of replace_label class from base preds and replace with higher f1 model_3 preds
    replace_labels = ["anatomical location", "animal", "bacteria", "biomedical technique", "statistical technique"]
    for label in replace_labels:
        # Collect replace_label preds from higher micro f1 preds
        class_preds = extract_class_predictions(model_5_preds, label)
        # Remove all preds of replace_label class from base preds and replace with higher f1 preds
        model_4_preds = remove_class_predictions(model_4_preds, label)
        model_4_preds = remove_overlapping_preds(model_4_preds, class_preds)
        model_4_preds = merge_prediction_dicts(model_4_preds, class_preds)


    # Write to output path
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(model_4_preds, f, indent=2, ensure_ascii=True)