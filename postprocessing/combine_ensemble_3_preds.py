from combine_ensemble_1_preds import *
import json
import argparse

if __name__ == "__main__":
    # Args for input/output
    parser = argparse.ArgumentParser(description='Process NER predictions with thresholds')
    parser.add_argument('--ensemble_1_preds', help='Path to eval format preds JSON file')
    parser.add_argument('--ensemble_2_preds', help='Path to eval format preds JSON file')
    parser.add_argument('--output', help='Path to output JSON file')

    args = parser.parse_args()

    # Load preds
    with open(args.ensemble_1_preds, 'r', encoding='utf-8') as f:
        ensemble_1_preds = json.load(f)
    with open(args.ensemble_2_preds, 'r', encoding='utf-8') as f:
        ensemble_2_preds = json.load(f)

    # Remove all preds of replace_label class from base preds and replace with higher f1 model_3 preds
    replace_labels = ["anatomical location", "animal", "drug"]
    for label in replace_labels:
        # Collect replace_label preds from higher micro f1 preds
        class_preds = extract_class_predictions(ensemble_1_preds, label)

        # Remove all preds of replace_label class from base preds and replace with higher f1 preds
        ensemble_2_preds = remove_class_predictions(ensemble_2_preds, label)
        ensemble_2_preds = remove_overlapping_preds(ensemble_2_preds, class_preds)
        ensemble_2_preds = merge_prediction_dicts(ensemble_2_preds, class_preds)


    # Write to output path
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(ensemble_2_preds, f, indent=2, ensure_ascii=True)