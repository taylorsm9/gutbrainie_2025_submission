import json
import argparse

def extract_class_predictions(data_dict: dict, class_label: str) -> dict:
    """
    Extracts all entities of a given class from the input dict.

    Args:
        data_dict: Original dictionary with predictions per PMID.
        class_label: The label of entities to retain.

    Returns:
        A new dictionary with only the specified class's entities.
    """
    result = {}
    for pmid, record in data_dict.items():
        matching_entities = [e for e in record.get("entities", []) if e.get("label") == class_label]
        if matching_entities:
            result[pmid] = {
                **record,
                "entities": matching_entities
            }
    return result

def spans_overlap(span1: dict, span2: dict) -> bool:
    """Check if two spans overlap within the same location (title/abstract)."""
    if span1["location"] != span2["location"]:
        return False
    return not (span1["end_idx"] <= span2["start_idx"] or span2["end_idx"] <= span1["start_idx"])

def remove_overlapping_preds(base_dict: dict, interfering_entities: dict) -> dict:
    """
    Removes any entities from base_dict that overlap with the provided interfering_entities.

    Args:
        base_dict: Original prediction set (lower-precision model).
        interfering_entities: A dict with PMIDs and the high-precision entities to protect.

    Returns:
        A modified base_dict with overlapping entities removed.
    """
    for pmid, record in base_dict.items():
        base_entities = record.get("entities", [])
        interference = interfering_entities.get(pmid, {}).get("entities", [])

        if not interference:
            continue

        filtered_entities = [
            e for e in base_entities
            if not any(spans_overlap(e, ie) for ie in interference)
        ]
        record["entities"] = filtered_entities

    return base_dict

def add_extracted_predictions(base_dict: dict, extracted_dict: dict) -> dict:
    """
    Adds extracted predictions back into the base_dict.

    Args:
        base_dict: The cleaned base prediction set.
        extracted_dict: The high-precision entities to add.

    Returns:
        The updated base_dict with new predictions inserted.
    """
    for pmid, extracted_record in extracted_dict.items():
        base_record = base_dict.setdefault(pmid, {
            "metadata": extracted_record.get("metadata", {}),
            "entities": [],
            "relations": []
        })

        base_record["entities"].extend(extracted_record.get("entities", []))

    return base_dict

def overwrite_predictions_for_class(base_dict: dict, class_dict: dict, class_label: str) -> dict:
    """
    For a given label, replaces overlapping base_dict predictions with class_dict ones.

    Args:
        base_dict: Lower-precision prediction dict.
        class_dict: Higher-precision prediction dict.
        class_label: The entity label to overwrite (e.g., "gene").

    Returns:
        A new prediction dict with class_label predictions replaced.
    """
    # Step 1: Extract target class from the high-precision model
    extracted = extract_class_predictions(class_dict, class_label)

    # Step 2: Remove all overlapping predictions from the base model
    cleaned = remove_overlapping_preds(base_dict.copy(), extracted)

    # Step 3: Insert high-precision predictions
    final = add_extracted_predictions(cleaned, extracted)

    return final

def remove_class_predictions(data_dict: dict, class_label: str) -> dict:
    """
    Removes all entities of a given class from the input dict.

    Args:
        data_dict: Dictionary with predictions per PMID.
        class_label: The label of entities to remove.

    Returns:
        The modified input dictionary (with specified class removed).
    """
    for record in data_dict.values():
        record["entities"] = [e for e in record.get("entities", []) if e.get("label") != class_label]
    return data_dict

def merge_prediction_dicts(base_dict: dict, class_dict: dict) -> dict:
    """
    Merges entities from class_dict into base_dict. Entities are appended per PMID.

    Args:
        base_dict: Base dictionary of predictions.
        class_dict: Dictionary containing additional predictions to merge.

    Returns:
        A new dictionary with combined predictions.
    """
    merged = {pmid: record.copy() for pmid, record in base_dict.items()}

    for pmid, record in class_dict.items():
        if pmid not in merged:
            merged[pmid] = record.copy()
        else:
            merged[pmid]["entities"].extend(record.get("entities", []))

    return merged

if __name__ == "__main__":
    # Args for input/output
    parser = argparse.ArgumentParser(description='Process NER predictions with thresholds')
    parser.add_argument('--recall_preds', help='Path to eval format preds JSON file')
    parser.add_argument('--precision_preds', help='Path to raw predictions JSON file')
    parser.add_argument('--model_3_preds', help='Path to raw predictions JSON file')
    parser.add_argument('--output', help='Path to output JSON file')

    args = parser.parse_args()

    # Load raw gliner preds
    with open(args.recall_preds, 'r', encoding='utf-8') as f:
        base_preds = json.load(f)
    with open(args.precision_preds, 'r', encoding='utf-8') as f:
        precision_preds = json.load(f)
    with open(args.model_3_preds, 'r', encoding='utf-8') as f:
        model_3_preds = json.load(f)

    # Remove all preds of replace_label class from base preds and replace with higher f1 model_3 preds
    replace_labels = ["anatomical location", "animal", "human"]
    for label in replace_labels:
        # Collect replace_label preds from higher micro f1 preds
        class_preds = extract_class_predictions(model_3_preds, label)
        # Remove all preds of replace_label class from base preds and replace with higher f1 preds
        base_preds = remove_class_predictions(base_preds, label)
        base_preds = remove_overlapping_preds(base_preds, class_preds)
        base_preds = merge_prediction_dicts(base_preds, class_preds)

    # Overwrite all existing spans with spans from higher precision model for classes in list
    overwrite_list = ["DDF", "biomedical technique", "dietary supplement",]
    for label in overwrite_list:
        base_preds = overwrite_predictions_for_class(base_preds, precision_preds, label)

    # Remove all preds of replace_label class from base preds and replace with higher f1 class preds
    replace_labels = ["microbiome", "statistical technique"]
    for label in replace_labels:
        # Collect replace_label preds from higher micro f1 preds
        class_preds = extract_class_predictions(precision_preds, label)
        # Remove all preds of replace_label class from base preds and replace with higher f1 preds
        base_preds = remove_class_predictions(base_preds, label)
        base_preds = merge_prediction_dicts(base_preds, class_preds)

    # Write to output path
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(base_preds, f, indent=2, ensure_ascii=True)
