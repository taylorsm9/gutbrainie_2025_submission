import argparse
import json
import pickle

def filter_entities_by_threshold(ner_predictions, label_thresholds):
    filtered_predictions = {}

    # Iterate through each sample in the predictions
    for sample_id, sample_data in ner_predictions.items():
        # Create a copy of the sample data
        filtered_sample = sample_data.copy()

        # Filter out entities with scores below their respective thresholds
        filtered_entities = []
        for entity in sample_data.get('pred_entities', []):
            # Get entity label (lowercase for case-insensitive matching)
            entity_label = entity['entity_label'].lower()

            # Check if the score is above the threshold for this entity type
            threshold = label_thresholds.get(entity_label)
            if threshold is None:  # Handle case where entity_label doesn't match our dictionary keys
                # Try matching with DDF which might be lowercase in predictions
                if entity_label == 'ddf':
                    threshold = label_thresholds['DDF']
                else:
                    # Skip entities with unknown labels
                    continue

            # Keep entity only if its score is above or equal to the threshold
            if entity['score'] >= threshold:
                filtered_entities.append(entity)

        # Update the sample with filtered entities
        filtered_sample['pred_entities'] = filtered_entities
        filtered_predictions[sample_id] = filtered_sample

    return filtered_predictions


def merge_consecutive_predictions(data):
    print("Merging consecutive NER predictions...")

    for pmid, doc in data.items():
        merged_entities = []
        current_entity = None

        full_text = doc.get("title", "") + " " + doc.get("abstract", "")

        for entity in doc.get("pred_entities", []):
            if current_entity is None:
                current_entity = entity
            else:
                same_label = current_entity["entity_label"] == entity["entity_label"]

                adjacent = (
                    current_entity["end_idx"] + 1 == entity["start_idx"] or
                    current_entity["end_idx"] == entity["start_idx"]
                )

                if same_label and adjacent:
                    # Use characters from source instead of hardcoding space
                    join_text = full_text[current_entity["end_idx"] : entity["start_idx"]]
                    current_entity["end_idx"] = entity["end_idx"]
                    current_entity["text_span"] += join_text + entity["text_span"]
                    current_entity["score"] = min(current_entity["score"], entity["score"])
                    continue

                merged_entities.append(current_entity)
                current_entity = entity

        if current_entity is not None:
            merged_entities.append(current_entity)

        doc["pred_entities"] = merged_entities


def adjust_predicted_indices(data):
    """
    Adjust the indices of predicted entities in the abstract by subtracting the length of the title
    from both the start and end indices, and decreasing the end index by 1.
    Returns a new dictionary with adjusted indices.
    """
    print("Adjusting indices for NER predictions...")

    # Create a deep copy of the data to avoid modifying the original
    result = {}

    # Process each document
    for pmid, doc in data.items():
        # Create a copy of the document
        new_doc = doc.copy()
        title_length = len(doc.get("title", ""))  # Calculate the length of the title

        # Create a new list for adjusted entities
        adjusted_entities = []

        for entity in doc.get("pred_entities", []):
            # Create a copy of the entity
            new_entity = entity.copy()

            new_entity["end_idx"] -= 1  # Adjust the end index to be exclusive

            # Fix lower case DDF
            if new_entity["entity_label"] == "ddf":
                new_entity["entity_label"] = "DDF"

            if new_entity["tag"] == "a":  # Process only entities from the abstract
                new_entity["start_idx"] -= title_length + 1
                new_entity["end_idx"] -= title_length + 1

            adjusted_entities.append(new_entity)

        # Set the adjusted entities in the new document
        new_doc["pred_entities"] = adjusted_entities
        result[pmid] = new_doc

    return result

def migrate_to_ground_truth_format(articles):
    return_dict = {}

    for pmid, article in articles.items():
        return_dict[pmid] = {}
        return_dict[pmid]['metadata'] = {}
        return_dict[pmid]['entities'] = []
        return_dict[pmid]['relations'] = []

        return_dict[pmid]['metadata']['title'] = article['title']
        return_dict[pmid]['metadata']['author'] = article['author']
        return_dict[pmid]['metadata']['journal'] = article['journal']
        return_dict[pmid]['metadata']['year'] = article['year']
        return_dict[pmid]['metadata']['abstract'] = article['abstract']
        return_dict[pmid]['metadata']['annotator'] = 'distant'

        for entity in article['pred_entities']:
            ent_dict = {
                "start_idx": entity['start_idx'],
                "end_idx": entity['end_idx'],
                "location": 'title' if entity['tag'] == 't' else 'abstract',
                "text_span": entity['text_span'],
                "label": entity['entity_label']
            }
            return_dict[pmid]['entities'].append(ent_dict)

    return return_dict

if __name__ == "__main__":
    # Args for input/output
    parser = argparse.ArgumentParser(description='Process NER predictions with thresholds')
    parser.add_argument('--preds', help='Path to raw predictions JSON file')
    parser.add_argument('--output', help='Path to output JSON file')
    parser.add_argument('--thresholds', help='Path to pickled dictionary of per-class thresholds')

    args = parser.parse_args()


    # Load dict of learned thresholds
    with open(args.thresholds, 'rb') as f:
        thresholds = pickle.load(f)

    # Load raw gliner preds
    with open(args.preds, 'r', encoding='utf-8') as f:
        ner_predictions = json.load(f)

    # Remove preds under per class thresholds
    ner_predictions = filter_entities_by_threshold(ner_predictions, thresholds)

    # Combine preds + fix indices for evaluation
    merge_consecutive_predictions(ner_predictions)
    ner_predictions = adjust_predicted_indices(ner_predictions)

    # Final migration to eval format
    predictions = migrate_to_ground_truth_format(ner_predictions)

    # Write to output_path
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=True)
