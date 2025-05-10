import json
from gliner import GLiNER
from tqdm import tqdm
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GLiNER prediction script")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the checkpoint to use for prediction"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input articles JSON file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the predicted NER output"
    )

    args = parser.parse_args()

    # Path to checkpoint to use for prediction
    checkpoint_path = args.checkpoint_path

    # Define the path to articles for which the final trained will generate predicted entities
    PATH_ARTICLES = args.input_path
    PATH_OUTPUT_NER_PREDICTIONS = args.output_path

    # Define the confidence threshold to be used in evaluation
    THRESHOLD = 0.01

    # Legal entities for GliNER
    eval_data = {
        "entity_types": [
            "anatomical location",
            "animal",
            "biomedical technique",
            "bacteria",
            "chemical",
            "dietary supplement",
            "ddf",
            "drug",
            "food",
            "gene",
            "human",
            "microbiome",
            "statistical technique",
        ]
    }

    print("CWD")
    print(os.getcwd())

    # Load model from checkpoint
    md = GLiNER.from_pretrained(checkpoint_path)

    print(f"## GENERATING NER PREDICTIONS FOR {PATH_ARTICLES}")
    with open(PATH_ARTICLES, 'r', encoding='utf-8') as file:
        articles = json.load(file)

    print(f"len(articles): {len(articles)}")
    entity_labels = eval_data['entity_types']

    # Dictionary to hold predicted entities
    # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'entity_label': ..., 'score': ...}, ...}
    predictions = {}

    for pmid, content in tqdm(articles.items(), total=len(articles), desc="Predicting entities..."):
        title = content['title']
        abstract = content['abstract']

        # Predict entities
        title_entities = md.predict_entities(title, entity_labels, threshold=THRESHOLD, flat_ner=True, multi_label=False)
        abstract_entities = md.predict_entities(abstract, entity_labels, threshold=THRESHOLD, flat_ner=True, multi_label=False)

        # Adjust indices for predicted entities in the abstract
        for entity in abstract_entities:
            entity['start'] += len(title) + 1
            entity['end'] += len(title) + 1

        # Remove duplicates from predicted entities
        unique_entities = []
        seen_entities = set()

        # Remove duplicates from title entities and add tag field with value 't'
        for entity in title_entities:
            key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
            if key not in seen_entities:
                tmp_entity = {
                    'start_idx': entity['start'],
                    'end_idx': entity['end'],
                    'tag': 't',
                    'text_span': entity['text'],
                    'entity_label': entity['label'],
                    'score': entity['score']
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        # Remove duplicates from abstract entities and add tag field with value 'a'
        for entity in abstract_entities:
            key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
            if key not in seen_entities:
                tmp_entity = {
                    'start_idx': entity['start'],
                    'end_idx': entity['end'],
                    'tag': 'a',
                    'text_span': entity['text'],
                    'entity_label': entity['label'],
                    'score': entity['score']
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        predictions[pmid] = unique_entities
        articles[pmid]['pred_entities'] = unique_entities

    # Convert any non-serializable data if necessary
    def default_serializer(obj):
        if isinstance(obj, set):
            return list(obj)
        # Add other types if needed
        raise TypeError(f'Type {type(obj)} not serializable')

    with open(PATH_OUTPUT_NER_PREDICTIONS, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"## Predictions have been exported in JSON format to '/{PATH_OUTPUT_NER_PREDICTIONS}' ##")

    