import json
from collections import Counter
import re
import argparse

def adjust_entities_intervention_rule(text: str, entities: list[dict], location_flag: str, verbose: bool = False) -> Counter:
    """
    Adjust entity spans by extending any entity to include "intervention" or "interventions"
    that follow them.

    Args:
        text: The source text containing the entities
        entities: List of entity dictionaries
        location_flag: Filter for entities with this location value
        verbose: Whether to print debug information

    Returns:
        Counter with statistics about the adjustments
    """
    # Filter entities to those matching the location flag (any label class)
    filtered_entities = [e for e in entities if e["location"] == location_flag]

    if verbose:
        print(f"Found {len(filtered_entities)} entities in {location_flag}")

    stats = Counter({"total_entities": len(filtered_entities), "entities_extended": 0})

    # Track changes for logging
    changes_log = []

    for entity in filtered_entities:
        if verbose:
            print(f"\n--- Processing entity: '{entity['text_span']}' ({entity['start_idx']}, {entity['end_idx']}) ---")

        # Get the text following this entity
        end_pos = entity["end_idx"] + 1

        # Check if there's enough text after the entity
        if end_pos < len(text):
            # Get the next 20 characters (or fewer if at the end of text)
            following_text = text[end_pos:min(end_pos + 20, len(text))]

            if verbose:
                print(f"Text after entity: '{following_text}'")

            # Check if the next word is "intervention" or "interventions"
            # First, we check if following_text starts with a space
            if following_text.startswith(' '):
                if verbose:
                    print(f"Found space after entity")

                # Then check if the word after the space is "intervention" or "interventions"
                rest_text = following_text[1:].lstrip()
                if verbose:
                    print(f"Text after space (trimmed): '{rest_text}'")

                # Use regex to match "intervention" or "interventions" followed by punctuation or space or end of string
                intervention_match = re.match(r'(intervention|interventions)([.,;:!?\s]|$)', rest_text)

                if intervention_match:
                    intervention_word = intervention_match.group(1)  # The matched word (intervention or interventions)
                    intervention_len = len(intervention_word)

                    if verbose:
                        print(f"Matched '{intervention_word}' followed by '{intervention_match.group(2) if intervention_match.group(2) else 'end of text'}'")

                    # Calculate the new end position - need to add 1 for the space before "intervention" and 1 for inclusivity
                    space_offset = following_text.find(intervention_word) - 1  # -1 to account for the space we already found
                    new_end_idx = end_pos + space_offset + intervention_len + 1

                    # Create the new entity text
                    new_text_span = text[entity["start_idx"]:new_end_idx]

                    if verbose:
                        print(f"CHANGE: Entity '{entity['text_span']}' → '{new_text_span}'")

                    # Store change info for logging
                    change_info = {
                        "original": entity.copy(),
                        "original_text": entity["text_span"],
                        "extended_text": new_text_span,
                        "intervention_word": intervention_word
                    }
                    changes_log.append(change_info)

                    # Update the entity
                    entity["end_idx"] = new_end_idx - 1  # -1 because end_idx is inclusive
                    entity["text_span"] = new_text_span

                    stats["entities_extended"] += 1
                elif verbose:
                    print(f"No 'intervention' pattern found after space")
            elif verbose:
                print(f"No space after entity")
        elif verbose:
            print(f"Entity is at the end of text")

    # Print detailed change log
    if verbose and changes_log:
        print("\n=== DETAILED CHANGE LOG ===")
        for idx, change in enumerate(changes_log, 1):
            print(f"{idx}. EXTENDED: '{change['original_text']}' → '{change['extended_text']}'")
            print(f"   Added: '{change['intervention_word']}'")
            print(f"   Original indices: ({change['original']['start_idx']}:{change['original']['end_idx']})")
            print()

    # Print summary
    if verbose:
        print(f"\nSummary for {location_flag}:")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Entities extended with 'intervention(s)': {stats['entities_extended']}")

    return stats

def process_pmid_with_intervention_rule(record: dict, pmid: str, verbose: bool = False) -> Counter:
    """Process a single PMID record and apply the intervention rule."""
    if verbose:
        print(f"\n==== Processing PMID: {pmid} ====")

    # Validate record structure
    if not isinstance(record, dict):
        print(f"WARNING: Record for PMID {pmid} is not a dictionary")
        return Counter()

    # Get title and abstract from metadata
    metadata = record.get("metadata", {})
    title = metadata.get("title", "")
    abstract = metadata.get("abstract", "")

    if verbose:
        print(f"Title present: {'Yes' if title else 'No'}, length: {len(title)}")
        print(f"Abstract present: {'Yes' if abstract else 'No'}, length: {len(abstract)}")

    # Check entities
    entities = record.get("entities", [])
    if not entities:
        if verbose:
            print(f"No entities found for PMID {pmid}")
        return Counter()

    if verbose:
        print(f"Total entities in record: {len(entities)}")

    # Process text fields with our new function
    title_stats = adjust_entities_intervention_rule(title, entities, "title", verbose)
    abstract_stats = adjust_entities_intervention_rule(abstract, entities, "abstract", verbose)

    combined_stats = title_stats + abstract_stats

    if verbose:
        print(f"\nResults for PMID {pmid}:")
        print(f"  Total entities: {combined_stats['total_entities']}")
        print(f"  Entities extended: {combined_stats['entities_extended']}")

    return combined_stats

def process_json_intervention_rule(data: dict, verbose: bool = False):
    """
    Process JSON data to adjust entity spans by adding "intervention" or "interventions".

    Args:
        data: Dictionary containing the JSON data
        verbose: Whether to print detailed processing information

    Returns:
        The processed data dictionary
    """
    print(f"Processing {len(data)} records")

    # Process the data
    grand_stats = Counter()
    pmids_with_changes = 0

    for pmid, record in data.items():
        pmid_stats = process_pmid_with_intervention_rule(record, pmid, verbose)
        grand_stats += pmid_stats

        if pmid_stats["entities_extended"] > 0:
            pmids_with_changes += 1

    # Print summary
    print("\n===== PROCESSING SUMMARY =====")
    print(f"PMIDs with changes: {pmids_with_changes} out of {len(data)}")
    print(f"Total entities: {grand_stats['total_entities']}")
    print(f"Entities extended with 'intervention(s)': {grand_stats['entities_extended']}")
    print(f"Successfully processed {len(data)} records")

    return data

def adjust_entities_treatment_rule(text: str, entities: list[dict], location_flag: str, verbose: bool = False) -> Counter:
    """
    Adjust entity spans by extending drug entities to include "treatment" or "treatments"
    that follow them.

    Args:
        text: The source text containing the entities
        entities: List of entity dictionaries
        location_flag: Filter for entities with this location value
        verbose: Whether to print debug information

    Returns:
        Counter with statistics about the adjustments
    """
    # Filter entities to those matching the location flag and with label "drug"
    drug_entities = [e for e in entities if e["location"] == location_flag and e["label"] == "drug"]

    if verbose:
        print(f"Found {len(drug_entities)} drug entities in {location_flag}")

    stats = Counter({"total_drugs": len(drug_entities), "drugs_extended": 0})

    # Track changes for logging
    changes_log = []

    for entity in drug_entities:
        if verbose:
            print(f"\n--- Processing entity: '{entity['text_span']}' ({entity['start_idx']}, {entity['end_idx']}) ---")

        # Get the text following this entity
        end_pos = entity["end_idx"] + 1

        # Check if there's enough text after the entity
        if end_pos < len(text):
            # Get the next 20 characters (or fewer if at the end of text)
            following_text = text[end_pos:min(end_pos + 20, len(text))]

            if verbose:
                print(f"Text after entity: '{following_text}'")

            # Check if the next word is "treatment" or "treatments"
            # First, we check if following_text starts with a space
            if following_text.startswith(' '):
                if verbose:
                    print(f"Found space after entity")

                # Then check if the word after the space is "treatment" or "treatments"
                rest_text = following_text[1:].lstrip()
                if verbose:
                    print(f"Text after space (trimmed): '{rest_text}'")

                # Use regex to match "treatment" or "treatments" followed by punctuation or space or end of string
                treatment_match = re.match(r'(treatment|treatments)([.,;:!?\s]|$)', rest_text)

                if treatment_match:
                    treatment_word = treatment_match.group(1)  # The matched word (treatment or treatments)
                    treatment_len = len(treatment_word)

                    if verbose:
                        print(f"Matched '{treatment_word}' followed by '{treatment_match.group(2) if treatment_match.group(2) else 'end of text'}'")

                    # Calculate the new end position - need to add 1 for the space before "treatment" and 1 for inclusivity
                    space_offset = following_text.find(treatment_word) - 1  # -1 to account for the space we already found
                    new_end_idx = end_pos + space_offset + treatment_len + 1

                    # Create the new entity text
                    new_text_span = text[entity["start_idx"]:new_end_idx]

                    if verbose:
                        print(f"CHANGE: Drug entity '{entity['text_span']}' → '{new_text_span}'")

                    # Store change info for logging
                    change_info = {
                        "original": entity.copy(),
                        "original_text": entity["text_span"],
                        "extended_text": new_text_span,
                        "treatment_word": treatment_word
                    }
                    changes_log.append(change_info)

                    # Update the entity
                    entity["end_idx"] = new_end_idx - 1  # -1 because end_idx is inclusive
                    entity["text_span"] = new_text_span

                    stats["drugs_extended"] += 1
                elif verbose:
                    print(f"No 'treatment' pattern found after space")
            elif verbose:
                print(f"No space after entity")
        elif verbose:
            print(f"Entity is at the end of text")

    # Print detailed change log
    if verbose and changes_log:
        print("\n=== DETAILED CHANGE LOG ===")
        for idx, change in enumerate(changes_log, 1):
            print(f"{idx}. EXTENDED: '{change['original_text']}' → '{change['extended_text']}'")
            print(f"   Added: '{change['treatment_word']}'")
            print(f"   Original indices: ({change['original']['start_idx']}:{change['original']['end_idx']})")
            print()

    # Print summary
    if verbose:
        print(f"\nSummary for {location_flag}:")
        print(f"  Total drug entities: {stats['total_drugs']}")
        print(f"  Drugs extended with 'treatment(s)': {stats['drugs_extended']}")

    return stats

def process_pmid_with_treatment_rule(record: dict, pmid: str, verbose: bool = False) -> Counter:
    """Process a single PMID record and apply the treatment rule."""
    if verbose:
        print(f"\n==== Processing PMID: {pmid} ====")

    # Validate record structure
    if not isinstance(record, dict):
        print(f"WARNING: Record for PMID {pmid} is not a dictionary")
        return Counter()

    # Get title and abstract from metadata
    metadata = record.get("metadata", {})
    title = metadata.get("title", "")
    abstract = metadata.get("abstract", "")

    if verbose:
        print(f"Title present: {'Yes' if title else 'No'}, length: {len(title)}")
        print(f"Abstract present: {'Yes' if abstract else 'No'}, length: {len(abstract)}")

    # Check entities
    entities = record.get("entities", [])
    if not entities:
        if verbose:
            print(f"No entities found for PMID {pmid}")
        return Counter()

    if verbose:
        print(f"Total entities in record: {len(entities)}")

    # Process text fields with our new function
    title_stats = adjust_entities_treatment_rule(title, entities, "title", verbose)
    abstract_stats = adjust_entities_treatment_rule(abstract, entities, "abstract", verbose)

    combined_stats = title_stats + abstract_stats

    if verbose:
        print(f"\nResults for PMID {pmid}:")
        print(f"  Total drug entities: {combined_stats['total_drugs']}")

    return combined_stats

def process_json_treatment_rule(data: dict, verbose: bool = False):
    """
    Process JSON data to adjust drug entity spans by adding "treatment" or "treatments".

    Args:
        data: Dictionary containing the JSON data
        verbose: Whether to print detailed processing information

    Returns:
        The processed data dictionary
    """
    print(f"Processing {len(data)} records")

    # Process the data
    grand_stats = Counter()
    pmids_with_changes = 0

    for pmid, record in data.items():
        pmid_stats = process_pmid_with_treatment_rule(record, pmid, verbose)
        grand_stats += pmid_stats

        if pmid_stats["drugs_extended"] > 0:
            pmids_with_changes += 1

    # Print summary
    print("\n===== PROCESSING SUMMARY =====")
    print(f"PMIDs with changes: {pmids_with_changes} out of {len(data)}")
    print(f"Total drug entities: {grand_stats['total_drugs']}")
    print(f"Drugs extended with 'treatment(s)': {grand_stats['drugs_extended']}")
    print(f"Successfully processed {len(data)} records")

    return data

if __name__ == "__main__":
    # Args for input/output
    parser = argparse.ArgumentParser(description='Process NER predictions with rules-based postprocessing')
    parser.add_argument('--preds', default="ensemble_preds_replace_microbiome.json",
                        help='Path to raw predictions JSON file')
    parser.add_argument('--output', default="ensemble_preds_with_rules.json",
                        help='Path to output JSON file')

    args = parser.parse_args()

    # Load evaluation format preds
    with open(args.preds, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply [drug] + treatment and [any] + intervention rules
    data = process_json_treatment_rule(data)
    data = process_json_intervention_rule(data)

    # Write to output path
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
