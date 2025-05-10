import os
import json

### IMPORTANT ###
# This file is meant to be run using the same working directory as generate_submission_preds.sh

def remove_metadata_and_relations_from_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Remove metadata and relations for each entry
            for record in data.values():
                record.pop("metadata", None)
                record.pop("relations", None)

            # Overwrite the file with the cleaned data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=True, indent=2)

if __name__ == "__main__":
    remove_metadata_and_relations_from_json_files("final_predictions/")