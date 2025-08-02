import os
import json

def deduplicate_json(file_path):
    """Deduplicate JSON entries based on unique_question_id."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    seen_ids = set()
    deduped_data = []
    
    for entry in data:
        unique_id = entry.get("unique_question_id")
        if unique_id and unique_id not in seen_ids:
            seen_ids.add(unique_id)
            deduped_data.append(entry)
    
    return deduped_data

def process_json_files():
    """Find all JSON files in the current directory and process them."""
    for file_name in os.listdir():
        if file_name.endswith(".json") and not file_name.startswith("dedup_"):
            deduped_data = deduplicate_json(file_name)
            output_file = f"dedup_{file_name}"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(deduped_data, f, indent=4)
            
            print(f"Processed: {file_name} -> {output_file}")

if __name__ == "__main__":
    process_json_files()
