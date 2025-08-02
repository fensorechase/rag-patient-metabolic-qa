import os
import json

def deduplicate_json(file_path):
    """Deduplicate evaluation entries based on unique question_text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    if "evaluations" not in full_data:
        print(f"Skipping {file_path}: No 'evaluations' key found.")
        return None

    seen_questions = set()
    deduped_evaluations = []

    for entry in full_data["evaluations"]:
        question = entry.get("question_text", "").strip()
        if question and question not in seen_questions:
            seen_questions.add(question)
            deduped_evaluations.append(entry)

    # Replace evaluations with deduplicated version
    full_data["evaluations"] = deduped_evaluations
    full_data["metadata"]["num_samples"] = len(deduped_evaluations)

    return full_data

def process_json_files():
    """Find all JSON files in the current directory and deduplicate them."""
    for file_name in os.listdir():
        if file_name.endswith(".json") and not file_name.startswith("dedup_"):
            deduped_data = deduplicate_json(file_name)
            if deduped_data is not None:
                output_file = f"dedup_{file_name}"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(deduped_data, f, indent=4)
                print(f"Processed: {file_name} -> {output_file}")

if __name__ == "__main__":
    process_json_files()
