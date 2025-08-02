import os
import pandas as pd
import json

def convert_csv_to_json():
    for file in os.listdir():
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            
            # Identify the "*_answer_vanilla" column dynamically
            answer_col = [col for col in df.columns if col.endswith("_answer_vanilla")]
            if answer_col:
                answer_col = answer_col[0]  # Take the first matching column
            else:
                print(f"Skipping {file}: No column matching '*_answer_vanilla' found.")
                continue
            
            # Convert to JSON format
            json_data = df.to_dict(orient="records")
            
            # Save JSON file with the same base name
            json_file = file.replace(".csv", ".json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            
            print(f"Converted {file} to {json_file}")

if __name__ == "__main__":
    convert_csv_to_json()
