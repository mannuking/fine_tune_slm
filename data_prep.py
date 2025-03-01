import json

input_json_path = "output.json"
output_text_path = "fine_tune_slm/training_data.txt"

# Load JSON data
with open(input_json_path, "r") as f:
    json_data = json.load(f)

# Extract text from each element and join with newlines
training_text = "\n".join([element["text"] for element in json_data])

# Save to text file
with open(output_text_path, "w", encoding="utf-8") as f:
    f.write(training_text)

print(f"Training data prepared and saved to {output_text_path}")
