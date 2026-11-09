import json

# Suppose your JSON file is named data.json
with open("data.json", "r") as f:
    data = json.load(f)

for item in data:
    response_text = item["response"]
    # First, remove "Final Output:" if present
    response_text = response_text.replace("Final Output:", "").strip()
    
    # Next, if you want to remove everything before the numbered list (e.g., "1."),
    # find the index of "1." and keep everything from there onward
    index = response_text.find("1.")
    if index != -1:
        response_text = response_text[index:]
    
    # Store the cleaned response back
    item["response"] = response_text.strip()

# Save the modified data to a new file
with open("data_cleaned.json", "w") as f:
    json.dump(data, f, indent=4)
