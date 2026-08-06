# Generate top-5 differential diagnoses (n=196) using Claude Opus 4.5
import json
import pandas as pd
from tqdm import tqdm
import anthropic
from dotenv import load_dotenv
import datetime

# Load API key from .env file
load_dotenv()

# Import the vignette dataset
dataset_path = "../../../../../../datasets/combined/medical_literature_only.json"
dataset_name = str(dataset_path).split("/")[-1].split(".")[0]  # Extract the dataset name
with open(dataset_path, "r") as f:
    combined = json.load(f)
dataset = pd.DataFrame(combined)  # Convert to DataFrame

# Define system instructions and user prompt
with open("../../../../../prompts/top_5_accuracy/system_prompt.txt") as f:
    system_prompt = f.read()

with open("../../../../../prompts/top_5_accuracy/user_prompt.txt") as f:
    user_prompt = f.read()


# Generate differential diagnosis for one case using Claude Opus 4.5
def generate_top5_diagnoses(client, model, system_prompt, user_prompt, vignette):
    # Prepare API call parameters
    # Anthropic Claude: Make API call and create response object
    if model.startswith("claude"):
        response = client.messages.create(
            model=model,
            max_tokens=20000,  # Max output for Claude Opus 4.5 is 64k but >20k requires streaming
            system=system_prompt,
            # Extended thinking mode is not compatible with temperature, top_p, or top_k sampling
            thinking={
                "type": "enabled",
                "budget_tokens": 19000  # Allocate tokens for thinking - model may not use entire budget
            },
            messages=[
                {
                    "role": "user", 
                    "content": user_prompt + "\n<vignette>\n" + vignette + "\n</vignette>"
                }
            ]
        )

        # Handle model refusal to answer
        if response.stop_reason == "refusal":
            reasoning = "N/A"
            answer = "Model refused to answer the prompt."
            return reasoning, answer

        # Extract the response content
        for block in response.content:
            if block.type == "text":  # Extract differential diagnosis block
                answer = block.text
            elif block.type == "thinking":  # Extract summarized thinking block
                reasoning = block.thinking
            elif block.type == "redacted_thinking":  # Handle redacted thinking block
                print(f"Redacted thinking detected for \"{vignette[:30]}...\"")
                reasoning = block.thinking

    return reasoning, answer


# Initialize Anthropic client
anthropic_client = anthropic.Anthropic()
model = "claude-opus-4-5-20251101"

print("***********************************************")
print(f"Processing model {model} on dataset {dataset_name}...")

# Process each case sequentially
pbar = tqdm(dataset.iterrows(), total=dataset.shape[0])  # Progress bar for tracking

for index, row in pbar:
    pbar.set_description(f"Generating differential diagnoses for case {index + 1} out of {dataset.shape[0]} (case {row['case_id']})")
    reasoning, answer = generate_top5_diagnoses(anthropic_client,
                                                model,
                                                system_prompt,
                                                user_prompt,
                                                row["vignette"],
                                                # Temperature not compatible with extended thinking mode
                                                )
    dataset.loc[index, "model_thoughts"] = reasoning
    dataset.loc[index, "model_diagnosis"] = answer
    print(f"Completed case {index + 1} out of {dataset.shape[0]} (case {row['case_id']}).")

# Save to a JSON file
output_path = f"../../../../../../results/top_5_accuracy/predicted_diagnoses/predicted_diagnoses_{model}_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
dataset.to_json(output_path, orient="records", indent=2)
print("***********************************************")
print(f"{model} predicted diagnoses for calculation of top-5 accuracy saved to JSON.")
