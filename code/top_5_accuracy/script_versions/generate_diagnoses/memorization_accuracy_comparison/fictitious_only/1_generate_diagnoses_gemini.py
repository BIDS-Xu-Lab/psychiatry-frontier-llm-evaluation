# Generate top-5 differential diagnoses (n=196) using Gemini 3 Pro
import json
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
import datetime

# Load API key from .env file
load_dotenv()

# Import the vignette dataset
dataset_path = "../../../../../../datasets/combined/fictitious_only.json"
dataset_name = str(dataset_path).split("/")[-1].split(".")[0]  # Extract the dataset name
with open(dataset_path, "r") as f:
    combined = json.load(f)
dataset = pd.DataFrame(combined)  # Convert to DataFrame

# Define system instructions and user prompt
with open("../../../../../prompts/top_5_accuracy/system_prompt.txt") as f:
    system_prompt = f.read()

with open("../../../../../prompts/top_5_accuracy/user_prompt.txt") as f:
    user_prompt = f.read()


# Generate differential diagnosis for one case using Gemini 3 Pro
def generate_top5_diagnoses(client, model, system_prompt, user_prompt, vignette, temperature):
    response = client.models.generate_content(
                model=model,
                contents=user_prompt + "\n<vignette>\n" + vignette + "\n</vignette>",  # User prompt with inserted vignette
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="high",  # Use thinking_level for Gemini 3, not thinking_budget since it may result in subpar performance
                        include_thoughts=True  # Include thought summaries in parts/thought within `response` parameters
                        ),
                    system_instruction=system_prompt,  # System prompt
                    temperature=temperature  # Model temperature
                ),
            )

    # Handle content filter triggering
    prompt_feedback = getattr(response, "prompt_feedback", None)  # Check if prompt_feedback exists
    if prompt_feedback and getattr(prompt_feedback, "block_reason", None):  # If block_reason exists within prompt_feedback
        block_reason = prompt_feedback.block_reason  # Get the block_reason object
        block_name = block_reason.name if getattr(block_reason, "name", None) else str(block_reason)  # Safely get the name attribute or convert to string
        print("Content filter triggered:", block_name)
        print("Skipping case...")
        reasoning = "Content filter triggered."
        answer = "Content filter triggered."
        return reasoning, answer

    # Iterate through response object to extract thought summary and differential diagnosis list
    for part in response.parts:
        if not part.text:
            continue
        if part.thought:
            reasoning = part.text  # Extract thought summary
        else:
            answer = part.text  # Extract differential diagnosis list

    return reasoning, answer


# Initialize Gemini 3 Pro client
google_client = genai.Client()
model = "gemini-3-pro-preview"

print("***********************************************")
print(f"Processing model {model} on dataset {dataset_name}...")

# Process each case sequentially
pbar = tqdm(dataset.iterrows(), total=dataset.shape[0])  # Progress bar for tracking

for index, row in pbar:
    pbar.set_description(f"Generating differential diagnoses for case {index + 1} out of {dataset.shape[0]} (case {row['case_id']})")
    reasoning, answer = generate_top5_diagnoses(google_client,
                                                model,
                                                system_prompt,
                                                user_prompt,
                                                row["vignette"],
                                                1,  # Google advises keeping temperature at 1 for Gemini 3 to avoid messing with reasoning behavior
                                                )

    if "Content filter triggered." in reasoning or "Content filter triggered." in answer:
        dataset.loc[index, "model_thoughts"] = reasoning
        dataset.loc[index, "model_diagnosis"] = answer
        print(f"Content filter triggered for case {index + 1} out of {dataset.shape[0]} (case {row['case_id']}).")

    else:
        dataset.loc[index, "model_thoughts"] = reasoning
        dataset.loc[index, "model_diagnosis"] = answer
        print(f"Completed case {index + 1} out of {dataset.shape[0]} (case {row['case_id']}).")

# Save to a JSON file
output_path = f"../../../../../../results/top_5_accuracy/predicted_diagnoses/predicted_diagnoses_{model}_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
dataset.to_json(output_path, orient="records", indent=2)
print("***********************************************")
print(f"{model} predicted diagnoses for calculation of top-5 accuracy saved to JSON.")
