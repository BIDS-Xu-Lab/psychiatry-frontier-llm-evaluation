# Generate top-5 differential diagnoses (n=196) using GPT-5.2
import json
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import datetime

# Load API key from .env file
load_dotenv()

# Import the vignette dataset
dataset_path = "../../../datasets/combined/combined_jama.json"
dataset_name = str(dataset_path).split("/")[-1].split(".")[0]  # Extract the dataset name
with open(dataset_path, "r") as f:
    combined = json.load(f)
dataset = pd.DataFrame(combined)  # Convert to DataFrame

# Define system instructions and user prompt
with open("../../prompts/top_5_accuracy/system_prompt.txt") as f:
    system_prompt = f.read()

with open("../../prompts/top_5_accuracy/user_prompt.txt") as f:
    user_prompt = f.read()


# Generate differential diagnosis for one case using GPT-5.2
def generate_top5_diagnoses(client, model: str, system_prompt: str, user_prompt: str, vignette: str) -> tuple:
    # Prepare API call parameters
    # OpenAI: Make API call and create response object
    if model.startswith("gpt"):
        response = client.responses.create(
            model=model,  # gpt-5.2-pro is way too expensive; use gpt-5.2
            reasoning={
                "effort": "xhigh",  # Favors even more complete reasoning
                "summary": "detailed"  # Give as much detail as possible in thinking block
            },
            text={
                "verbosity": "low"  # To keep the model on task for diagnosis
            },
            input=[
                {
                    "role": "developer",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt + "\n<vignette>\n" + vignette + "\n</vignette>"
                }
            ]
        )

    # Handle content filter triggering
    prompt_feedback = getattr(response, "incomplete_details", None)
    if prompt_feedback and getattr(prompt_feedback, "reason", None):
        block_reason = prompt_feedback.reason
        block_name = block_reason.name if getattr(block_reason, "name", None) else str(block_reason)
        print("Content filter triggered:", block_name)

    # Extract reasoning and answer from response object
    # Handle different output array lengths dynamically
    if len(response.output) == 1:
        # Only differential diagnosis present
        reasoning = None
        answer = response.output[0].content[0].text
    elif len(response.output) >= 2:
        # Extract the thought summary by concatenating all thinking blocks using newlines and a list comprehension
        reasoning = "\n\n".join([block.text for block in response.output[0].summary])

        # Extract the differential diagnosis list
        answer = response.output[1].content[0].text
    else:
        # Handle unexpected cases
        reasoning = None
        answer = None

    return reasoning, answer


# Initialize Gemini 3 Pro client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-5.2"

print("***********************************************")
print(f"Processing model {model} on dataset {dataset_name}...")

# Process each case sequentially
pbar = tqdm(dataset.iterrows(), total=dataset.shape[0])  # Progress bar for tracking

for index, row in pbar:
    pbar.set_description(f"Generating diagnostic reasoning trace {index + 1} out of {dataset.shape[0]} (case {row['case_id']})")
    reasoning, answer = generate_top5_diagnoses(openai_client,
                                                model,
                                                system_prompt,
                                                user_prompt,
                                                row["vignette"],
                                                # Temperature not supported with reasoning effort set to high 
                                                )
    dataset.loc[index, "model_thoughts"] = reasoning
    dataset.loc[index, "model_diagnosis"] = answer
    print(f"Completed case {index + 1} out of {dataset.shape[0]} (case {row['case_id']}).")

# Check for missing or empty values and rerun until none remain
max_iterations = 10  # Prevent infinite loop
iteration = 0

while iteration < max_iterations:
    # Check for missing or empty values
    missing_values = dataset[
        (dataset["model_thoughts"].isnull()) | 
        (dataset["model_thoughts"] == "") |
        (dataset["model_diagnosis"].isnull()) | 
        (dataset["model_diagnosis"] == "")
    ]

    if missing_values.empty:
        print("No missing or empty values found in the results.")
        break

    iteration += 1
    print(f"\n=== Iteration {iteration} ===")
    print(f"Missing or empty values found in {len(missing_values)} cases:")
    print(missing_values[["case_id", "model_thoughts", "model_diagnosis"]])

    # Get all indices with missing values
    missing_indices = missing_values.index.tolist()
    print(f"Rerunning {len(missing_indices)} cases with missing data: {missing_indices}")

    for index_to_rerun in missing_indices:
        print(f"\nRerunning case {index_to_rerun} (case_id: {dataset.iloc[index_to_rerun]['case_id']})")
        try:
            reasoning, answer = generate_top5_diagnoses(openai_client,
                                                            model,
                                                            system_prompt,
                                                            user_prompt,
                                                            dataset.iloc[index_to_rerun]["vignette"],
                                                        )

            dataset.loc[index_to_rerun, "model_thoughts"] = reasoning
            dataset.loc[index_to_rerun, "model_diagnosis"] = answer
            print(f"Successfully completed case {index_to_rerun}")

        except Exception as e:
            print(f"Error processing case {index_to_rerun}: {str(e)}")
            continue

print(f"\nCompleted after {iteration} iteration(s).")


# Save to a JSON file
output_path = f"../../../results/top_5_accuracy/predicted_diagnoses/predicted_diagnoses_{model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
dataset.to_json(output_path, orient="records", indent=2)
print("***********************************************")
print(f"{model} predicted diagnoses for calculation of top-5 accuracy saved to JSON.")
