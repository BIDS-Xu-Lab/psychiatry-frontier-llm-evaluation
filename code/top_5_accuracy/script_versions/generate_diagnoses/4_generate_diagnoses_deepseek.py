# Generate top-5 differential diagnoses (n=196) using Claude Opus 4.5
import json
import pandas as pd
import os
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


# Generate differential diagnosis for one case using Claude Opus 4.5
def generate_top5_diagnoses(client, model, system_prompt, user_prompt, vignette, temperature):
    # Prepare API call parameters
    # DeepSeek: Make API call and create response object
    if model.startswith("deepseek"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n<vignette>\n" + vignette + "\n</vignette>"},
            ],
            temperature=temperature,
            stream=False
        )
        # Extract the response content
        answer = response.choices[0].message.content

        # Extract the thinking block
        reasoning = response.choices[0].message.reasoning_content

    return reasoning, answer


# Initialize Anthropic client
deepseek_client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
model = "deepseek-reasoner"  # Select latest reasoning model; in this case, DeepSeek-V3.2

print("***********************************************")
print(f"Processing model {model} on dataset {dataset_name}...")

# Process each case sequentially
pbar = tqdm(dataset.iterrows(), total=dataset.shape[0])  # Progress bar for tracking

for index, row in pbar:
    pbar.set_description(f"Generating differential diagnoses for case {index + 1} out of {dataset.shape[0]} (case {row['case_id']})")
    reasoning, answer = generate_top5_diagnoses(deepseek_client,
                                                model,
                                                system_prompt,
                                                user_prompt,
                                                row["vignette"],
                                                0  # DeepSeek recommends temperature 0 for coding/math tasks where there is a correct answer
                                                )
    dataset.loc[index, "model_thoughts"] = reasoning
    dataset.loc[index, "model_diagnosis"] = answer
    print(f"Completed case {index + 1} out of {dataset.shape[0]} (case {row['case_id']}).")

# Save to a JSON file
output_path = f"../../../results/top_5_accuracy/predicted_diagnoses/predicted_diagnoses_{model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
dataset.to_json(output_path, orient="records", indent=2)
print("***********************************************")
print(f"{model} predicted diagnoses for calculation of top-5 accuracy saved to JSON.")