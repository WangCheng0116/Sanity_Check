from datasets import load_dataset
import pandas as pd
import os
import random
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import time

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Create directories
os.makedirs("malicious", exist_ok=True)
os.makedirs("benign", exist_ok=True)
os.makedirs("malicious_but_cleaned", exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key="<>")

def api_call(message_content):
    """Call OpenAI API to clean malicious content"""
    messages = [
        {"role": "user", "content": message_content}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip()
        
        # Trim quotation marks if present
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        elif content.startswith("'") and content.endswith("'"):
            content = content[1:-1]
        
        return content
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def create_cleaning_prompt(original_text):
    """Create a professional prompt for GPT to clean malicious content"""
    prompt = f"""Task: Content Sanitization and Lexical Substitution

Objective: Transform the provided text by replacing potentially harmful, dangerous, illegal, or malicious terminology with benign, contextually appropriate alternatives while preserving the original structure and meaning.

Requirements:
1. Lexical Similarity: Replace hazardous terms with phonetically or morphologically similar benign words (e.g., "bomb" → "book", "launder money" → "lauder clothes")
2. Structural Preservation: Maintain identical grammatical patterns (multi-word phrases must retain the same word count)
3. Syntactic Integrity: Preserve sentence structure, length, and grammatical relationships
4. Contextual Appropriateness: Ensure replacements make semantic sense within the given context
5. Output Format: Return only the sanitized text without additional commentary or explanations

Input Text: "{original_text}"

Sanitized Output:"""
    return prompt

# Malicious datasets
datasets_malicious = [
    {"name": "walledai/AdvBench", "split": "train", "column": "prompt", "filename": "advbench.csv"},
    {"name": "walledai/ForbiddenQuestions", "split": "train", "column": "prompt", "filename": "forbiddenquestions.csv"},
    {"name": "walledai/MaliciousInstruct", "split": "train", "column": "prompt", "filename": "maliciousinstruct.csv"},
    {"name": "walledai/HarmBench", "split": "train", "column": "prompt", "filename": "harmbench.csv", "subset": "standard"}
]

print("Processing malicious datasets...")
for dataset_info in tqdm(datasets_malicious, desc="Loading malicious datasets"):
    print(f"\nProcessing {dataset_info['name']}...")
    
    try:
        if "subset" in dataset_info:
            ds = load_dataset(dataset_info["name"], dataset_info["subset"], split=dataset_info["split"])
        else:
            ds = load_dataset(dataset_info["name"], split=dataset_info["split"])
        
        df = pd.DataFrame({"prompt": ds[dataset_info["column"]]})
        df.to_csv(f"malicious/{dataset_info['filename']}", index=False)
        print(f"Saved {len(df)} samples to malicious/{dataset_info['filename']}")
    except Exception as e:
        print(f"Error processing {dataset_info['name']}: {e}")

# Benign datasets
datasets_benign = [
    {"name": "tatsu-lab/alpaca", "split": "train", "column": "instruction", "filename": "alpaca.csv", "sample": 400},
    {"name": "euclaise/writingprompts", "split": "test", "column": "prompt", "filename": "writingprompts.csv", "sample": 400},
    {"name": "basicv8vc/SimpleQA", "split": "test", "column": "problem", "filename": "simpleqa.csv", "sample": 400},
    {"name": "math-ai/minervamath", "split": "test", "column": "question", "filename": "minervamath.csv"}
]

print("\nProcessing benign datasets...")
for dataset_info in tqdm(datasets_benign, desc="Loading benign datasets"):
    print(f"\nProcessing {dataset_info['name']}...")
    
    try:
        ds = load_dataset(dataset_info["name"], split=dataset_info["split"])
        
        if "sample" in dataset_info:
            ds = ds.shuffle(seed=42).select(range(dataset_info["sample"]))
        
        df = pd.DataFrame({"prompt": ds[dataset_info["column"]]})
        df.to_csv(f"benign/{dataset_info['filename']}", index=False)
        print(f"Saved {len(df)} samples to benign/{dataset_info['filename']}")
    except Exception as e:
        print(f"Error processing {dataset_info['name']}: {e}")

# Clean malicious datasets using GPT
print("\nCleaning malicious datasets with GPT...")
malicious_files = [info["filename"] for info in datasets_malicious]

for filename in malicious_files:
    filepath = f"malicious/{filename}"
    
    if not os.path.exists(filepath):
        print(f"File {filepath} not found, skipping...")
        continue
    
    print(f"\nCleaning {filename}...")
    
    # Read the malicious dataset
    df = pd.read_csv(filepath)
    
    # Clean each prompt
    cleaned_prompts = []
    failed_count = 0
    
    for i, prompt in enumerate(tqdm(df['prompt'], desc=f"Cleaning {filename}")):
        # Create cleaning prompt
        cleaning_prompt = create_cleaning_prompt(prompt)
        
        # Call GPT API
        cleaned_prompt = api_call(cleaning_prompt)
        
        # Print the original and cleaned prompts for monitoring
        print(f"\n--- Sample {i+1}/{len(df)} ---")
        print(f"Original: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        if cleaned_prompt is not None:
            print(f"Cleaned:  {cleaned_prompt[:100]}{'...' if len(cleaned_prompt) > 100 else ''}")
            cleaned_prompts.append(cleaned_prompt)
        else:
            print("Cleaned:  [API FAILED - KEPT ORIGINAL]")
            # If API call fails, keep original prompt
            cleaned_prompts.append(prompt)
            failed_count += 1
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Save cleaned dataset
    cleaned_df = pd.DataFrame({"prompt": cleaned_prompts})
    cleaned_filepath = f"malicious_but_cleaned/{filename}"
    cleaned_df.to_csv(cleaned_filepath, index=False)
    
    print(f"Saved {len(cleaned_df)} cleaned samples to {cleaned_filepath}")
    if failed_count > 0:
        print(f"Warning: {failed_count} prompts failed to clean and were kept original")

print("\nAll datasets processed successfully!")
print("\nSummary:")
print("- Original malicious datasets saved in 'malicious/' directory")
print("- Cleaned malicious datasets saved in 'malicious_but_cleaned/' directory") 
print("- Benign datasets saved in 'benign/' directory")