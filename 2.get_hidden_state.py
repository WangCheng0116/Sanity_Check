import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple
import os
import numpy as np
import warnings
import glob

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Environment & global config
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------------------------------------------
# Utility functions for I/O
# -----------------------------------------------------------------------------

def load_csv_prompts(file_path: str) -> List[str]:
    """Load prompts from a CSV file."""
    df = pd.read_csv(file_path)
    return df["prompt"].tolist()


def save_hidden_states(
    hidden_states: np.ndarray,
    model_name: str,
    category: str,
    csv_name: str,
    layer_idx: int = -1,
):
    """Save hidden states to .npy file with organized directory structure."""
    model_short_name = model_name.split("/")[-1]
    
    # Create model directory if it doesn't exist
    model_dir = model_short_name
    os.makedirs(model_dir, exist_ok=True)
    
    # Create filename: {category}_{csv_name}.npy
    csv_basename = os.path.splitext(os.path.basename(csv_name))[0]
    filename = f"{category}_{csv_basename}.npy"
    filepath = os.path.join(model_dir, filename)
    
    np.save(filepath, hidden_states)
    print(f"[saved] {filepath} • shape={hidden_states.shape}")

# -----------------------------------------------------------------------------
# Hidden states extraction
# -----------------------------------------------------------------------------

def batched_forward_extract_hidden_states(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer_idx: int = -1,
) -> np.ndarray:
    """Extract hidden states from the last token of input prompts.
    Returns:
        last_token_hidden: (B, hidden_dim) ndarray – hidden state of last input token
    """
    device = model.device
    tok_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**tok_out, output_hidden_states=True)
        hidden_states_layer = outputs.hidden_states[layer_idx]  # (B, L, D)
        
        # Get index of last input token for each sequence
        seq_lens = tok_out["attention_mask"].sum(dim=1) - 1  # (B,)
        last_token_hidden = hidden_states_layer[range(len(prompts)), seq_lens]
        last_token_hidden = last_token_hidden.detach().cpu().float().numpy()

    return last_token_hidden


def process_csv_file(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    csv_path: str,
    category: str,
    model_name: str,
    layer_idx: int,
    batch_size: int,
):
    """Process a single CSV file and extract hidden states."""
    print(f"[processing] {csv_path}")
    
    # Load prompts from CSV
    prompts = load_csv_prompts(csv_path)
    print(f"[loaded] {len(prompts)} prompts from {csv_path}")
    
    # Container for all hidden states
    all_hidden_states = []
    
    # Process in batches
    for start in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {category}", unit="batch"):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        
        # Extract hidden states for this batch
        hidden_states = batched_forward_extract_hidden_states(
            model, tokenizer, batch_prompts, layer_idx=layer_idx
        )
        all_hidden_states.append(hidden_states)
    
    # Combine all batches
    final_hidden_states = np.vstack(all_hidden_states)
    
    # Save to file
    save_hidden_states(final_hidden_states, model_name, category, csv_path, layer_idx)
    
    return final_hidden_states


def get_folder_name(base_name: str, augmentation: str) -> str:
    """Generate folder name based on base category and augmentation method."""
    if augmentation:
        return f"{base_name}_{augmentation}"
    return base_name


def process_category_folder(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    base_category: str,
    augmentation: str,
    model_path: str,
    layer_idx: int,
    batch_size: int,
):
    """Process all CSV files in a category folder (with optional augmentation suffix)."""
    folder_name = get_folder_name(base_category, augmentation)
    
    if os.path.exists(folder_name):
        csv_files = glob.glob(os.path.join(folder_name, "*.csv"))
        print(f"[found] {len(csv_files)} CSV files in {folder_name} folder")
        
        for csv_file in csv_files:
            # Use the full folder name as category for file naming
            process_csv_file(
                model, tokenizer, csv_file, folder_name, model_path, layer_idx, batch_size
            )
    else:
        print(f"[warning] Folder '{folder_name}' not found")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Hidden state extraction from CSV files")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64, help="Number of prompts per batch")
    parser.add_argument("--augmentation", type=str, default="", 
                       help="Augmentation method suffix (e.g., 'flip', 'inception', 'base64'). If empty, uses base folders.")
    args = parser.parse_args()

    # Model / tokenizer
    print(f"[load] {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        output_hidden_states=True,
    ).eval()

    # Print augmentation info
    if args.augmentation:
        print(f"[augmentation] Using augmentation method: {args.augmentation}")
        print(f"[folders] Looking for: malicious_{args.augmentation}, benign_{args.augmentation}, cleaned_{args.augmentation}")
    else:
        print(f"[folders] Using base folders: malicious, benign, cleaned")

    # Process each category with augmentation support
    categories = ["malicious", "benign", "cleaned"]
    
    for category in categories:
        process_category_folder(
            model, tokenizer, category, args.augmentation, 
            args.model_path, args.layer_idx, args.batch_size
        )

    print("[complete] All hidden states extracted and saved")


if __name__ == "__main__":
    main()