import argparse
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from typing import List, Tuple, Dict
import random
from itertools import product

# Set random seeds for reproducibility
def set_random_seeds(seed: int):
    """Set random seeds for all libraries."""
    random.seed(seed)
    np.random.seed(seed)

def load_npy_files(model_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Load all .npy files and separate into benign, malicious, and cleaned categories."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found")
    
    npy_files = glob.glob(os.path.join(model_dir, "*.npy"))
    
    benign_files = [f for f in npy_files if os.path.basename(f).startswith("benign_")]
    malicious_files = [f for f in npy_files if os.path.basename(f).startswith("malicious_")]
    cleaned_files = [f for f in npy_files if os.path.basename(f).startswith("cleaned_")]
    
    print(f"[found] {len(benign_files)} benign files, {len(malicious_files)} malicious files, {len(cleaned_files)} cleaned files")
    
    if not benign_files or not malicious_files:
        raise ValueError("Need at least one benign and one malicious file")
    if not cleaned_files:
        raise ValueError("Need at least one cleaned file")
    
    return benign_files, malicious_files, cleaned_files

def load_and_balance_data(file1: str, file2: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from two files and balance classes by downsampling."""
    # Load data
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    print(f"[loaded] {os.path.basename(file1)}: {data1.shape}")
    print(f"[loaded] {os.path.basename(file2)}: {data2.shape}")
    
    # Balance classes by downsampling
    min_samples = min(data1.shape[0], data2.shape[0])
    
    # Set seed for reproducible sampling
    np.random.seed(seed)
    
    if data1.shape[0] > min_samples:
        indices = np.random.choice(data1.shape[0], min_samples, replace=False)
        data1 = data1[indices]
    
    if data2.shape[0] > min_samples:
        indices = np.random.choice(data2.shape[0], min_samples, replace=False)
        data2 = data2[indices]
    
    # Create labels (first file = 0, second file = 1)
    labels1 = np.zeros(data1.shape[0])
    labels2 = np.ones(data2.shape[0])
    
    # Combine data
    X = np.vstack([data1, data2])
    y = np.hstack([labels1, labels2])
    
    print(f"[balanced] Final dataset shape: {X.shape}, Classes: {np.bincount(y.astype(int))}")
    
    return X, y

def apply_label_noise(y: np.ndarray, noise_ratio: float, seed: int) -> np.ndarray:
    """Apply label noise by randomly shuffling a portion of labels."""
    if noise_ratio == 0:
        return y.copy()
    
    np.random.seed(seed)
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(n_samples * noise_ratio)
    
    # Randomly select indices to add noise
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    
    # Randomly shuffle the labels for selected indices
    shuffled_labels = np.random.permutation(y[noisy_indices])
    y_noisy[noisy_indices] = shuffled_labels
    
    return y_noisy

def train_and_evaluate_classifiers(X: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, float]:
    """Train MLP and SVM classifiers and return their accuracies."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    results = {}
    
    # MLP Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,), 
        max_iter=300, 
        alpha=0.01,
        solver='adam', 
        verbose=0, 
        random_state=seed,
        learning_rate_init=0.01
    )
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    results['MLP'] = accuracy_score(y_test, mlp_pred)
    
    # SVM Classifier
    svm_model = SVC(kernel='linear', random_state=seed)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    results['SVM'] = accuracy_score(y_test, svm_pred)
    
    return results

def get_comparison_combinations(benign_files: List[str], malicious_files: List[str], 
                               cleaned_files: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Generate all combinations for each comparison type."""
    combinations = {
        'benign_vs_malicious': [(b, m) for b in benign_files for m in malicious_files],
        'benign_vs_cleaned': [(b, c) for b in benign_files for c in cleaned_files],
        'cleaned_vs_malicious': [(c, m) for c in cleaned_files for m in malicious_files]
    }
    return combinations

def create_accuracy_matrices(combinations: Dict[str, List[Tuple[str, str]]], 
                            random_ratios: List[float],
                            results: Dict[Tuple[str, str, str, float], Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Create accuracy matrices for MLP and SVM with comparison types as rows."""
    comparison_types = ['benign_vs_malicious', 'benign_vs_cleaned', 'cleaned_vs_malicious']
    
    # Calculate total combinations across all comparison types
    max_combinations = max(len(combinations[comp_type]) for comp_type in comparison_types)
    n_comparisons = len(comparison_types)
    
    # For simplicity, we'll use the first noise ratio for the main visualization
    # If multiple ratios are provided, we'll average them or handle separately
    primary_ratio = random_ratios[0]
    
    mlp_matrix = np.full((n_comparisons, max_combinations), np.nan)
    svm_matrix = np.full((n_comparisons, max_combinations), np.nan)
    
    combination_labels = []
    comparison_labels = []
    
    # Create labels for each comparison type
    for i, comp_type in enumerate(comparison_types):
        comparison_labels.append(comp_type.replace('_', ' ').title())
        
        # Create column labels for this comparison type
        if i == 0:  # Only create labels once for the longest combination list
            for j, (file1, file2) in enumerate(combinations[comp_type]):
                if j < max_combinations:
                    name1 = get_file_basename(file1)
                    name2 = get_file_basename(file2)
                    combination_labels.append(f"{name1}\nvs\n{name2}")
            
            # Pad with empty labels if needed
            while len(combination_labels) < max_combinations:
                combination_labels.append("")
    
    # Fill matrices
    for i, comp_type in enumerate(comparison_types):
        for j, (file1, file2) in enumerate(combinations[comp_type]):
            if j < max_combinations:
                key = (comp_type, file1, file2, primary_ratio)
                if key in results:
                    mlp_matrix[i, j] = results[key]['MLP']
                    svm_matrix[i, j] = results[key]['SVM']
    
    return mlp_matrix, svm_matrix, combination_labels, comparison_labels

def get_file_basename(filepath: str) -> str:
    """Extract clean basename from file path."""
    basename = os.path.basename(filepath)
    for prefix in ["benign_", "malicious_", "cleaned_"]:
        if basename.startswith(prefix):
            basename = basename[len(prefix):]
    return basename.replace(".npy", "")

def visualize_accuracy_matrices(mlp_matrix: np.ndarray, svm_matrix: np.ndarray, 
                               combination_labels: List[str], comparison_labels: List[str],
                               model_name: str, seed: int, noise_ratio: float):
    """Create visualization of accuracy matrices with comparison types as rows."""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # MLP heatmap
    sns.heatmap(mlp_matrix, annot=True, fmt='.3f', cmap='viridis', 
                xticklabels=combination_labels, yticklabels=comparison_labels,
                ax=ax1, cbar_kws={'label': 'Accuracy'}, mask=np.isnan(mlp_matrix))
    ax1.set_title(f'MLP Classifier Accuracy Matrix (Noise: {noise_ratio:.1f})')
    ax1.set_xlabel('Dataset Combinations')
    ax1.set_ylabel('Comparison Types')
    ax1.tick_params(axis='x', rotation=45)
    
    # SVM heatmap
    sns.heatmap(svm_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=combination_labels, yticklabels=comparison_labels,
                ax=ax2, cbar_kws={'label': 'Accuracy'}, mask=np.isnan(svm_matrix))
    ax2.set_title(f'SVM Classifier Accuracy Matrix (Noise: {noise_ratio:.1f})')
    ax2.set_xlabel('Dataset Combinations')
    ax2.set_ylabel('Comparison Types')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure in model directory
    model_short_name = model_name.split("/")[-1]
    os.makedirs(model_short_name, exist_ok=True)
    output_filename = os.path.join(model_short_name, f"{model_short_name}_three_way_analysis_seed{seed}_noise{noise_ratio:.1f}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[saved] Visualization: {output_filename}")
    
    plt.show()

def save_results_summary(results: Dict[Tuple[str, str, str, float], Dict[str, float]], 
                        model_name: str, seed: int):
    """Save detailed results to CSV file."""
    summary_data = []
    
    for (comp_type, file1, file2, noise_ratio), accuracies in results.items():
        name1 = get_file_basename(file1)
        name2 = get_file_basename(file2)
        
        summary_data.append({
            'comparison_type': comp_type,
            'dataset1': name1,
            'dataset2': name2,
            'noise_ratio': noise_ratio,
            'mlp_accuracy': accuracies['MLP'],
            'svm_accuracy': accuracies['SVM']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV in model directory
    model_short_name = model_name.split("/")[-1]
    os.makedirs(model_short_name, exist_ok=True)
    csv_filename = os.path.join(model_short_name, f"{model_short_name}_three_way_results_seed{seed}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"[saved] Results summary: {csv_filename}")
    
    # Print summary statistics by comparison type and noise ratio
    print("\n[summary] Overall Results by Comparison Type and Noise Ratio:")
    for comp_type in sorted(df['comparison_type'].unique()):
        print(f"\n{comp_type.replace('_', ' ').title()}:")
        subset_comp = df[df['comparison_type'] == comp_type]
        for ratio in sorted(subset_comp['noise_ratio'].unique()):
            subset = subset_comp[subset_comp['noise_ratio'] == ratio]
            print(f"  Noise {ratio:.1f} - MLP: {subset['mlp_accuracy'].mean():.3f}±{subset['mlp_accuracy'].std():.3f}, "
                  f"SVM: {subset['svm_accuracy'].mean():.3f}±{subset['svm_accuracy'].std():.3f}")

def main():
    parser = argparse.ArgumentParser("Three-way hidden states classification analysis")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model path (used for directory naming)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--random_ratio", type=float, nargs='+', 
                       default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       help="List of noise ratios to test")
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Get model directory name
    model_dir = args.model_path.split("/")[-1]
    
    # Load .npy files
    benign_files, malicious_files, cleaned_files = load_npy_files(model_dir)
    
    # Get all combinations for each comparison type
    combinations = get_comparison_combinations(benign_files, malicious_files, cleaned_files)
    
    total_combinations = sum(len(combs) for combs in combinations.values()) * len(args.random_ratio)
    print(f"[analysis] Processing {total_combinations} total combinations across 3 comparison types")
    print(f"  - Benign vs Malicious: {len(combinations['benign_vs_malicious'])} combinations")
    print(f"  - Benign vs Cleaned: {len(combinations['benign_vs_cleaned'])} combinations")
    print(f"  - Cleaned vs Malicious: {len(combinations['cleaned_vs_malicious'])} combinations")
    
    # Store results for all combinations
    all_results = {}
    
    # Process all comparison types
    for comp_type, file_combinations in combinations.items():
        print(f"\n[processing] {comp_type.replace('_', ' ').title()}")
        
        for file1, file2 in file_combinations:
            name1 = get_file_basename(file1)
            name2 = get_file_basename(file2)
            
            print(f"\n  [combination] {name1} vs {name2}")
            
            # Load and balance data once for this combination
            X, y_original = load_and_balance_data(file1, file2, args.seed)
            
            # Test different noise ratios
            for noise_ratio in args.random_ratio:
                print(f"    [noise ratio] {noise_ratio:.1f}")
                
                # Apply label noise
                y_noisy = apply_label_noise(y_original, noise_ratio, args.seed)
                
                # Train and evaluate classifiers
                results = train_and_evaluate_classifiers(X, y_noisy, args.seed)
                
                print(f"      [results] MLP: {results['MLP']:.3f}, SVM: {results['SVM']:.3f}")
                
                # Store results
                all_results[(comp_type, file1, file2, noise_ratio)] = results
    
    # Create and visualize results for each noise ratio
    for noise_ratio in args.random_ratio:
        print(f"\n[visualizing] Results for noise ratio: {noise_ratio:.1f}")
        
        # Create accuracy matrices for this noise ratio
        mlp_matrix, svm_matrix, combination_labels, comparison_labels = create_accuracy_matrices(
            combinations, [noise_ratio], all_results
        )
        
        # Visualize results
        visualize_accuracy_matrices(mlp_matrix, svm_matrix, combination_labels, 
                                   comparison_labels, args.model_path, args.seed, noise_ratio)
    
    # Save detailed results
    save_results_summary(all_results, args.model_path, args.seed)
    
    print("\n[complete] Three-way analysis finished!")

if __name__ == "__main__":
    main()