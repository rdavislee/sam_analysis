"""
Generate a comprehensive markdown summary of experiment results.
"""
import json
import os
import glob
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Any

def load_results(results_dir: str = 'results') -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    records = []
    full_data = {}
    
    for filepath in glob.glob(os.path.join(results_dir, '*.json')):
        with open(filepath) as f:
            data = json.load(f)
            
        # Create a unique key for this run
        key = f"{data['config']['optimizer']}_data{data['config']['data_fraction']}_noise{data['config']['noise_fraction']}"
        full_data[key] = data
            
        record = {
            'optimizer': data['config']['optimizer'],
            'data_fraction': data['config']['data_fraction'],
            'noise_fraction': data['config']['noise_fraction'],
            'num_train': data['config']['num_train_samples'],
            'best_test_acc': data['summary']['best_test_acc'],
            'final_test_acc': data['summary']['final_test_acc'],
            'final_train_acc': data['summary']['final_train_acc'],
            'generalization_gap': data['summary']['generalization_gap'],
            'sharpness': data['sharpness']['sharpness'],
            'training_time': data['summary']['training_time_minutes'],
        }
        records.append(record)
        
    return pd.DataFrame(records), full_data

def generate_heatmap_table(df: pd.DataFrame) -> str:
    """Generate the data table corresponding to Figure 1 (SAM Improvement Heatmap)."""
    pivot = df.pivot_table(
        index='data_fraction', 
        columns='noise_fraction', 
        values='final_test_acc',
        aggfunc='first' # Should only be one value per cell
    )
    
    # We need separate pivots for SGD and SAM to compute difference
    sgd_df = df[df['optimizer'] == 'sgd']
    sam_df = df[df['optimizer'] == 'sam']
    
    sgd_pivot = sgd_df.pivot_table(index='data_fraction', columns='noise_fraction', values='final_test_acc')
    sam_pivot = sam_df.pivot_table(index='data_fraction', columns='noise_fraction', values='final_test_acc')
    
    improvement_pivot = sam_pivot - sgd_pivot
    
    md = "## 1. SAM Improvement Heatmap Data\n\n"
    md += "Difference in Test Accuracy (SAM - SGD) for each configuration:\n\n"
    md += "| Data Fraction | Noise 0.0 | Noise 0.2 | Noise 0.4 |\n"
    md += "|---|---|---|---|\n"
    
    for idx, row in improvement_pivot.iterrows():
        md += f"| {idx:.2f} | {row[0.0]:.2f}% | {row[0.2]:.2f}% | {row[0.4]:.2f}% |\n"
        
    md += "\n**Raw Test Accuracies (SGD / SAM):**\n\n"
    md += "| Data Fraction | Noise 0.0 (SGD/SAM) | Noise 0.2 (SGD/SAM) | Noise 0.4 (SGD/SAM) |\n"
    md += "|---|---|---|---|\n"
    
    for idx in improvement_pivot.index:
        row_str = f"| {idx:.2f} |"
        for col in [0.0, 0.2, 0.4]:
            sgd_val = sgd_pivot.loc[idx, col]
            sam_val = sam_pivot.loc[idx, col]
            row_str += f" {sgd_val:.2f}% / {sam_val:.2f}% |"
        md += row_str + "\n"
        
    return md

def generate_sharpness_table(df: pd.DataFrame) -> str:
    """Generate data table for Figure 3 (Sharpness vs Generalization Gap)."""
    md = "\n## 2. Sharpness vs Generalization Gap Data\n\n"
    md += "Comparison of Loss Landscape Sharpness and Generalization Gap (Train Acc - Test Acc):\n\n"
    md += "| Optimizer | Data Frac | Noise Frac | Sharpness | Gen Gap (%) | Test Acc (%) |\n"
    md += "|---|---|---|---|---|---|\n"
    
    # Sort for readability
    sorted_df = df.sort_values(['optimizer', 'data_fraction', 'noise_fraction'])
    
    for _, row in sorted_df.iterrows():
        md += f"| {row['optimizer'].upper()} | {row['data_fraction']} | {row['noise_fraction']} | {row['sharpness']:.4f} | {row['generalization_gap']:.2f} | {row['final_test_acc']:.2f} |\n"
        
    return md

def summarize_learning_curves(full_data: Dict[str, Any]) -> str:
    """Condense Figure 2 data (learning curves) into key statistics."""
    md = "\n## 3. Training Dynamics (Learning Curve Summary)\n\n"
    md += "Condensed summary of training dynamics. 'Early' = Epoch 10, 'Mid' = Epoch 50, 'Final' = Epoch 100.\n\n"
    
    # Define conditions to iterate over
    data_fractions = [0.01, 0.1, 1.0]
    noise_fractions = [0.0, 0.2, 0.4]
    
    md += "| Config (Data/Noise) | Optimizer | Early Test Acc | Mid Test Acc | Final Test Acc | Convergence Speed (Epoch to 90% of Final) |\n"
    md += "|---|---|---|---|---|---|\n"
    
    for data_frac in data_fractions:
        for noise_frac in noise_fractions:
            config_str = f"D={data_frac}/N={noise_frac}"
            
            for opt in ['sgd', 'sam']:
                key = f"{opt}_data{data_frac}_noise{noise_frac}"
                if key not in full_data:
                    continue
                    
                run = full_data[key]
                test_accs = run['test_acc']
                final_acc = test_accs[-1]
                
                # Metrics
                early_acc = test_accs[9] if len(test_accs) > 9 else test_accs[-1]
                mid_acc = test_accs[49] if len(test_accs) > 49 else test_accs[-1]
                
                # Estimate convergence: first epoch reaching 90% of final accuracy
                threshold = 0.9 * final_acc
                conv_epoch = next((i for i, x in enumerate(test_accs) if x >= threshold), len(test_accs))
                
                md += f"| {config_str} | {opt.upper()} | {early_acc:.2f}% | {mid_acc:.2f}% | {final_acc:.2f}% | {conv_epoch} |\n"
    
    return md

def generate_accuracy_vs_data_summary(df: pd.DataFrame) -> str:
    """Summary of Figure 4 (Accuracy vs Data Size)."""
    md = "\n## 4. Scaling Behavior (Accuracy vs Data Size)\n\n"
    md += "Test Accuracy (%) as a function of dataset size for different noise levels:\n\n"
    
    for noise in [0.0, 0.2, 0.4]:
        md += f"**Noise Level: {noise*100:.0f}%**\n\n"
        md += "| Data Fraction | SGD Acc | SAM Acc | Delta |\n"
        md += "|---|---|---|---|\n"
        
        subset = df[df['noise_fraction'] == noise].sort_values('data_fraction')
        
        # Get unique data fractions
        fractions = subset['data_fraction'].unique()
        
        for frac in fractions:
            sgd_row = subset[(subset['optimizer'] == 'sgd') & (subset['data_fraction'] == frac)]
            sam_row = subset[(subset['optimizer'] == 'sam') & (subset['data_fraction'] == frac)]
            
            if not sgd_row.empty and not sam_row.empty:
                sgd_acc = sgd_row.iloc[0]['final_test_acc']
                sam_acc = sam_row.iloc[0]['final_test_acc']
                delta = sam_acc - sgd_acc
                md += f"| {frac} | {sgd_acc:.2f} | {sam_acc:.2f} | {delta:+.2f} |\n"
        md += "\n"
        
    return md

def main(results_dir: str = 'results', output_file: str = 'results_summary.md'):
    df, full_data = load_results(results_dir)
    
    if df.empty:
        print("No results found.")
        return

    print(f"Synthesizing results from {len(df)} runs...")
    
    md_content = "# Experiment Results Synthesis\n\n"
    md_content += "This document contains a structured synthesis of the experimental results for LLM analysis.\n"
    
    md_content += generate_heatmap_table(df)
    md_content += generate_sharpness_table(df)
    md_content += generate_accuracy_vs_data_summary(df)
    md_content += summarize_learning_curves(full_data)
    
    with open(output_file, 'w') as f:
        f.write(md_content)
        
    print(f"Summary written to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Auto-detect paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_results = os.path.join(project_root, 'results')
    default_output = os.path.join(project_root, 'results_summary.md')
    
    parser.add_argument('--results_dir', default=default_results)
    parser.add_argument('--output_file', default=default_output)
    
    args = parser.parse_args()
    main(args.results_dir, args.output_file)

