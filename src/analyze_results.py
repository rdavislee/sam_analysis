"""
Analyze results and generate figures for blog.
"""
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_results(results_dir: str = 'results') -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    records = []
    for filepath in glob.glob(os.path.join(results_dir, '*.json')):
        with open(filepath) as f:
            data = json.load(f)
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
    return pd.DataFrame(records)


def compute_sam_improvement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SAM improvement over SGD for each condition."""
    # Pivot to get SAM and SGD side by side
    pivot = df.pivot_table(
        index=['data_fraction', 'noise_fraction'],
        columns='optimizer',
        values='final_test_acc'
    ).reset_index()
    
    if 'sam' in pivot.columns and 'sgd' in pivot.columns:
        pivot['sam_improvement'] = pivot['sam'] - pivot['sgd']
    else:
        pivot['sam_improvement'] = 0.0 # Placeholder if data missing
        
    return pivot


def plot_figure1_heatmap(df: pd.DataFrame, output_path: str):
    """
    Figure 1: SAM improvement heatmap.
    X-axis: Noise fraction, Y-axis: Data fraction, Color: SAM - SGD accuracy
    """
    improvement_df = compute_sam_improvement(df)
    
    # Pivot for heatmap
    heatmap_data = improvement_df.pivot(
        index='data_fraction', 
        columns='noise_fraction', 
        values='sam_improvement'
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'SAM Improvement (%)'}
    )
    plt.xlabel('Label Noise Fraction')
    plt.ylabel('Training Data Fraction')
    plt.title('SAM Test Accuracy Improvement Over SGD')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_figure2_learning_curves(results_dir: str, output_path: str):
    """
    Figure 2: Learning curves for ALL 9 conditions.
    3x3 grid showing all regimes.
    Rows: Data Fraction (1%, 10%, 100%)
    Cols: Noise Fraction (0%, 20%, 40%)
    """
    data_fractions = [0.01, 0.1, 1.0]
    noise_fractions = [0.0, 0.2, 0.4]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, data_frac in enumerate(data_fractions):
        for j, noise_frac in enumerate(noise_fractions):
            ax = axes[i, j]
            
            # Load SGD and SAM
            for opt in ['sgd', 'sam']:
                filename = f"{opt}_data{data_frac}_noise{noise_frac}.json"
                filepath = os.path.join(results_dir, filename)
                
                if os.path.exists(filepath):
                    with open(filepath) as f:
                        data = json.load(f)
                    
                    # Plot test accuracy
                    ax.plot(data['test_acc'], label=f'{opt.upper()}' if i==0 and j==0 else "", linestyle='-')
                    # Plot train accuracy (dashed) - Optional, maybe too cluttered for 3x3?
                    # Let's keep it but make it faint
                    ax.plot(data['train_acc'], label=f'{opt.upper()} Train' if i==0 and j==0 else "", linestyle='--', alpha=0.3)
            
            ax.set_title(f'Data: {data_frac*100:.0f}% | Noise: {noise_frac*100:.0f}%')
            
            if i == 2:
                ax.set_xlabel('Epoch')
            if j == 0:
                ax.set_ylabel('Accuracy (%)')
                
            ax.grid(True, alpha=0.3)
            
            # Add simple legend to first plot only to save space
            if i == 0 and j == 0:
                 ax.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_figure3_sharpness_vs_gap(df: pd.DataFrame, output_path: str):
    """
    Figure 3: Sharpness vs generalization gap scatter plot.
    """
    plt.figure(figsize=(8, 6))
    
    for opt in ['sgd', 'sam']:
        subset = df[df['optimizer'] == opt]
        plt.scatter(
            subset['sharpness'],
            subset['generalization_gap'],
            label=opt.upper(),
            alpha=0.7,
            s=100
        )
    
    plt.xlabel('Sharpness')
    plt.ylabel('Generalization Gap (Train Acc - Test Acc)')
    plt.title('Sharpness vs Generalization Gap')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_figure4_accuracy_vs_data(df: pd.DataFrame, output_path: str):
    """
    Figure 4: Test accuracy vs dataset size, separate lines for SGD/SAM.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    noise_levels = sorted(df['noise_fraction'].unique())
    
    for idx, noise in enumerate(noise_levels):
        if idx >= 3: break
        
        ax = axes[idx]
        subset = df[df['noise_fraction'] == noise]
        
        for opt in ['sgd', 'sam']:
            opt_data = subset[subset['optimizer'] == opt].sort_values('data_fraction')
            if not opt_data.empty:
                ax.plot(
                    opt_data['data_fraction'] * 100,
                    opt_data['final_test_acc'],
                    'o-',
                    label=opt.upper(),
                    linewidth=2,
                    markersize=8
                )
        
        ax.set_xlabel('Training Data (%)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'Noise: {noise*100:.0f}%')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main(results_dir: str = 'results', output_dir: str = 'blog/figures'):
    """Generate all figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_results(results_dir)
    print(f"Loaded {len(df)} experiment results")
    if len(df) == 0:
        print("No results found. Run experiments first.")
        return

    print(df.head())
    
    try:
        plot_figure1_heatmap(df, os.path.join(output_dir, 'fig1_heatmap.png'))
    except Exception as e:
        print(f"Could not plot heatmap (need full matrix?): {e}")

    try:
        plot_figure2_learning_curves(results_dir, os.path.join(output_dir, 'fig2_learning_curves.png'))
    except Exception as e:
        print(f"Could not plot learning curves: {e}")

    try:
        plot_figure3_sharpness_vs_gap(df, os.path.join(output_dir, 'fig3_sharpness.png'))
    except Exception as e:
        print(f"Could not plot sharpness: {e}")

    try:
        plot_figure4_accuracy_vs_data(df, os.path.join(output_dir, 'fig4_accuracy.png'))
    except Exception as e:
        print(f"Could not plot accuracy vs data: {e}")
    
    print("\nAll figures generated!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Default to finding results relative to this script location if not specified
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_results = os.path.join(project_root, 'results')
    default_output = os.path.join(project_root, 'blog', 'figures')
    
    parser.add_argument('--results_dir', default=default_results)
    parser.add_argument('--output_dir', default=default_output)
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)

