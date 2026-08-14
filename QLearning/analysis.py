import matplotlib.pyplot as plt
import os
import pandas as pd

METRICS_DIR = "/Users/shreyanakum/Documents/HighRoller/QLearning/metrics"
FIGS_DIR = f"{METRICS_DIR}/figs"

def make_histograms(df, testset):
    exclude_cols = {'model', 'env_name', 'iteration'}
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in exclude_cols]

    if testset == 'eval':
        testset = 'Evaluation'
    elif testset == 'training':
        testset = 'Training'

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        
        # plot histogram with kernel density estimation edge styling
        plt.hist(df[col].dropna(), bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        
        # formatted titles and labels
        col_title = col.replace('_', ' ').title()
        plt.title(f"{col_title} Distribution ({testset})", fontsize=14, fontweight='bold', pad=12)
        plt.xlabel(col_title, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()

        # save
        save_path = os.path.join(FIGS_DIR, f"{testset}_{col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

# Ensure base figures directory exists
os.makedirs(FIGS_DIR, exist_ok=True)

for metrics_csv in os.listdir(METRICS_DIR):
    if not metrics_csv.endswith('.csv'):
        continue

    # Note: Using path.join prevents issues with missing slashes
    testset = metrics_csv.split('_')[-1].replace('.csv', '')
    df = pd.read_csv(os.path.join(METRICS_DIR, metrics_csv))

    make_histograms(df, testset)