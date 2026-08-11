import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys


sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",        # Matches LaTeX document style
    "font.size": 18,               # Base font size
    "axes.labelsize": 20,          # X and Y axis labels
    "axes.titlesize": 24,          # Subplot titles
    "xtick.labelsize": 16,         # X axis tick numbers
    "ytick.labelsize": 16,         # Y axis tick numbers
    "legend.fontsize": 16,         # Legend text
    "legend.title_fontsize": 18    # Legend title
})

def get_two_latest_folders(base_pattern="results_*"):
    """Finds the two most recently created result folders."""
    folders = glob.glob(base_pattern)
    if len(folders) < 2:
        return None
    # Sort by modification time, newest first
    sorted_folders = sorted(folders, key=os.path.getmtime, reverse=True)
    return sorted_folders[:2]

def load_spectral_data(target_dir, label_prefix=""):
    records = []
    

    files = glob.glob(os.path.join(target_dir, "mcmc_fixed", "*.json"))
    for f in files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                spec = data["metrics"].get("Spectral Distance (Laplacian)", 0)
                records.append({
                    "Condition": label_prefix,
                    "Strategy": "Fixed Seed (Local)",
                    "Spectral Distance": spec
                })
        except: pass


    files = glob.glob(os.path.join(target_dir, "mcmc_varied", "*.json"))
    for f in files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                spec = data["metrics"].get("Spectral Distance (Laplacian)", 0)
                records.append({
                    "Condition": label_prefix,
                    "Strategy": "Varied Seeds (Global)",
                    "Spectral Distance": spec
                })
        except: pass
            
    return pd.DataFrame(records)

def plot_comparison(folder1, folder2):
    name1 = os.path.basename(folder1)
    name2 = os.path.basename(folder2)
    
    print(f"[*] Loading data from: {name1}")
    df1 = load_spectral_data(folder1, label_prefix="Experiment A (Recent)")
    
    print(f"[*] Loading data from: {name2}")
    df2 = load_spectral_data(folder2, label_prefix="Experiment B (Older)")
    
    if df1.empty or df2.empty:
        print("Error: One of the folders has no data.")
        return


    full_df = pd.concat([df1, df2], ignore_index=True)


    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    

    palette = {"Fixed Seed (Local)": "#36A2EB", "Varied Seeds (Global)": "#FF9F40"}


    sns.kdeplot(
        data=df2, x="Spectral Distance", hue="Strategy", 
        fill=True, palette=palette, alpha=0.4, linewidth=2, ax=axes[0]
    )
    #axes[0].set_title("n=40, m=78, D=12, CC=0.40", fontweight="bold")
    axes[0].set_ylabel("Density", fontweight="bold")
    axes[0].set_xlabel("Spectral Distance", fontweight="bold")
    

    sns.kdeplot(
        data=df1, x="Spectral Distance", hue="Strategy", 
        fill=True, palette=palette, alpha=0.4, linewidth=2, ax=axes[1]
    )
    #axes[1].set_title("n=40, m=195, D=4, CC=0.40", fontweight="bold")
    axes[1].set_xlabel("Spectral Distance", fontweight="bold")
    axes[1].set_ylabel("") # Share Y axis
    
    # Optional: clean up the automatically generated legends so they don't overlap your data
    sns.move_legend(axes[0], "upper right")
    sns.move_legend(axes[1], "upper right")

    plt.tight_layout()
    plt.savefig("constraint_comparison.pdf", dpi=300, bbox_inches="tight")
    print("\n[*] Plot saved to constraint_comparison.pdf")
    # plt.show() # Commented out for server environments, uncomment if running locally

if __name__ == "__main__":
    folders = get_two_latest_folders()
    
    if not folders:
        print("Error: Need at least 2 results folders to compare.")
        sys.exit(1)
        

    plot_comparison(folders[0], folders[1])
