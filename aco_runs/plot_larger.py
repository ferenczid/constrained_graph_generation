import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# --- GLOBALLY INCREASE FONT SIZES FOR LATEX SCALING ---
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,             # Base font size for colorbars and text
    'axes.labelsize': 20,        # X and Y axis labels
    'axes.titlesize': 24,        # Titles of the subplots
    'xtick.labelsize': 16,       # X axis tick marks
    'ytick.labelsize': 16,       # Y axis tick marks
})

def compute_diversity_metrics(json_path):

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        graphs = data.get("graphs", [])
        if len(graphs) < 2:
            return 0.0, 0.0, data['target_diam'], data['target_cc']
            
        j_sum = 0
        s_sum = 0
        pairs = list(combinations(range(len(graphs)), 2))
        
        for i, j in pairs:
            g1 = graphs[i]
            g2 = graphs[j]
            
            # Jaccard
            s1 = set(tuple(sorted(e)) for e in g1['edges'])
            s2 = set(tuple(sorted(e)) for e in g2['edges'])
            
            intersect = len(s1.intersection(s2))
            union = len(s1.union(s2))
            j_sum += (1.0 - (intersect / union))
            
            # Spectral
            evals1 = np.array(g1['eigenvalues'])
            evals2 = np.array(g2['eigenvalues'])
            spec_dist = np.linalg.norm(evals1 - evals2) / np.sqrt(data['N'])
            s_sum += spec_dist
            
        avg_jac = j_sum / len(pairs)
        avg_spec = s_sum / len(pairs)
        
        return avg_jac, avg_spec, data['target_diam'], data['target_cc']

    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def build_dataframe(folder_path):
    print(f"[*] Analyzing results in {folder_path}...")
    

    metrics_files = glob.glob(os.path.join(folder_path, "metrics", "*.csv"))
    rows = []
    
    print(f"    - Loading {len(metrics_files)} metric files...")
    for f in metrics_files:
        try:
            df_temp = pd.read_csv(f, header=None)
            rows.append(df_temp.values[0])
        except:
            pass
            
    if not rows:
        print("Error: No data found.")
        return None
        
    df = pd.DataFrame(rows, columns=["Diam", "Target_Trans", "Success", "Gap"])
    

    diversity_map = {}
    
    json_files = glob.glob(os.path.join(folder_path, "graphs", "*.json"))
    print(f"    - Computing diversity for {len(json_files)} graph sets...")
    
    for jf in json_files:
        res = compute_diversity_metrics(jf)
        if res:
            jac, spec, d, cc = res
            diversity_map[(int(d), round(float(cc), 2))] = (jac, spec)
            

    df["Jaccard"] = 0.0
    df["Spectral"] = 0.0
    
    for idx, row in df.iterrows():
        key = (int(row["Diam"]), round(float(row["Target_Trans"]), 2))
        if key in diversity_map:
            df.at[idx, "Jaccard"] = diversity_map[key][0]
            df.at[idx, "Spectral"] = diversity_map[key][1]
            
    return df

def plot_heatmaps(df, output_file):
    df["Diam"] = df["Diam"].astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    

    p1 = df.pivot(index="Diam", columns="Target_Trans", values="Success")

    sns.heatmap(p1, annot=False, cmap="RdYlGn", vmin=0, vmax=1, ax=axes[0], fmt=".2f", cbar_kws={'label': 'Ratio'})
    axes[0].set_title("Success Ratio", fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Target Diameter", fontweight='bold')
    axes[0].set_xlabel("Target Transitivity", fontweight='bold')
    

    p2 = df.pivot(index="Diam", columns="Target_Trans", values="Spectral")
    sns.heatmap(p2, annot=False, cmap="magma", vmin=0, ax=axes[1], fmt=".2f", cbar_kws={'label': 'Distance'})
    

    axes[1].set_title("Spectral Distance", fontweight='bold')
    axes[1].set_facecolor('black')
    axes[1].invert_yaxis()
    axes[1].set_ylabel("") # Left blank intentionally as requested
    axes[1].set_xlabel("Target Transitivity", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[*] Plot saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_results.py <experiment_folder> <output_filename.pdf>")
        sys.exit(1)
        
    folder = sys.argv[1]
    output_filename = sys.argv[2]
    
    df = build_dataframe(folder)
    
    if df is not None:
        plot_heatmaps(df, output_filename)
