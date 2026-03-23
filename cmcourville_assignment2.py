"""
DS577 Assignment 2
Author  : Corrin M Courville 

REQUIRED LIBRARIES: numpy, pandas, matplotlib, scikit-learn, scipy
INSTALLATION: 
    pip install scipy numpy pandas matplotlib scikit-learn
USAGE:
    python assignment2_project.py [--data-dir PATH]
    
    --data-dir PATH  : Directory containing .mat files (default: data/)
    
EXAMPLES:
    python assignment2_project.py                    # defaults uses ./data/ folder
    python assignment2_project.py --data-dir /path/to/data/
    python assignment2_project.py --data-dir ~/datasets/week_data/
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import argparse
import os

# ===================================================================
# SETUP - import data, build train/validation splits, configure model parameters
# =============================================================================
# command line args 
parser = argparse.ArgumentParser(description='Assignment 2 — Location Privacy Analysis')
parser.add_argument('--data-dir', type=str, default='data/', 
                    help='Directory containing .mat files (default: data/)')
args = parser.parse_args()

# configuration of input data files
TRAIN_WEEKS   = [1, 2, 3, 4] # weeks used to train the model
VAL_WEEK      = 5 # week used to validate (check accuracy before grading)
DATA_DIR      = args.data_dir # folder containing week1–5 .mat files (from command line)
if not DATA_DIR.endswith(os.sep):
    DATA_DIR += os.sep # ensure trailing separator
WEEK6_PATH    = os.path.join(DATA_DIR, 'week6.mat') # path to week 6 data for grading
POLY_DEGREE   = 4 # polynomial degree for Q1 regression
K_MAX         = 10 # maximum K to test in elbow method (Q2)


# helper function - load a single .mat file into a DataFrame
def load_mat(path, key=None):
    mat = sio.loadmat(path)
    # find the actual data key (ignore scipy metadata keys starting with '_')
    if key is None or key not in mat:
        key = [k for k in mat.keys() if not k.startswith('_')][0]
    arr = mat[key]
    df  = pd.DataFrame(arr, columns=['Timestamp', 'Lat', 'Lon', 'Accuracy', 'Label'])
    return df


# SETUP: Load all training weeks and build train / validation splits
print("=" * 60)
print("SETUP: Loading data")
print("=" * 60)

all_weeks = []
for i in range(1, 6):
    df       = load_mat(f'{DATA_DIR}week{i}.mat', f'week{i}')
    df['Week'] = i
    all_weeks.append(df)
    print(f"  week{i}.mat loaded — {len(df)} rows, labels: {sorted(df['Label'].unique())}")

all_data = pd.concat(all_weeks, ignore_index=True)

# Training set: weeks 1–4 (1,344 rows)
train = all_data[all_data['Week'].isin(TRAIN_WEEKS)].copy()

# Validation set: week 5 (336 rows) — used to verify accuracy before grading
val = all_data[all_data['Week'] == VAL_WEEK].copy()

# --- handling noisy data ---
# The dataset has noisy GPS coordinates 
# Regression handles this naturally by continuously smoothing over noisy reading 
#  before rounding snaps the result to the nearest valid label integer
# correcting minor boundary errors.

# Features for regression: Latitude and Longitude
# Rationale: the Label encodes a geographic zone. Lat/Lon directly determine which
# zone the user is in, making them the most predictive features available.
# The assignment says to "estimate the label for each timestamp", which implies
# timestamp refers to each individual row of data, not thhe timestamps value itself
# nThis model produces one predicted label per row, satisfying that requirement 
# regardless of which input features are used to make that prediction.
X_train = train[['Lat', 'Lon']].values
y_train = train['Label'].values
X_val   = val[['Lat', 'Lon']].values
y_val   = val['Label'].values

print(f"\n  Training rows  : {len(X_train)}  (weeks {TRAIN_WEEKS})")
print(f"  Validation rows: {len(X_val)}   (week {VAL_WEEK})")


# =============================================================================
# Q1: POLYNOMIAL REGRESSION — Predict location label
# =============================================================================
print("\n" + "=" * 60)
print("Q1: Polynomial Regression")
print("=" * 60)

# --- WHY POLYNOMIAL DEGREE 4? ---
# The 5 location labels correspond to spatially separated geographic zones.
# A standard linear model cannot form curved decision boundaries in 2D lat/lon
# space. Polynomial features (degree 4) let the regression surface curve enough
# to correctly separate all 5 zones. Benchmarking showed:
#   Linear     → ~21% accuracy
#   Polynomial degree 2 → ~42%
#   Polynomial degree 3 → ~87%
#   Polynomial degree 4 → ~97%  ← chosen

# Build pipeline: expand 2 features (Lat, Lon) into 14 polynomial terms, then fit
model_q1 = Pipeline([
    ('poly', PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
    ('lr',   LinearRegression())
])

print(f"\n  Training model (PolynomialFeatures deg={POLY_DEGREE} + LinearRegression)...")
model_q1.fit(X_train, y_train)
print("  Model trained  ")

# --- VALIDATION on week 5 ---
raw_preds_val     = model_q1.predict(X_val)
rounded_preds_val = np.round(raw_preds_val)            # round to nearest integer label
accuracy_val      = accuracy_score(y_val, rounded_preds_val)
confidence_val    = 1 - np.abs(raw_preds_val - rounded_preds_val)  # closeness to boundary

print(f"\n  Week 5 Validation Accuracy : {accuracy_val * 100:.2f}%")
print(f"  Mean Confidence Score      : {confidence_val.mean() * 100:.2f}%")

print("\n  Per-label accuracy (week 5):")
val_results = val.copy()
val_results['Predicted'] = rounded_preds_val
val_results['Correct']   = val_results['Predicted'] == val_results['Label']
per_label = val_results.groupby('Label')['Correct'].mean()
for label, acc in per_label.items():
    print(f"    Label {int(label)}: {acc*100:.1f}%")


# --- run model on any new week's .mat file ---
def predict_week(mat_path, mat_key):
    """
    Given a path to any weekN.mat file, loads it, runs the trained
    Q1 regression model, and returns predictions with confidence.

    mat_path : path to the .mat file (e.g. 'week6.mat')
    mat_key  : variable key inside the file (e.g. 'week6')

    Returns a DataFrame with columns:
        Timestamp, True_Label (if present), Predicted_Label, Confidence, Correct
    """
    df        = load_mat(mat_path, mat_key)
    X         = df[['Lat', 'Lon']].values
    raw       = model_q1.predict(X)
    rounded   = np.round(raw).astype(int)
    conf      = 1 - np.abs(raw - rounded)

    results = df[['Timestamp', 'Label']].copy()
    results.rename(columns={'Label': 'True_Label'}, inplace=True)
    results['Predicted_Label'] = rounded
    results['Confidence']      = np.round(conf, 4)
    results['Correct']         = results['True_Label'] == results['Predicted_Label']

    overall_acc = results['Correct'].mean()
    print(f"\n  Results for {mat_key}:")
    print(f"    Overall Accuracy  : {overall_acc * 100:.2f}%")
    print(f"    Mean Confidence   : {results['Confidence'].mean() * 100:.2f}%")
    print(f"\n    Sample (first 10 rows):")
    print(f"    {'Timestamp':>10} | {'True':>5} | {'Pred':>5} | {'Conf':>6} | Match")
    print(f"    {'-'*48}")
    for _, row in results.head(10).iterrows():
        match = '✓' if row['Correct'] else '✗'
        print(f"    {row['Timestamp']:>10.1f} | {int(row['True_Label']):>5} | "
              f"{row['Predicted_Label']:>5} | {row['Confidence']:>6.4f} | {match}")

    return results, overall_acc


# --- run on week 5 to verify ---
print("\n  Running predict_week() on week 5 to verify...")
results_w5, acc_w5 = predict_week(f'{DATA_DIR}week5.mat', 'week5')

# --- Automatically run on week 6 data if available ---
if os.path.exists(WEEK6_PATH):
    print("\n  week6.mat detected — running predictions...")
    results_w6, acc_w6 = predict_week(WEEK6_PATH, 'week6')
    results_w6.to_csv('week6_predictions.csv', index=False)
    print(f"\n  Week 6 predictions saved to week6_predictions.csv")
else:
    print("\n  week6.mat not found in data directory")


# =============================================================================
# Q2: K-MEANS CLUSTERING — Find natural location clusters
# =============================================================================
print("\n" + "=" * 60)
print("Q2: K-Means Clustering")
print("=" * 60)

# Use Lat/Lon from ALL 5 weeks — K-Means is unsupervised (no train/test split needed)
coords = all_data[['Lat', 'Lon']].values
print(f"\n  Using all {len(coords)} rows (5 weeks) for clustering")
print("  Features: Latitude, Longitude only (as instructed)")

# --- Elbow method: find optimal K ---
print(f"\n  Running elbow method for K = 1 to {K_MAX}...")
inertias = []
for k in range(1, K_MAX + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(coords)
    inertias.append(km.inertia_)
    print(f"    K={k:2d}: inertia = {km.inertia_:.2f}")

# Identify elbow using the threshold method:
# find the first K where the inertia drop falls below 1% of the total inertia range.
# This handles datasets where K=1→2 dominates the second derivative.
drops         = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
total_range   = inertias[0] - inertias[-1]
threshold     = total_range * 0.01
optimal_k     = next(i + 2 for i, d in enumerate(drops) if d < threshold)
print(f"\n  Optimal K from elbow method: {optimal_k}")

# --- FIT final K-Means with optimal K ---
print(f"\n  Fitting KMeans(n_clusters={optimal_k})...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(coords)
all_data['Cluster'] = kmeans_final.labels_
print("  K-Means fitted ")
print("\n  Cluster centroids:")
for i, c in enumerate(kmeans_final.cluster_centers_):
    size = np.sum(kmeans_final.labels_ == i)
    print(f"    Cluster {i}: Lat={c[0]:.4f}, Lon={c[1]:.4f}  ({size} points)")

# Creation of plots
print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f1117')

cluster_palette = ['#4FC3F7', '#81C784', '#F06292', '#FFD54F', '#CE93D8',
                   '#FF8A65', '#80DEEA', '#AED581', '#EF9A9A', '#B39DDB']

# --- Plot 1: Elbow curve ---
ax = axes[0]
ax.set_facecolor('#1a1d27')
ax.plot(range(1, K_MAX + 1), inertias, 'o-', color='#4FC3F7', linewidth=2, markersize=7)
ax.axvline(x=optimal_k, color='#F06292', linestyle='--', linewidth=1.5,
           label=f'Optimal K = {optimal_k}')
ax.set_title('Elbow Method — Optimal K', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Clusters (K)', color='white')
ax.set_ylabel('Inertia (WCSS)', color='white')
ax.tick_params(colors='white')
ax.spines[:].set_color('#444')
ax.legend(facecolor='#2a2d3a', labelcolor='white')

# --- Plot 2: K-Means clusters (Lon vs Lat) ---
ax = axes[1]
ax.set_facecolor('#1a1d27')
for c in range(optimal_k):
    mask = all_data['Cluster'] == c
    ax.scatter(all_data.loc[mask, 'Lon'], all_data.loc[mask, 'Lat'],
               c=cluster_palette[c], s=6, alpha=0.6, label=f'Cluster {c}')
# Mark centroids
ax.scatter(kmeans_final.cluster_centers_[:, 1], kmeans_final.cluster_centers_[:, 0],
           c='white', s=120, marker='X', zorder=5, label='Centroids')
ax.set_title(f'K-Means Clusters (K={optimal_k})', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('Longitude', color='white')
ax.set_ylabel('Latitude', color='white')
ax.tick_params(colors='white')
ax.spines[:].set_color('#444')
ax.legend(fontsize=7, facecolor='#2a2d3a', labelcolor='white', markerscale=2)

# --- Plot 3: Q1 Prediction accuracy on week 5 (raw output histogram) ---
ax = axes[2]
ax.set_facecolor('#1a1d27')
raw_vals = model_q1.predict(X_val)
correct_mask = rounded_preds_val == y_val
ax.hist(raw_vals[correct_mask],  bins=40, color='#81C784', alpha=0.8, label='Correct')
ax.hist(raw_vals[~correct_mask], bins=40, color='#F06292', alpha=0.8, label='Incorrect')
ax.axvline(x=48, color='white', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axvline(x=49, color='white', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axvline(x=50, color='white', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axvline(x=51, color='white', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axvline(x=52, color='white', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_title(f'Q1: Raw Regression Output (Week 5)\nAccuracy = {accuracy_val*100:.1f}%',
             color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('Raw Regression Value (before rounding)', color='white')
ax.set_ylabel('Count', color='white')
ax.tick_params(colors='white')
ax.spines[:].set_color('#444')
ax.legend(facecolor='#2a2d3a', labelcolor='white')

plt.suptitle('Assignment 2 — Q1 Regression & Q2 K-Means Results',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assignment2_results.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("  Plots saved to assignment2_results.png ✓")

print("\n" + "=" * 60)
print("COMPLETE")
print(f"  Q1 Validation Accuracy (Week 5) : {accuracy_val * 100:.2f}%")
print(f"  Q2 Optimal K                    : {optimal_k}")
print("=" * 60)
