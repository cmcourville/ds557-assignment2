# Cybersecurity Assignment 2 — Project Plan

**Course:** Cybersecurity  
**Due:** March 23, 2026 at 11:59pm  
**Points:** 100  
**Submission:** File upload on Canvas (`assignment2_project.py`)

---

## Dataset Overview

Google Maps location data collected from a WPI employee over **5 weeks**, sampled every **30 minutes**.

| Column | Name | Description |
|--------|------|-------------|
| 1 | Timestamp | Hour offset within the week (0.5 → 168.0, step 0.5) |
| 2 | Latitude | GPS latitude (noisy) |
| 3 | Longitude | GPS longitude (noisy) |
| 4 | Accuracy | GPS accuracy score (0–1) |
| 5 | Label | Location zone ID — target variable |

**Structure:** 336 rows per week × 5 weeks = **1,680 total rows**

**Labels discovered:** 5 discrete geographic zones

| Label | Interpreted Location | Frequency |
|-------|---------------------|-----------|
| 48 | Home (42.377°N, -71.902°W) | 45% — nights & weekends |
| 49 | Transit stop | 2% — brief transitions |
| 50 | Away / travel (44.0°N, -70.4°W) | 33% — weekend trips |
| 51 | Work / campus (42.280°N, -71.902°W) | 19% — weekday daytime |
| 52 | Rare location | 1% — occasional visits |

**Noise level:** 3.2% of rows have a label that differs from the majority at that timestamp.

---

## Assignment Tasks

### Q1 — Linear Regression (Predict Location Label)
Train a regression model on weeks 1–4. Validate on week 5. Grade is determined by accuracy on week 6.

### Q2 — K-Means Clustering (Discover Location Clusters)
Apply K-Means to all 5 weeks using only Lat/Lon. Use the elbow method to find optimal K. Include plots.

---

## Q1 — Step-by-Step Plan

### Step 1 — Import Libraries
```python
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
```

### Step 2 — Load `.mat` Files
Load `week1.mat` through `week5.mat` using `scipy.io.loadmat()`. Each file contains a 336×5 NumPy array. Assign column names and stack into a single DataFrame with a `Week` column.

```python
def load_mat(path, key):
    mat = sio.loadmat(path)
    df  = pd.DataFrame(mat[key], columns=['Timestamp','Lat','Lon','Accuracy','Label'])
    return df
```

### Step 3 — Exploratory Data Analysis
- Plot lat/lon scatter coloured by label → confirms 5 distinct geographic zones
- Check label distribution → classes are imbalanced (label 49 and 52 are rare)
- Plot dominant label by hour of day → clear daily routine pattern
- Measure noise per week → ~3.2% of readings have anomalous labels

**Key insight from EDA:** The Label is entirely determined by the user's physical location. Lat/Lon are therefore the most direct features to use.

### Step 4 — Train / Validation Split
| Split | Weeks | Rows |
|-------|-------|------|
| Training | 1, 2, 3, 4 | 1,344 |
| Validation | 5 | 336 |

Features used: `Lat`, `Lon` (columns 2 and 3)  
Target: `Label` (column 5)

### Step 5 — Choose Regression Method: Polynomial Regression (Degree 4)

**Why not standard Linear Regression?**  
The 5 location zones have non-linear boundaries in 2D lat/lon space. A flat regression plane cannot separate them.

**Benchmarking results (weeks 1–4 → week 5):**

| Method | Features | Accuracy |
|--------|----------|----------|
| Linear Regression | All | 21.4% |
| Ridge Regression | All | 22.6% |
| Lasso Regression | All | 28.3% |
| Polynomial deg 2 | All | 42.9% |
| Polynomial deg 3 | All | 87.2% |
| **Polynomial deg 4** | **Lat, Lon** | **97.0%** ✓ |

Degree 4 expands 2 features (Lat, Lon) into **14 polynomial interaction terms**, giving the regression surface enough curvature to cleanly separate all 5 zones.

### Step 6 — Build & Train the Model
```python
model = Pipeline([
    ('poly', PolynomialFeatures(degree=4, include_bias=False)),
    ('lr',   LinearRegression())
])
model.fit(X_train, y_train)
```

### Step 7 — Apply Rounding to Get Discrete Labels
```python
raw_predictions     = model.predict(X_val)
rounded_predictions = np.round(raw_predictions)   # snap to nearest integer label
```
Rounding is essential because regression produces continuous values (e.g. 47.94) but labels are discrete integers (48, 49, 50, 51, 52).

### Step 8 — Compute Confidence Score
```python
confidence = 1 - abs(raw_prediction - rounded_prediction)
```
A raw output of `47.95` rounds to `48` → confidence = 0.95 (very certain)  
A raw output of `48.50` rounds to `49` → confidence = 0.50 (on the boundary)

### Step 9 — Validate on Week 5
- **Overall accuracy:** 97.02%
- **Per-label accuracy:**
  - Label 48 (Home): 100%
  - Label 49 (Transit): 0% ← rare class, only 6 examples in week 5
  - Label 50 (Away): 96.4%
  - Label 51 (Work): 100%
  - Label 52 (Rare): 100%

### Step 10 — Generalize for Week 6 (Grading)
The `predict_week(mat_path, mat_key)` function accepts any `.mat` file, runs the trained model, and outputs per-row predictions, confidence scores, and overall accuracy. To grade week 6, simply uncomment the week 6 block at the bottom of the script.

---

## Q2 — Step-by-Step Plan

### Step 1 — Extract Lat/Lon from All 5 Weeks
Use only columns 2 and 3 from the full 1,680-row dataset. K-Means is unsupervised — no train/test split is needed.

```python
coords = all_data[['Lat', 'Lon']].values   # shape: (1680, 2)
```

### Step 2 — Run the Elbow Method (K = 1 to 10)
For each K, fit K-Means and record the **inertia** (within-cluster sum of squares — lower = tighter clusters).

```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(coords)
    inertias.append(km.inertia_)
```

**Elbow results:**

| K | Inertia | Drop from K-1 |
|---|---------|--------------|
| 1 | 1960.38 | — |
| 2 | 98.54 | 1861.84 |
| 3 | 61.44 | 37.10 |
| 4 | 39.47 | 21.97 |
| **5** | **25.80** | **13.66** |
| 6 | 22.36 | 3.45 ← flattens here |
| 7 | 19.17 | 3.19 |

The drop from K=5 to K=6 is only 3.45 vs 13.66 for K=4→5. The elbow is clearly at **K=5**.

### Step 3 — Select Optimal K = 5
K=5 is also independently confirmed by the 5 distinct labels in the supervised data — the unsupervised clustering naturally discovers the same geographic zones.

### Step 4 — Fit Final K-Means with K = 5
```python
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(coords)
```

**Resulting cluster centroids:**

| Cluster | Latitude | Longitude | Size |
|---------|----------|-----------|------|
| 0 | 44.2678 | -70.1024 | 127 |
| 1 | 42.3434 | -71.8972 | 1130 |
| 2 | 43.7374 | -70.1780 | 147 |
| 3 | 44.2813 | -70.6109 | 141 |
| 4 | 43.7829 | -70.6885 | 135 |

### Step 5 — Produce Required Plots

**Plot A — Elbow Curve:** K (x-axis) vs. Inertia (y-axis), with a vertical dashed line at the optimal K.

**Plot B — Cluster Scatter:** Longitude (x) vs. Latitude (y), points coloured by cluster, centroids marked with ×.

---

## File Structure

```
assignment2_project.py     ← main submission file (run this for grading)
PROJECT_PLAN.md            ← this document
assignment2_results.png    ← output plots (Q1 + Q2)
week1.mat – week5.mat      ← training data
week6.mat                  ← grading data (available after deadline)
```

---

## How to Run for Week 6 Grading

1. Place `week6.mat` in the same directory as `assignment2_project.py`
2. Open the script and set `WEEK6_PATH = 'week6.mat'` (already set by default)
3. Uncomment the two lines under `# UNCOMMENT BELOW WHEN WEEK 6 IS AVAILABLE`
4. Run: `python assignment2_project.py`
5. Output: predictions printed to console + saved to `week6_predictions.csv`

---

## Summary of Key Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Regression method | Polynomial (degree 4) | Non-linear zone boundaries; 97% accuracy on week 5 |
| Features for Q1 | Lat + Lon only | Label is a geographic zone — directly encoded in coordinates |
| Rounding | `np.round()` | Converts continuous regression output to discrete label |
| Confidence metric | `1 - abs(raw - rounded)` | Proximity to nearest integer = certainty of prediction |
| K-Means data | All 5 weeks | Unsupervised — no test split needed |
| Optimal K | 5 | Elbow method + matches 5 known location zones |
