# Assignment 2 — Location Privacy Analysis

**Course:** Cybersecurity  
**Due:** March 23, 2026 at 11:59pm  
**Script:** `assignment2_project.py`

---

## Requirements

### Python Version
Python **3.8 or higher** is required.

### Dependencies
Install all required packages with a single command:

```bash
pip install scipy numpy pandas matplotlib scikit-learn
```

| Package | Version Tested | Purpose |
|---------|---------------|---------|
| `scipy` | 1.17.0 | Loading `.mat` files |
| `numpy` | 2.4.2 | Numerical operations and rounding |
| `pandas` | 3.0.1 | Data manipulation and DataFrames |
| `matplotlib` | 3.10.8 | Generating plots |
| `scikit-learn` | 1.8.0 | Polynomial regression and K-Means |

---

## Required Datasets

The script expects the following `.mat` files. Each file contains a **336 × 5 matrix** of GPS readings sampled every 30 minutes over one week.

| File | Variable Key | Role | Rows |
|------|-------------|------|------|
| `week1.mat` | `week1` | Training | 336 |
| `week2.mat` | `week2` | Training | 336 |
| `week3.mat` | `week3` | Training | 336 |
| `week4.mat` | `week4` | Training | 336 |
| `week5.mat` | `week5` | Validation | 336 |
| `week6.mat` | `week6` | **Grading** (available after deadline) | 336 |

### Data Columns

| Column | Name | Description |
|--------|------|-------------|
| 1 | Timestamp | Hour offset within the week (0.5 → 168.0, step 0.5) |
| 2 | Latitude | GPS latitude coordinate (noisy) |
| 3 | Longitude | GPS longitude coordinate (noisy) |
| 4 | Accuracy | GPS accuracy score (0.0 – 1.0) |
| 5 | Label | Location zone ID — **the prediction target** (values: 48, 49, 50, 51, 52) |

---

## File Structure

Place all files in the same directory before running:

```
assignment2_project.py   ← main script
week1.mat
week2.mat
week3.mat
week4.mat
week5.mat
week6.mat                ← add this when available for grading
```

---

## Configuration

At the top of `assignment2_project.py`, update these variables if needed:

```python
DATA_DIR   = '/path/to/your/mat/files/'  # folder containing week1–5 .mat files
WEEK6_PATH = 'week6.mat'                 # path to week 6 file for grading
```

---

## How to Run

### Standard run (weeks 1–5)

```bash
python assignment2_project.py
```

This trains the model on weeks 1–4 and validates it on week 5.

### Grading run (week 6)

When `week6.mat` becomes available:

1. Place `week6.mat` in the same directory as the script
2. Open `assignment2_project.py` and find this block near line 174:

```python
# --- UNCOMMENT BELOW WHEN WEEK 6 IS AVAILABLE ---
# print("\n  Running predict_week() on week 6 (grading)...")
# results_w6, acc_w6 = predict_week(WEEK6_PATH, 'week6')
# results_w6.to_csv('week6_predictions.csv', index=False)
# print(f"\n  Week 6 predictions saved to week6_predictions.csv")
```

3. Remove the `#` from those four lines
4. Run the script again:

```bash
python assignment2_project.py
```

---

## Expected Output

### Console output

Running the script prints the following sections in order:

#### Setup
```
============================================================
SETUP: Loading data
============================================================
  week1.mat loaded — 336 rows, labels: [48.0, 49.0, 50.0, 51.0, 52.0]
  week2.mat loaded — 336 rows, labels: [48.0, 49.0, 50.0, 51.0, 52.0]
  week3.mat loaded — 336 rows, labels: [48.0, 49.0, 50.0, 51.0]
  week4.mat loaded — 336 rows, labels: [48.0, 49.0, 50.0, 51.0, 52.0]
  week5.mat loaded — 336 rows, labels: [48.0, 49.0, 50.0, 51.0, 52.0]

  Training rows  : 1344  (weeks [1, 2, 3, 4])
  Validation rows: 336   (week 5)
```

#### Q1 — Polynomial Regression
```
============================================================
Q1: Polynomial Regression
============================================================

  Training model (PolynomialFeatures deg=4 + LinearRegression)...
  Model trained ✓

  Week 5 Validation Accuracy : 97.02%
  Mean Confidence Score      : 84.96%

  Per-label accuracy (week 5):
    Label 48: 100.0%
    Label 49: 0.0%
    Label 50: 96.4%
    Label 51: 100.0%
    Label 52: 100.0%

  Running predict_week() on week 5 to verify...

  Results for week5:
    Overall Accuracy  : 97.02%
    Mean Confidence   : 84.96%

    Sample (first 10 rows):
     Timestamp |  True |  Pred |   Conf | Match
    ------------------------------------------------
           0.5 |    48 |    48 | 0.9452 | ✓
           1.0 |    48 |    48 | 0.8420 | ✓
           ...
```

#### Q2 — K-Means Clustering
```
============================================================
Q2: K-Means Clustering
============================================================

  Using all 1680 rows (5 weeks) for clustering
  Features: Latitude, Longitude only (as instructed)

  Running elbow method for K = 1 to 10...
    K= 1: inertia = 1960.38
    K= 2: inertia = 98.54
    ...
    K= 5: inertia = 25.80
    ...
    K=10: inertia = 12.16

  Optimal K from elbow method: 5

  Fitting KMeans(n_clusters=5)...
  K-Means fitted ✓

  Cluster centroids:
    Cluster 0: Lat=44.2678, Lon=-70.1024  (127 points)
    Cluster 1: Lat=42.3434, Lon=-71.8972  (1130 points)
    Cluster 2: Lat=43.7374, Lon=-70.1780  (147 points)
    Cluster 3: Lat=44.2813, Lon=-70.6109  (141 points)
    Cluster 4: Lat=43.7829, Lon=-70.6885  (135 points)
```

#### Final summary
```
============================================================
COMPLETE
  Q1 Validation Accuracy (Week 5) : 97.02%
  Q2 Optimal K                    : 5
============================================================
```

### Generated files

| File | Description |
|------|-------------|
| `assignment2_results.png` | Three-panel plot: elbow curve, cluster scatter, regression output histogram |
| `week6_predictions.csv` | Per-row predictions for week 6 (only generated when week 6 block is uncommented) |

### `week6_predictions.csv` format (when generated)

| Column | Description |
|--------|-------------|
| `Timestamp` | Hour offset within the week |
| `True_Label` | Actual label from the `.mat` file |
| `Predicted_Label` | Model's predicted label (rounded integer) |
| `Confidence` | Score 0–1: how far the raw output was from the rounding boundary |
| `Correct` | `True` / `False` — whether the prediction matched |

---

## Confidence Score Explained

The model outputs a continuous value (e.g. `47.94`) which is rounded to the nearest integer label. The confidence score measures how far that raw value is from the decision boundary:

```
Confidence = 1 - |raw_prediction - rounded_prediction|
```

| Raw Output | Rounded To | Confidence | Interpretation |
|-----------|------------|------------|----------------|
| 47.95 | 48 | 0.95 | Very certain |
| 50.12 | 50 | 0.88 | Confident |
| 48.50 | 49 | 0.50 | On the boundary |

---

## Troubleshooting

**`FileNotFoundError: week1.mat not found`**  
Update `DATA_DIR` in the configuration section at the top of the script to point to the folder containing your `.mat` files.

**`KeyError: 'week1'`**  
The variable name inside the `.mat` file does not match. Verify using:
```python
import scipy.io as sio
mat = sio.loadmat('week1.mat')
print(mat.keys())
```

**`ModuleNotFoundError`**  
Run `pip install scipy numpy pandas matplotlib scikit-learn` and try again.

**Plots not displaying**  
The script uses `matplotlib.use('Agg')` which saves plots to a file instead of opening a window. Check the directory for `assignment2_results.png`.
