# Assignment 2 — Location Privacy Analysis

**Course:** DS557 - Machine Learning in Cybersecurity  
**Script:** `cmcourville_assignment2.py`

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
| `week6.mat` | `week6` | **Grading** | 336 |

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

Place all data files in the `/data` folder:

```
assignment2_project.py   ← main script
/data/
  week1.mat
  week2.mat
  week3.mat
  week4.mat
  week5.mat
  week6.mat              ← Available after deadline
```

---

## Configuration

The script accepts a command-line argument to specify the data directory:

```python
--data-dir PATH  : Directory containing .mat files (default: data/)
```

If no `--data-dir` is specified, it defaults to the `data/` folder.

---

## How to Run

### Basic Usage

```bash
python assignment2_project.py [--data-dir PATH]
```

**Options:**
- `--data-dir PATH`: Specify the directory containing .mat files (default: `data/`)

**Examples:**
```bash
# Use default data/ folder
python assignment2_project.py

# Specify custom data directory
python assignment2_project.py --data-dir /path/to/data/

# Use relative path
python assignment2_project.py --data-dir ./datasets/
```

### What the Script Does

The script automatically:
1. **Loads training data** (weeks 1–4) and **validation data** (week 5)
2. **Trains a polynomial regression model** to predict location zones
3. **Validates the model** on week 5 data
4. **Runs K-Means clustering** to find natural location clusters
5. **Generates plots** and saves them as `assignment2_results.png`
6. **Automatically detects and processes week6.mat** if present (for grading)

### Design Decisions

**Handling noisy data:** The GPS coordinates in this dataset contain noisy 
readings where the recorded location occasionally deviates from the user's 
actual zone. Polynomial regression handles this naturally because the continuous 
output smooths over noisy readings before rounding corrects them to the 
nearest valid label.

**Feature choice — Latitude & Longitude:** The assignment asks to estimate 
the label for each timestamp, where timestamp refers to each individual row 
of data rather than the time value itself as a model input. Latitude and 
Longitude are used as input features because they directly encode which 
geographic zone the user is in, making them the most predictive features 
available in the dataset.

### Grading with Week 6

When `week6.mat` becomes available after the deadline:

1. Place `week6.mat` in your data directory (same folder as week1.mat–week5.mat)
2. Run the script normally — it will automatically detect and process week6.mat
3. The script will save predictions to `week6_predictions.csv` in the current working directory

**No manual code changes required!** The script handles week 6 detection automatically.

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
| `week6_predictions.csv` | Per-row predictions for week 6 (automatically generated when week6.mat is present in the data directory) |

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
