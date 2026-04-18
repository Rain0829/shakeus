"""
train.py
────────
Trains a pose classifier from the CSV collected by collect_data.py.
Outputs pose_classifier.pkl and prints a classification report.

Usage:
  python3 train.py
"""

import csv
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

INPUT_CSV    = "detection/pose_data.csv"
OUTPUT_MODEL = "detection/pose_classifier.pkl"


def normalize_row(raw):
    """Hip-centered, shoulder-width-scaled. Matches classifier.py inference."""
    # landmarks are stored as flat [x0,y0,z0, x1,y1,z1, ...]
    hip_x = (raw[23*3] + raw[24*3]) / 2
    hip_y = (raw[23*3+1] + raw[24*3+1]) / 2
    scale = abs(raw[11*3] - raw[12*3]) + 1e-6
    out = []
    for i in range(33):
        out.append((raw[i*3]   - hip_x) / scale)
        out.append((raw[i*3+1] - hip_y) / scale)
        out.append(raw[i*3+2])
    return out


# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
X, y = [], []

with open(INPUT_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw = [float(row[col]) for col in reader.fieldnames if col != "label"]
        X.append(normalize_row(raw))
        y.append(row["label"])

X = np.array(X)
y = np.array(y)

labels, counts = np.unique(y, return_counts=True)
print(f"Loaded {len(X)} samples across {len(labels)} labels:")
for lbl, cnt in zip(labels, counts):
    print(f"  {lbl}: {cnt} samples")

if len(X) < 50:
    print("\nWARNING: Very few samples. Collect more data for better accuracy.")

# ─────────────────────────────────────────
#  ENCODE LABELS
# ─────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ─────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────
print("\nTraining classifier...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# ─────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n── Classification Report ──────────────────")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("── Confusion Matrix ───────────────────────")
cm = confusion_matrix(y_test, y_pred)
print(f"Labels: {list(le.classes_)}")
print(cm)

cv_scores = cross_val_score(clf, X, y_enc, cv=5)
print(f"\n5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────
with open(OUTPUT_MODEL, "wb") as f:
    pickle.dump({"classifier": clf, "label_encoder": le}, f)

print(f"\nModel saved to: {OUTPUT_MODEL}")
print("Labels order:", list(le.classes_))