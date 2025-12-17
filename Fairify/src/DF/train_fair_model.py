import sys
import os

import random
import numpy as np
import tensorflow as tf

def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  

set_all_seeds(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_dir)
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score
from utils.verif_utils import *
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

ORIGINAL_MODEL_NAME = "DF-5"        
FAIRER_MODEL_NAME = "DF-5-Retrained"
learning_rate = 0.00001

# DF-1 
# DF-2 0.005
# DF-3 0.005
# DF-4 0.005
# DF-5 0.005

print("Loading original model...")
original_model = load_model(f'Fairify/models/default/{ORIGINAL_MODEL_NAME}.h5')
print(original_model.summary())

# Load Default dataset
df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_default()

print(f"X_train_orig shape: {X_train_orig.shape}")

print("\nLoading synthetic counterexamples...")
df_synthetic = pd.read_csv(f'Fairify/counterexamples/DF/counterexamples-{ORIGINAL_MODEL_NAME}.csv')
print(f"Loaded {len(df_synthetic)} counterexamples")

# Rename decision column
if 'decision' in df_synthetic.columns:
    df_synthetic.rename(columns={'decision': 'default.payment.next.month'}, inplace=True)

label_name = 'default.payment.next.month'

# Map categorical string values to numeric
print("\nMapping categorical values to numeric...")

# Map PAY columns: Map to simplified 3-level system (0=on-time, 1=slight delay, 2=severe delay)
pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for col in pay_cols:
    if col in df_synthetic.columns:
        # If string values exist, map them to numeric
        if df_synthetic[col].dtype == 'object':
            # Assuming CE generation might use descriptive labels
            df_synthetic[col] = df_synthetic[col].map({
                'On-time': 0, 
                'Slight-delay': 1, 
                'Severe-delay': 2,
                '0': 0, '1': 1, '2': 2  # Also handle string numbers
            })
        df_synthetic[col] = pd.to_numeric(df_synthetic[col], errors='coerce')

# Map SEX: ordinal encoding (0 or 1)
if 'SEX' in df_synthetic.columns:
    if df_synthetic['SEX'].dtype == 'object':
        df_synthetic['SEX'] = df_synthetic['SEX'].map({'Male': 0, 'Female': 1})
    df_synthetic['SEX'] = pd.to_numeric(df_synthetic['SEX'], errors='coerce')

# Map EDUCATION: ordinal encoding (0-3)
if 'EDUCATION' in df_synthetic.columns:
    if df_synthetic['EDUCATION'].dtype == 'object':
        # Map any string representations to ordinal values
        edu_map = {'Edu-0': 0, 'Edu-1': 1, 'Edu-2': 2, 'Edu-3': 3}
        df_synthetic['EDUCATION'] = df_synthetic['EDUCATION'].map(edu_map)
    df_synthetic['EDUCATION'] = pd.to_numeric(df_synthetic['EDUCATION'], errors='coerce')
    df_synthetic['EDUCATION'] = np.clip(df_synthetic['EDUCATION'], 0, 3)

# Map MARRIAGE: ordinal encoding (0-2)
if 'MARRIAGE' in df_synthetic.columns:
    if df_synthetic['MARRIAGE'].dtype == 'object':
        # Map any string representations to ordinal values
        mar_map = {'Mar-0': 0, 'Mar-1': 1, 'Mar-2': 2}
        df_synthetic['MARRIAGE'] = df_synthetic['MARRIAGE'].map(mar_map)
    df_synthetic['MARRIAGE'] = pd.to_numeric(df_synthetic['MARRIAGE'], errors='coerce')
    df_synthetic['MARRIAGE'] = np.clip(df_synthetic['MARRIAGE'], 0, 2)

# Convert all continuous feature columns to numeric (they should already be binned 0-4)
continuous_features = [
    'LIMIT_BAL', 'AGE', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]
for col in continuous_features:
    if col in df_synthetic.columns:
        df_synthetic[col] = pd.to_numeric(df_synthetic[col], errors='coerce')

print(f"NaN counts per column after mapping:")
print(df_synthetic.isna().sum().sum())

# Drop rows with ANY NaN
df_synthetic.dropna(inplace=True)
print(f"After dropping NaN: {len(df_synthetic)} counterexamples")

if len(df_synthetic) == 0:
    print("\nERROR: All counterexamples were dropped!")
    sys.exit(1)

# Extract features and labels
X_synthetic = df_synthetic.drop(columns=[label_name]).values.astype(np.float32)
y_synthetic = df_synthetic[label_name].values.astype(np.float32)

print(f"X_synthetic shape: {X_synthetic.shape}")

# Verify shapes match
if X_synthetic.shape[1] != X_train_orig.shape[1]:
    print(f"ERROR: Feature mismatch - counterexamples have {X_synthetic.shape[1]} features but training has {X_train_orig.shape[1]}")
    sys.exit(1)

# Create training pairs from counterexamples
X_train_ce = []
y_train_ce = []
for i in range(0, len(X_synthetic)-1, 2):
    x = X_synthetic[i]
    x_prime = X_synthetic[i+1]
    label = max(y_synthetic[i], y_synthetic[i+1])
   
    X_train_ce.append(x)
    X_train_ce.append(x_prime)
    y_train_ce.append(label)
    y_train_ce.append(label)

X_train_ce = np.array(X_train_ce, dtype=np.float32)
y_train_ce = np.array(y_train_ce, dtype=np.float32)

print(f"X_train_ce shape: {X_train_ce.shape}")

# Ensure original data is float32
X_train_orig = X_train_orig.astype(np.float32)
y_train_orig = y_train_orig.astype(np.float32)

# Combine original training data with counterexamples
X_train_mixed = np.vstack([X_train_orig, X_train_ce])
y_train_mixed = np.hstack([y_train_orig, y_train_ce])

print(f"X_train_mixed shape: {X_train_mixed.shape}")

# Compile and retrain
original_model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])

epochs = 5
iterations = 1
print(f"\nTraining for {iterations} iterations...")
for iteration in range(iterations):
    print(f"\nIteration {iteration+1}/{iterations}")
    original_model.fit(X_train_mixed, y_train_mixed,
                      epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)

original_model.save(f'Fairify/models/default/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Model saved as {FAIRER_MODEL_NAME}.h5")