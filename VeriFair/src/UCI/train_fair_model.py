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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

ORIGINAL_MODEL_NAME = "UCI-5"        
FAIRER_MODEL_NAME = "UCI-5-Retrained"
learning_rate = 0.002  # Adjust as needed

print("Loading original model...")
original_model = load_model(f'Fairify/models/uci/{ORIGINAL_MODEL_NAME}.h5')
print(original_model.summary())

def load_student():
    """
    Load and preprocess Student Performance dataset with discretization for easier CE generation.
    
    Returns:
        df, X_train, y_train, X_test, y_test, binners
    """
    file_path = 'Fairify/data/uci/student-mat.csv'
    print("Loading student performance (mat)...")
    
    df = pd.read_csv(file_path, sep=';')
    
    # Bin age into 2 bins (0, 1)
    age_binner = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
    df['age'] = age_binner.fit_transform(df[['age']]).astype(int).flatten()
    
    # Bin absences into 5 bins (0-4)
    absence_binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    df['absences'] = absence_binner.fit_transform(df[['absences']]).astype(int).flatten()
    
    # Binary encoding
    df['sex'] = (df['sex'] == 'M').astype(int)  # 0=F, 1=M
    df['school'] = (df['school'] == 'GP').astype(int)  # 0=MS, 1=GP
    df['address'] = (df['address'] == 'U').astype(int)  # 0=R, 1=U
    df['famsize'] = (df['famsize'] == 'GT3').astype(int)  # 0=LE3, 1=GT3
    df['Pstatus'] = (df['Pstatus'] == 'T').astype(int)  # 0=A, 1=T
    df['schoolsup'] = (df['schoolsup'] == 'yes').astype(int)
    df['famsup'] = (df['famsup'] == 'yes').astype(int)
    df['paid'] = (df['paid'] == 'yes').astype(int)
    df['activities'] = (df['activities'] == 'yes').astype(int)
    df['nursery'] = (df['nursery'] == 'yes').astype(int)
    df['higher'] = (df['higher'] == 'yes').astype(int)
    df['internet'] = (df['internet'] == 'yes').astype(int)
    df['romantic'] = (df['romantic'] == 'yes').astype(int)
    
    # Categorical to ordinal encoding
    reason_map = {'home': 0, 'reputation': 1, 'course': 2, 'other': 3}
    df['reason'] = df['reason'].map(reason_map)
    
    guardian_map = {'mother': 0, 'father': 1, 'other': 2}
    df['guardian'] = df['guardian'].map(guardian_map)
    
    # Job encoding (nominal -> ordinal for simplicity)
    job_map = {'at_home': 0, 'services': 1, 'other': 2, 'teacher': 3, 'health': 4}
    df['Mjob'] = df['Mjob'].map(job_map)
    df['Fjob'] = df['Fjob'].map(job_map)
    
    # Target: Binarize G3 (final grade) - pass (>=10) vs fail (<10)
    df['pass'] = (df['G3'] >= 10).astype(int)
    
    # Drop G1, G2 (intermediate grades - too correlated with target)
    # Drop G3 (we use 'pass' instead)
    df = df.drop(columns=['G1', 'G2', 'G3'])
    
    label_name = 'pass'
    X = df.drop(columns=[label_name])
    y = df[label_name]
    
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    
    # Store binners for later use
    binners = {
        'age': age_binner,
        'absences': absence_binner
    }
    
    return (
        df,
        X_train.to_numpy(),
        y_train.to_numpy().astype('int'),
        X_test.to_numpy(),
        y_test.to_numpy().astype('int'),
        binners
    )

# Load original training data
df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, binners = load_student()

print(f"Original training data shape: {X_train_orig.shape}")
print(f"Original test data shape: {X_test_orig.shape}")

# Load synthetic counterexamples
print("Loading synthetic counterexamples...")
df_synthetic = pd.read_csv(f'Fairify/counterexamples/UCI/counterexamples-{ORIGINAL_MODEL_NAME}.csv')
df_synthetic.dropna(inplace=True)

print(f"Synthetic data shape before processing: {df_synthetic.shape}")

# Apply same preprocessing as original data
# Bin age and absences
df_synthetic['age'] = binners['age'].transform(df_synthetic[['age']]).astype(int).flatten()
df_synthetic['absences'] = binners['absences'].transform(df_synthetic[['absences']]).astype(int).flatten()

# Binary encoding (if not already done)
binary_yes_no_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for col in binary_yes_no_cols:
    if df_synthetic[col].dtype == 'object':
        df_synthetic[col] = (df_synthetic[col] == 'yes').astype(int)

binary_other_cols = {
    'sex': 'M',
    'school': 'GP',
    'address': 'U',
    'famsize': 'GT3',
    'Pstatus': 'T'
}
for col, val in binary_other_cols.items():
    if df_synthetic[col].dtype == 'object':
        df_synthetic[col] = (df_synthetic[col] == val).astype(int)

# Categorical mappings
reason_map = {'home': 0, 'reputation': 1, 'course': 2, 'other': 3}
guardian_map = {'mother': 0, 'father': 1, 'other': 2}
job_map = {'at_home': 0, 'services': 1, 'other': 2, 'teacher': 3, 'health': 4}

if df_synthetic['reason'].dtype == 'object':
    df_synthetic['reason'] = df_synthetic['reason'].map(reason_map)
if df_synthetic['guardian'].dtype == 'object':
    df_synthetic['guardian'] = df_synthetic['guardian'].map(guardian_map)
if df_synthetic['Mjob'].dtype == 'object':
    df_synthetic['Mjob'] = df_synthetic['Mjob'].map(job_map)
if df_synthetic['Fjob'].dtype == 'object':
    df_synthetic['Fjob'] = df_synthetic['Fjob'].map(job_map)

# Handle label column (assuming 'decision' or 'pass')
if 'decision' in df_synthetic.columns:
    df_synthetic.rename(columns={'decision': 'pass'}, inplace=True)

label_name = 'pass'

# Extract features and labels
X_synthetic = df_synthetic.drop(columns=[label_name]).values
y_synthetic = df_synthetic[label_name].values

print(f"Synthetic data shape after processing: {X_synthetic.shape}")

# Create counterexample pairs with max label
X_train_ce = []
y_train_ce = []
for i in range(0, len(X_synthetic)-1, 2):
    x = X_synthetic[i]
    x_prime = X_synthetic[i+1]
   
    # Both get the maximum label (more lenient prediction)
    label = max(y_synthetic[i], y_synthetic[i+1])
   
    X_train_ce.append(x)
    X_train_ce.append(x_prime)
    y_train_ce.append(label)
    y_train_ce.append(label)

X_train_ce = np.array(X_train_ce)
y_train_ce = np.array(y_train_ce)

print(f"Counterexample pairs created: {len(X_train_ce)}")

# Mix original and counterexample data
X_train_mixed = np.vstack([X_train_orig, X_train_ce])
y_train_mixed = np.hstack([y_train_orig, y_train_ce])

print(f"Mixed training data shape: {X_train_mixed.shape}")

# Compile and train
original_model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])

epochs = 5
iterations = 1

print(f"\nTraining model iteratively for {iterations} iterations...")
for iteration in range(iterations):
    print(f"\nIteration {iteration+1}/{iterations}")
    original_model.fit(X_train_mixed, y_train_mixed,
                      epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)

# Save the retrained model
original_model.save(f'Fairify/models/uci/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred = (original_model.predict(X_test_orig) > 0.5).astype(int).flatten()
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_test_orig, y_pred)
f1 = f1_score(y_test_orig, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")