#!/usr/bin/env python3
"""
Bayesian Optimization for Learning Rate Search
Uses Gaussian Process with UCB acquisition to efficiently find optimal LR
Balances exploration vs exploitation to minimize evaluations needed
"""

import sys
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

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

from metric_aif360 import measure_fairness_aif360
from metric_themis_causality_v2 import AdvancedCausalDiscriminationDetector
from metric_random_unfairness import FairnessEvaluator

ORIGINAL_MODEL_NAME = "UCI-1"
FAIRER_MODEL_NAME = "UCI-1-Retrained"

# Configuration
MAX_EVALUATIONS = 15
N_INITIAL_RANDOM = 3
EPOCHS = 5
ITERATIONS = 1
LR_RANGE = (0.0001, 0.05)
MAX_ACCURACY_DROP = 0.02

# Multi-objective weights
WEIGHTS = {
    'consistency': 0.33,
    'causality': 0.33,
    'individual': 0.34
}


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


class BayesianLROptimizer:
    """Bayesian optimization for learning rate search"""
    
    def __init__(self, lr_range=(0.0001, 0.05)):
        self.lr_range = lr_range
        self.evaluated_lrs = []
        self.results = []
        
    def suggest_next_lr(self, n_suggestions=1):
        """Suggest next LR using Bayesian optimization"""
        if len(self.evaluated_lrs) < N_INITIAL_RANDOM:
            # Initial random samples using log-uniform distribution
            return np.exp(np.random.uniform(
                np.log(self.lr_range[0]), 
                np.log(self.lr_range[1]), 
                n_suggestions
            ))
        
        # Prepare data for Gaussian Process
        X = np.array(self.evaluated_lrs).reshape(-1, 1)
        X_log = np.log(X)
        
        # Get scores (higher is better)
        y = np.array([r.get('total_score', -np.inf) for r in self.results])
        
        # Only use finite scores
        valid_idx = np.isfinite(y)
        if len(y[valid_idx]) < 2:
            return np.exp(np.random.uniform(
                np.log(self.lr_range[0]), 
                np.log(self.lr_range[1]), 
                n_suggestions
            ))
        
        # Fit Gaussian Process
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42
        )
        gp.fit(X_log[valid_idx], y[valid_idx])
        
        # Sample candidate points
        n_candidates = 1000
        X_candidates_log = np.random.uniform(
            np.log(self.lr_range[0]),
            np.log(self.lr_range[1]),
            n_candidates
        ).reshape(-1, 1)
        
        # Predict mean and std
        mu, sigma = gp.predict(X_candidates_log, return_std=True)
        
        # Upper Confidence Bound acquisition function
        kappa = 2.0
        ucb = mu + kappa * sigma
        
        # Get top suggestions that haven't been evaluated
        suggestions = []
        sorted_idx = np.argsort(ucb)[::-1]
        
        for idx in sorted_idx:
            lr_candidate = np.exp(X_candidates_log[idx][0])
            # Check if sufficiently different from evaluated LRs (at least 50% different)
            if all(abs(lr_candidate - lr_eval) / lr_eval > 0.5 
                   for lr_eval in self.evaluated_lrs):
                suggestions.append(lr_candidate)
                if len(suggestions) >= n_suggestions:
                    break
        
        # If no diverse candidates found, pick best from candidates
        if len(suggestions) == 0:
            for idx in sorted_idx:
                lr_candidate = np.exp(X_candidates_log[idx][0])
                if lr_candidate not in self.evaluated_lrs:
                    suggestions.append(lr_candidate)
                    break
        
        return np.array(suggestions)



def evaluate_model_with_lr(lr, model_name, X_train_mixed, y_train_mixed, 
                           X_test_orig, y_test_orig, feature_names, 
                           df_original, constraint, baseline_metrics):
    """Evaluate a model trained with given learning rate"""
    
    print(f"\n{'='*80}")
    print(f"Testing LR = {lr:.6f}")
    print(f"{'='*80}")
    
    set_all_seeds(42)
    
    # Load fresh model
    test_model = load_model(f'Fairify/models/uci/{model_name}.h5')
    test_model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print(f"\nTraining with LR={lr:.6f}...")
    for iteration in range(ITERATIONS):
        test_model.fit(
            X_train_mixed, y_train_mixed,
            epochs=EPOCHS,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
    
    # Evaluate metrics
    print("\nEvaluating fairness metrics...")
    metrics = measure_fairness_aif360(
        test_model, X_test_orig, y_test_orig, feature_names, protected_attribute='sex'
    )
    
    accuracy = metrics['accuracy']
    consistency = metrics['consistency']
    acc_drop = baseline_metrics['accuracy'] - accuracy
    
    print(f"Accuracy: {accuracy:.4f} (drop: {acc_drop:.4f})")
    print(f"Consistency: {consistency:.4f}")
    
    # Check accuracy constraint
    if acc_drop > MAX_ACCURACY_DROP:
        print(f"❌ Accuracy drop too large ({acc_drop:.4f} > {MAX_ACCURACY_DROP})")
        return {
            'lr': lr,
            'accuracy': accuracy,
            'acc_drop': acc_drop,
            'consistency': consistency,
            'causal_disc': None,
            'individual_disc': None,
            'acceptable': False,
            'total_score': -np.inf,
            'reason': 'accuracy_drop_too_large'
        }
    
    # Evaluate Causality
    print("Evaluating Causality (Themis++)...")
    def build_predict_fn(model):
        def predict_fn(feature_dict):
            x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
            return int(model.predict(x, verbose=0)[0][0] > 0.5)
        return predict_fn
    
    detector = AdvancedCausalDiscriminationDetector(
        build_predict_fn(test_model),
        max_samples=5000,
        min_samples=500,
        sampling_method='sobol',
        random_seed=42
    )
    for fname in feature_names:
        unique_vals = sorted(set(df_original[fname]))
        detector.add_feature(fname, unique_vals)
    
    _, causal_rate, _, _ = detector.causal_discrimination(['sex'])
    print(f"Causal Discrimination: {causal_rate:.4f}")
    
    # Evaluate Individual Discrimination
    print("Evaluating Individual Discrimination...")
    ind_evaluator = FairnessEvaluator(test_model, constraint)
    ind_disc, _ = ind_evaluator.evaluate_individual_fairness(
        sample_round=10, num_gen=100
    )
    
    # Calculate improvements
    consistency_improvement = consistency - baseline_metrics['consistency']
    causal_improvement = baseline_metrics['causal_disc'] - causal_rate
    individual_improvement = baseline_metrics['individual_disc'] - ind_disc
    
    # Calculate weighted score
    total_score = (
        WEIGHTS['consistency'] * consistency_improvement * 10 +
        WEIGHTS['causality'] * causal_improvement * 10 +
        WEIGHTS['individual'] * individual_improvement * 10
    )
    
    print(f"\n{'='*60}")
    print(f"SUMMARY FOR LR={lr:.6f}")
    print(f"{'='*60}")
    print(f"Accuracy Drop: {acc_drop:.4f}")
    print(f"Consistency Improvement: {consistency_improvement:+.4f}")
    print(f"Causal Improvement: {causal_improvement:+.4f}")
    print(f"Individual Improvement: {individual_improvement:+.4f}")
    print(f"Total Score: {total_score:.4f}")
    
    return {
        'lr': lr,
        'accuracy': accuracy,
        'acc_drop': acc_drop,
        'consistency': consistency,
        'consistency_improvement': consistency_improvement,
        'causal_disc': causal_rate,
        'causal_improvement': causal_improvement,
        'individual_disc': ind_disc,
        'individual_improvement': individual_improvement,
        'total_score': total_score,
        'acceptable': True,
        'reason': 'fully_evaluated'
    }


# Main execution
print("="*80)
print("BAYESIAN OPTIMIZATION FOR LEARNING RATE SEARCH")
print("="*80)

print("\nLoading original model and data...")
original_model = load_model(f'Fairify/models/uci/{ORIGINAL_MODEL_NAME}.h5')
df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, binners = load_student()

feature_names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 
                'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
                'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

print("\nLoading synthetic counterexamples...")
df_synthetic = pd.read_csv(f'Fairify/counterexamples/UCI/counterexamples-{ORIGINAL_MODEL_NAME}.csv')

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

df_synthetic.dropna(inplace=True)

# Extract features and labels
X_synthetic = df_synthetic.drop(columns=[label_name]).values
y_synthetic = df_synthetic[label_name].values

# Prepare counterexample pairs
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

X_train_ce = np.array(X_train_ce)
y_train_ce = np.array(y_train_ce)

X_train_mixed = np.vstack([X_train_orig, X_train_ce])
y_train_mixed = np.hstack([y_train_orig, y_train_ce])

# Evaluate baseline model
print("\n" + "="*80)
print("BASELINE MODEL METRICS")
print("="*80)
baseline_metrics = measure_fairness_aif360(
    original_model, X_test_orig, y_test_orig, feature_names, protected_attribute='sex'
)
baseline_acc = baseline_metrics['accuracy']
print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
print(f"Baseline Consistency: {baseline_metrics['consistency']:.4f}")

# Baseline Causality
print("\nEvaluating baseline Causality (Themis++)...")
def build_predict_fn(model):
    def predict_fn(feature_dict):
        x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
        return int(model.predict(x, verbose=0)[0][0] > 0.5)
    return predict_fn

baseline_detector = AdvancedCausalDiscriminationDetector(
    build_predict_fn(original_model),
    max_samples=5000,
    min_samples=500,
    sampling_method='sobol',
    random_seed=42
)
for fname in feature_names:
    unique_vals = sorted(set(df_original[fname]))
    baseline_detector.add_feature(fname, unique_vals)

_, baseline_causal_rate, _, _ = baseline_detector.causal_discrimination(['sex'])
print(f"Baseline Causal Discrimination: {baseline_causal_rate:.4f}")

# Baseline Individual Discrimination
print("\nEvaluating baseline Individual Discrimination...")
constraint = np.array([
    [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    [0, 4], [0, 4], [0, 4], [0, 4], [0, 3], [0, 2],
    [1, 4], [1, 4], [0, 3], [0, 1], [0, 1],
    [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    [1, 5], [1, 5], [1, 5], [1, 5], [1, 5], [1, 5], [0, 4]
])
baseline_ind_evaluator = FairnessEvaluator(original_model, constraint)
baseline_ind_disc, _ = baseline_ind_evaluator.evaluate_individual_fairness(
    sample_round=10, num_gen=100
)

baseline_metrics['causal_disc'] = baseline_causal_rate
baseline_metrics['individual_disc'] = baseline_ind_disc

# Initialize Bayesian optimizer
print("\n" + "="*80)
print("STARTING BAYESIAN OPTIMIZATION")
print(f"Max Evaluations: {MAX_EVALUATIONS}")
print(f"LR Range: [{LR_RANGE[0]:.6f}, {LR_RANGE[1]:.6f}]")
print("="*80)

optimizer = BayesianLROptimizer(lr_range=LR_RANGE)

# Run optimization loop
for eval_num in range(MAX_EVALUATIONS):
    print(f"\n{'#'*80}")
    print(f"EVALUATION {eval_num + 1}/{MAX_EVALUATIONS}")
    print(f"{'#'*80}")
    
    # Get next LR suggestion
    next_lr = optimizer.suggest_next_lr(n_suggestions=1)[0]
    print(f"Suggested LR: {next_lr:.6f}")
    
    # Evaluate this LR
    result = evaluate_model_with_lr(
        next_lr, ORIGINAL_MODEL_NAME, X_train_mixed, y_train_mixed,
        X_test_orig, y_test_orig, feature_names, df_original,
        constraint, baseline_metrics
    )
    
    # Store results
    optimizer.evaluated_lrs.append(next_lr)
    optimizer.results.append(result)
    
    # Show current best
    valid_results = [r for r in optimizer.results if r['acceptable'] and r['total_score'] > -np.inf]
    if valid_results:
        current_best = max(valid_results, key=lambda x: x['total_score'])
        print(f"\n🏆 Current Best LR: {current_best['lr']:.6f} (Score: {current_best['total_score']:.4f})")

# Final results
print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)

results_df = pd.DataFrame(optimizer.results)
print("\n", results_df.to_string(index=False))

# Find optimal LR
acceptable_results = [r for r in optimizer.results if r['acceptable'] and r['total_score'] > -np.inf]

if not acceptable_results:
    print("\n❌ No learning rates satisfied the accuracy constraint!")
    sys.exit(1)

optimal = max(acceptable_results, key=lambda x: x['total_score'])

print("\n" + "="*80)
print("OPTIMAL LEARNING RATE FOUND")
print("="*80)
print(f"\nOptimal LR: {optimal['lr']:.6f}")
print(f"\nMetrics:")
print(f"  Accuracy: {optimal['accuracy']:.4f} (drop: {optimal['acc_drop']:.4f})")
print(f"  Consistency: {optimal['consistency']:.4f} (improvement: {optimal['consistency_improvement']:+.4f})")
print(f"  Causal Discrimination: {optimal['causal_disc']:.4f} (improvement: {optimal['causal_improvement']:+.4f})")
print(f"  Individual Discrimination: {optimal['individual_disc']:.4f} (improvement: {optimal['individual_improvement']:+.4f})")
print(f"  Total Score: {optimal['total_score']:.4f}")

# Train final model with optimal LR
print("\n" + "="*80)
print(f"TRAINING FINAL MODEL WITH OPTIMAL LR={optimal['lr']:.6f}")
print("="*80)

set_all_seeds(42)
final_model = load_model(f'Fairify/models/uci/{ORIGINAL_MODEL_NAME}.h5')
final_model.compile(
    optimizer=Adam(learning_rate=optimal['lr']),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

for iteration in range(ITERATIONS):
    print(f"\nIteration {iteration+1}/{ITERATIONS}")
    final_model.fit(
        X_train_mixed, y_train_mixed,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.1
    )

final_model.save(f'Fairify/models/uci/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Optimal model saved as {FAIRER_MODEL_NAME}.h5")