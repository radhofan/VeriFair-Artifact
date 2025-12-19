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

from utils.verif_utils import load_adult_ac1
from metric_aif360 import measure_fairness_aif360
from metric_themis_causality_v2 import AdvancedCausalDiscriminationDetector
from metric_random_unfairness import FairnessEvaluator

ORIGINAL_MODEL_NAME = "AC-2"
FAIRER_MODEL_NAME = "AC-2-Retrained"

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
    test_model = load_model(f'Fairify/models/adult/{model_name}.h5')
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
original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()

feature_names = ['age', 'workclass', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

print("\nLoading synthetic counterexamples...")
df_synthetic = pd.read_csv(f'Fairify/counterexamples/AC/counterexamples-{ORIGINAL_MODEL_NAME}.csv')
df_synthetic.dropna(inplace=True)

cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'native-country', 'sex']
for feature in cat_feat:
    if feature in encoders:
        df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

if 'race' in encoders:
    df_synthetic['race'] = encoders['race'].transform(df_synthetic['race'])

binning_cols = ['capital-gain', 'capital-loss']
for feature in binning_cols:
    if feature in encoders:
        df_synthetic[feature] = encoders[feature].transform(df_synthetic[[feature]])

df_synthetic.rename(columns={'decision': 'income-per-year'}, inplace=True)
X_synthetic = df_synthetic.drop(columns=['income-per-year']).values
y_synthetic = df_synthetic['income-per-year'].values

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
    [10, 100], [0, 6], [0, 15], [1, 16], [0, 6], [0, 13], [0, 5],
    [0, 4], [0, 1], [0, 19], [0, 19], [1, 100], [0, 40]
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
final_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
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

final_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Optimal model saved as {FAIRER_MODEL_NAME}.h5")


