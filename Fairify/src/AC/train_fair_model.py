################################################# ORIGINAL
# import sys
# import os

# import random
# import numpy as np
# import tensorflow as tf

# def set_all_seeds(seed=42):
#     """Set all random seeds for reproducible results"""
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'  

# set_all_seeds(42)

# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
# sys.path.append(src_dir)
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Model as KerasModel
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
# from sklearn.metrics import accuracy_score, f1_score
# from utils.verif_utils import *
# from collections import defaultdict
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics.pairwise import euclidean_distances

# ORIGINAL_MODEL_NAME = "AC-2"        
# FAIRER_MODEL_NAME = "AC-2-Retrained"
# learning_rate = 0.008

# # AC-1 0.03/2
# # AC-2 0.008
# # AC-4 0.015
# # AC-5 
# # AC-6 0.03
# # AC-10 0.02895
# # AC-11 0.03
# # AC-17 0.023

# print("Loading original model...")
# original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
# print(original_model.summary())

# df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()
# feature_names = ['age', 'workclass', 'education', 'education-num',
#                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# if len(feature_names) != X_test_orig.shape[1]:
#     print(f"Warning: Feature names length ({len(feature_names)}) doesn't match data columns ({X_test_orig.shape[1]})")
#     feature_names = [f'feature_{i}' for i in range(X_test_orig.shape[1])]
#     feature_names[8] = 'sex'  

# print("Loading synthetic counterexamples...")
# df_synthetic = pd.read_csv(f'Fairify/counterexamples/AC/counterexamples-{ORIGINAL_MODEL_NAME}.csv')
# df_synthetic.dropna(inplace=True)

# cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
#             'relationship', 'native-country', 'sex']

# for feature in cat_feat:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

# if 'race' in encoders:
#     df_synthetic['race'] = encoders['race'].transform(df_synthetic['race'])
# binning_cols = ['capital-gain', 'capital-loss']

# for feature in binning_cols:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[[feature]])

# df_synthetic.rename(columns={'decision': 'income-per-year'}, inplace=True)
# label_name = 'income-per-year'

# X_synthetic = df_synthetic.drop(columns=[label_name])
# y_synthetic = df_synthetic[label_name]
# X_synthetic = df_synthetic.drop(columns=['income-per-year']).values
# y_synthetic = df_synthetic['income-per-year'].values

# X_train_ce = []
# y_train_ce = []
# for i in range(0, len(X_synthetic)-1, 2):
#     x = X_synthetic[i]
#     x_prime = X_synthetic[i+1]
   
#     label = max(y_synthetic[i], y_synthetic[i+1])
   
#     X_train_ce.append(x)
#     X_train_ce.append(x_prime)
#     y_train_ce.append(label)
#     y_train_ce.append(label)
# X_train_ce = np.array(X_train_ce)
# y_train_ce = np.array(y_train_ce)

# X_train_mixed = np.vstack([X_train_orig, X_train_ce])
# y_train_mixed = np.hstack([y_train_orig, y_train_ce])

# original_model.compile(optimizer=Adam(learning_rate=learning_rate),
#                       loss='binary_crossentropy', metrics=['accuracy'])
# epochs = 5
# iterations = 1
# print(f"Training model iteratively for {iterations} iterations...")
# for iteration in range(iterations):
#     print(f"\nIteration {iteration+1}/{iterations}")
#     original_model.fit(X_train_mixed, y_train_mixed,
#                       epochs=epochs, batch_size=32, validation_split=0.1)
# original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
# print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")

################################################# SMOTE SAMPLING
# import sys
# import os

# import random
# import numpy as np
# import tensorflow as tf

# def set_all_seeds(seed=42):
#     """Set all random seeds for reproducible results"""
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'  

# set_all_seeds(42)

# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
# sys.path.append(src_dir)
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Model as KerasModel
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
# from sklearn.metrics import accuracy_score, f1_score
# from utils.verif_utils import *
# from collections import defaultdict
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics.pairwise import euclidean_distances

# ORIGINAL_MODEL_NAME = "AC-2"        
# FAIRER_MODEL_NAME = "AC-2-Retrained"
# learning_rate = 0.000001

# print("Loading original model...")
# original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
# print(original_model.summary())

# df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()
# feature_names = ['age', 'workclass', 'education', 'education-num',
#                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# if len(feature_names) != X_test_orig.shape[1]:
#     print(f"Warning: Feature names length ({len(feature_names)}) doesn't match data columns ({X_test_orig.shape[1]})")
#     feature_names = [f'feature_{i}' for i in range(X_test_orig.shape[1])]
#     feature_names[8] = 'sex'  

# print("Loading synthetic counterexamples...")
# df_synthetic = pd.read_csv(f'Fairify/counterexamples/AC/counterexamples-{ORIGINAL_MODEL_NAME}.csv')
# df_synthetic.dropna(inplace=True)

# cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
#             'relationship', 'native-country', 'sex']

# for feature in cat_feat:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

# if 'race' in encoders:
#     df_synthetic['race'] = encoders['race'].transform(df_synthetic['race'])
# binning_cols = ['capital-gain', 'capital-loss']

# for feature in binning_cols:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[[feature]])

# df_synthetic.rename(columns={'decision': 'income-per-year'}, inplace=True)
# label_name = 'income-per-year'

# X_synthetic = df_synthetic.drop(columns=[label_name])
# y_synthetic = df_synthetic[label_name]
# X_synthetic = df_synthetic.drop(columns=['income-per-year']).values
# y_synthetic = df_synthetic['income-per-year'].values

# X_train_ce = []
# y_train_ce = []
# for i in range(0, len(X_synthetic)-1, 2):
#     x = X_synthetic[i]
#     x_prime = X_synthetic[i+1]
   
#     label = max(y_synthetic[i], y_synthetic[i+1])
   
#     X_train_ce.append(x)
#     X_train_ce.append(x_prime)
#     y_train_ce.append(label)
#     y_train_ce.append(label)
# X_train_ce = np.array(X_train_ce)
# y_train_ce = np.array(y_train_ce)

# X_train_mixed = np.vstack([X_train_orig, X_train_ce])
# y_train_mixed = np.hstack([y_train_orig, y_train_ce])

# # CFSMOTE-inspired Fair Sampling
# print("\n=== Applying Fair SMOTE Sampling ===")
# sex_idx = 8  # sex feature index

# # Split into 4 groups: (unprivileged/privileged) x (negative/positive)
# groups = {
#     'unprivileged_negative': [],
#     'unprivileged_positive': [],
#     'privileged_negative': [],
#     'privileged_positive': []
# }

# for i in range(len(X_train_mixed)):
#     sex = X_train_mixed[i, sex_idx]
#     label = y_train_mixed[i]
    
#     if sex == 0:  # unprivileged (assuming 0 is unprivileged)
#         if label == 0:
#             groups['unprivileged_negative'].append(i)
#         else:
#             groups['unprivileged_positive'].append(i)
#     else:  # privileged
#         if label == 0:
#             groups['privileged_negative'].append(i)
#         else:
#             groups['privileged_positive'].append(i)

# # Convert to arrays
# for key in groups:
#     groups[key] = np.array(groups[key])
#     print(f"{key}: {len(groups[key])} samples")

# # Find target size (max group size)
# target_size = max(len(groups[key]) for key in groups)
# print(f"\nTarget size per group: {target_size}")

# def smote_oversample(X, y, indices, target_count, k=5):
#     """SMOTE oversampling for a specific group"""
#     if len(indices) >= target_count:
#         return X[indices], y[indices]
    
#     X_group = X[indices]
#     y_group = y[indices]
    
#     n_synthetic = target_count - len(indices)
    
#     # Fit k-NN
#     k_actual = min(k, len(indices) - 1)
#     if k_actual < 1:
#         # Not enough samples, just replicate
#         return np.vstack([X_group] * (target_count // len(indices) + 1))[:target_count], \
#                np.hstack([y_group] * (target_count // len(indices) + 1))[:target_count]
    
#     knn = NearestNeighbors(n_neighbors=k_actual + 1)
#     knn.fit(X_group)
    
#     synthetic_X = []
#     synthetic_y = []
    
#     for _ in range(n_synthetic):
#         # Random sample from group
#         idx = np.random.randint(0, len(X_group))
#         sample = X_group[idx]
        
#         # Find neighbors
#         distances, neighbor_indices = knn.kneighbors([sample])
#         neighbor_idx = np.random.choice(neighbor_indices[0][1:])  # Exclude self
#         neighbor = X_group[neighbor_idx]
        
#         # Generate synthetic sample
#         gap = np.random.uniform(0, 1)
#         synthetic_sample = sample + gap * (neighbor - sample)
        
#         synthetic_X.append(synthetic_sample)
#         synthetic_y.append(y_group[idx])
    
#     return np.vstack([X_group, synthetic_X]), np.hstack([y_group, synthetic_y])

# # Oversample each group
# balanced_X = []
# balanced_y = []

# for group_name, group_indices in groups.items():
#     print(f"Oversampling {group_name}...")
#     X_group, y_group = smote_oversample(X_train_mixed, y_train_mixed, group_indices, target_size)
#     balanced_X.append(X_group)
#     balanced_y.append(y_group)

# X_train_balanced = np.vstack(balanced_X)
# y_train_balanced = np.hstack(balanced_y)

# print(f"\nOriginal mixed data: {len(X_train_mixed)} samples")
# print(f"Balanced data: {len(X_train_balanced)} samples")

# # Situation Testing: Filter discriminatory samples
# print("\n=== Applying Situation Testing ===")
# # Train proxy model
# from sklearn.linear_model import LogisticRegression
# proxy_model = LogisticRegression(max_iter=1000, random_state=42)
# proxy_model.fit(X_train_balanced, y_train_balanced)

# # Test each sample
# fair_indices = []
# for i in range(len(X_train_balanced)):
#     sample = X_train_balanced[i:i+1]
    
#     # Create counterfactual (flip sex)
#     counterfactual = sample.copy()
#     counterfactual[0, sex_idx] = 1 - counterfactual[0, sex_idx]
    
#     # Predict both
#     pred_original = proxy_model.predict(sample)[0]
#     pred_counter = proxy_model.predict(counterfactual)[0]
    
#     # Keep if predictions are same (fair)
#     if pred_original == pred_counter:
#         fair_indices.append(i)

# X_train_fair = X_train_balanced[fair_indices]
# y_train_fair = y_train_balanced[fair_indices]

# print(f"After situation testing: {len(X_train_fair)} samples ({len(fair_indices)/len(X_train_balanced)*100:.1f}% kept)")

# original_model.compile(optimizer=Adam(learning_rate=learning_rate),
#                       loss='binary_crossentropy', metrics=['accuracy'])
# epochs = 5
# iterations = 1
# print(f"\nTraining model iteratively for {iterations} iterations...")
# for iteration in range(iterations):
#     print(f"\nIteration {iteration+1}/{iterations}")
#     original_model.fit(X_train_fair, y_train_fair,
#                       epochs=epochs, batch_size=32, validation_split=0.1)
# original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
# print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")

################################################# KAMISHIMA ORIGINAL DATA RELABEL + GRID SEARCH

#!/usr/bin/env python3
import sys
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import defaultdict
from copy import deepcopy

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
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score
from utils.verif_utils import *

# Import metric evaluation functions
from metric_aif360 import measure_fairness_aif360
from metric_themis_causality_v2 import AdvancedCausalDiscriminationDetector
from metric_random_unfairness import FairnessEvaluator


###### RELABEL BY LIPSCHITZ WITH EQUALIZED ODDS (KAMISHIMA EXTENSION)
def relabel_by_lipschitz_individual_fairness(df, feature_cols=None, sensitive_col=None, true_label_col=None):
    """
    Implements Kamishima et al.'s extension of Dwork's individual fairness to account for 
    Equalized Odds (EOD) metric.
    """
    if df.empty or 'output' not in df.columns:
        return df
    
    df = df.copy()
    
    has_sensitive = sensitive_col is not None and sensitive_col in df.columns
    has_true_label = true_label_col is not None and true_label_col in df.columns
    
    if not has_true_label:
        raise ValueError("True label column required for Kamishima Individual EOD")
    
    if feature_cols is None:
        exclude_cols = ['output', 'decision']
        if has_sensitive:
            exclude_cols.append(sensitive_col)
        if has_true_label:
            exclude_cols.append(true_label_col)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    df['pair_id'] = df.index // 2
    
    def apply_lipschitz_eod(group):
        if len(group) < 2:
            return group
            
        idx1, idx2 = group.index[:2]
        
        x1 = group.loc[idx1, feature_cols].values
        x2 = group.loc[idx2, feature_cols].values
        
        y1 = group.loc[idx1, true_label_col]
        y2 = group.loc[idx2, true_label_col]
        
        distance = 0
        for i in range(len(x1)):
            if isinstance(x1[i], str):
                distance += 0 if x1[i] == x2[i] else 1
            else:
                distance += (float(x1[i]) - float(x2[i])) ** 2
        d_xy = np.sqrt(distance)
        
        output1 = group.loc[idx1, 'output']
        output2 = group.loc[idx2, 'output']
        
        p1 = np.array([1-output1, output1])
        p2 = np.array([1-output2, output2])
        d_tv = 0.5 * np.sum(np.abs(p1 - p2))
        
        if has_sensitive:
            s1 = group.loc[idx1, sensitive_col]
            s2 = group.loc[idx2, sensitive_col]
            
            if y1 == y2 and s1 != s2:
                allowed_distance = d_xy * 0.3
            elif y1 == y2:
                allowed_distance = d_xy * 0.7
            else:
                allowed_distance = d_xy * 0.95
        else:
            if y1 == y2:
                allowed_distance = d_xy * 0.7
            else:
                allowed_distance = d_xy * 0.95
        
        if d_tv > allowed_distance:
            target_tv = allowed_distance
            current_diff = abs(output1 - output2)
            new_diff = target_tv * 2
            
            if current_diff > 0:
                mid = (output1 + output2) / 2
                half_diff = new_diff / 2
                
                if output1 > output2:
                    new_output1 = mid + half_diff
                    new_output2 = mid - half_diff
                else:
                    new_output1 = mid - half_diff
                    new_output2 = mid + half_diff
                
                new_output1 = np.clip(new_output1, 0, 1)
                new_output2 = np.clip(new_output2, 0, 1)
                
                group.loc[idx1, 'output'] = new_output1
                group.loc[idx2, 'output'] = new_output2
                group.loc[idx1, 'decision'] = 1 if new_output1 >= 0.5 else 0
                group.loc[idx2, 'decision'] = 1 if new_output2 >= 0.5 else 0
        
        return group
    
    df = df.groupby('pair_id').apply(apply_lipschitz_eod).reset_index(drop=True)
    df = df.drop(columns=['pair_id'])
    
    return df


def evaluate_all_metrics(model, X_test, y_test, feature_names, constraint):
    """
    Evaluate all three fairness metrics: AIF360 (EOD, CNT), Themis Causality, and Unfairness
    Returns: dict with all metric scores
    """
    print("\n" + "="*40)
    print("EVALUATING ALL FAIRNESS METRICS")
    print("="*40)
    
    metrics = {}
    
    # 1. AIF360 Metrics (EOD and CNT)
    print("\n📊 1. AIF360 Metrics (EOD, CNT)...")
    try:
        aif_metrics = measure_fairness_aif360(model, X_test, y_test, feature_names, 
                                              protected_attribute='sex')
        metrics['accuracy'] = aif_metrics['accuracy']
        metrics['f1_score'] = aif_metrics['f1_score']
        metrics['eod'] = aif_metrics['equal_opportunity_diff']  # EOD
        metrics['cnt'] = aif_metrics['consistency']  # CNT
        print(f"   EOD: {metrics['eod']:.4f} (target: 0.0)")
        print(f"   CNT: {metrics['cnt']:.4f} (target: 1.0)")
    except Exception as e:
        print(f"   ❌ Error in AIF360 evaluation: {e}")
        metrics['eod'] = float('inf')
        metrics['cnt'] = 0.0
    
    # 2. Themis Causality
    print("\n📊 2. Themis Causality...")
    try:
        def build_predict_fn(model):
            def predict_fn(feature_dict):
                x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
                return int(model.predict(x, verbose=0)[0][0] > 0.5)
            return predict_fn
        
        detector = AdvancedCausalDiscriminationDetector(
            build_predict_fn(model),
            max_samples=5000,  # Reduced for faster evaluation
            min_samples=500,
            sampling_method='sobol',
            random_seed=42
        )
        
        # Load data for feature values
        df, _, _, _, _, _ = load_adult_ac1()
        for fname in feature_names:
            unique_vals = sorted(set(df[fname]))
            detector.add_feature(fname, unique_vals)
        
        _, causality_rate, _, stats = detector.causal_discrimination(['sex'])
        metrics['causality'] = causality_rate
        print(f"   Causality: {metrics['causality']:.4f} (target: 0.0)")
    except Exception as e:
        print(f"   ❌ Error in Themis evaluation: {e}")
        metrics['causality'] = float('inf')
    
    # 3. Unfairness (Individual Discrimination)
    print("\n📊 3. Unfairness (Individual Discrimination)...")
    try:
        evaluator = FairnessEvaluator(model, constraint)
        unfairness_rate, interval = evaluator.evaluate_individual_fairness(sample_round=10, num_gen=100)
        metrics['unfairness'] = unfairness_rate
        print(f"   Unfairness: {metrics['unfairness']:.4f} ± {interval:.4f} (target: 0.0)")
    except Exception as e:
        print(f"   ❌ Error in Unfairness evaluation: {e}")
        metrics['unfairness'] = float('inf')
    
    print("\n" + "="*80)
    return metrics


def check_if_improved(baseline_metrics, current_metrics):
    """
    Check if current metrics show improvement over baseline.
    Returns: (improved, reason)
    """
    # Check if ANY fairness metric improved
    eod_improved = abs(current_metrics['eod']) < abs(baseline_metrics['eod'])
    cnt_improved = abs(1 - current_metrics['cnt']) < abs(1 - baseline_metrics['cnt'])
    causality_improved = current_metrics['causality'] < baseline_metrics['causality']
    unfairness_improved = current_metrics['unfairness'] < baseline_metrics['unfairness']
    
    # Check if accuracy didn't drop too much (within 2%)
    acc_drop = baseline_metrics['accuracy'] - current_metrics['accuracy']
    acc_acceptable = acc_drop <= 0.02
    
    if not acc_acceptable:
        return False, f"Accuracy dropped too much: {acc_drop*100:.2f}%"
    
    if eod_improved or cnt_improved or causality_improved or unfairness_improved:
        improvements = []
        if eod_improved:
            improvements.append("EOD")
        if cnt_improved:
            improvements.append("CNT")
        if causality_improved:
            improvements.append("Causality")
        if unfairness_improved:
            improvements.append("Unfairness")
        return True, f"Improved: {', '.join(improvements)}"
    
    return False, "No fairness metrics improved"


def grid_search_retrain(original_model, X_train_mixed, y_train_mixed, X_test, y_test, 
                       feature_names, constraint, learning_rates, epochs=5, max_iterations=3):
    """
    Grid search over learning rates. Evaluates only AFTER all training is complete for each LR.
    
    Args:
        original_model: Base model to retrain
        X_train_mixed: Combined training data (original + counterexamples)
        y_train_mixed: Combined training labels
        X_test, y_test: Test data
        feature_names: List of feature names
        constraint: Constraint array for unfairness metric
        learning_rates: List of learning rates to try
        epochs: Epochs per iteration
        max_iterations: Maximum training iterations per LR
    
    Returns:
        best_model: Model with best fairness improvements
        best_lr: Best learning rate found
        results_log: Detailed results for each LR
    """
    print("\n" + "="*80)
    print("GRID SEARCH")
    print("="*80)
    print(f"Learning Rates: {learning_rates}")
    print(f"Epochs per iteration: {epochs}")
    print(f"Iterations per LR: {max_iterations}")
    print("="*80)
    
    # Evaluate baseline model
    print("\n🔍 Evaluating BASELINE model...")
    baseline_metrics = evaluate_all_metrics(original_model, X_test, y_test, 
                                           feature_names, constraint)
    
    print("\n📋 BASELINE METRICS:")
    print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"   EOD: {baseline_metrics['eod']:.4f}")
    print(f"   CNT: {baseline_metrics['cnt']:.4f}")
    print(f"   Causality: {baseline_metrics['causality']:.4f}")
    print(f"   Unfairness: {baseline_metrics['unfairness']:.4f}")
    
    best_model = None
    best_lr = None
    best_metrics = baseline_metrics.copy()
    best_fairness_score = calculate_fairness_score(baseline_metrics)
    results_log = []
    
    # Grid search over learning rates
    for lr_idx, lr in enumerate(learning_rates):
        print("\n" + "="*80)
        print(f"LR {lr_idx+1}/{len(learning_rates)}: {lr}")
        print("="*80)
        
        # Clone original model for this LR
        model_copy = clone_model(original_model)
        model_copy.set_weights(original_model.get_weights())
        model_copy.compile(optimizer=Adam(learning_rate=lr),
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])
        
        # TRAIN ALL ITERATIONS
        print(f"Training {max_iterations} iterations...")
        for iteration in range(max_iterations):
            model_copy.fit(X_train_mixed, y_train_mixed,
                          epochs=epochs, 
                          batch_size=32, 
                          validation_split=0.1,
                          verbose=0)
        print("✅ Training complete")
        
        # EVALUATE ONCE AFTER ALL TRAINING
        print("\n🔍 Evaluating...")
        current_metrics = evaluate_all_metrics(model_copy, X_test, y_test,
                                               feature_names, constraint)
        
        lr_results = {
            'learning_rate': lr,
            'metrics': current_metrics.copy()
        }
        results_log.append(lr_results)
        
        print(f"\n📊 Results for LR={lr}:")
        print(f"   Accuracy: {current_metrics['accuracy']:.4f} (Δ: {current_metrics['accuracy']-baseline_metrics['accuracy']:+.4f})")
        print(f"   EOD: {current_metrics['eod']:.4f} (Δ: {current_metrics['eod']-baseline_metrics['eod']:+.4f})")
        print(f"   CNT: {current_metrics['cnt']:.4f} (Δ: {current_metrics['cnt']-baseline_metrics['cnt']:+.4f})")
        print(f"   Causality: {current_metrics['causality']:.4f} (Δ: {current_metrics['causality']-baseline_metrics['causality']:+.4f})")
        print(f"   Unfairness: {current_metrics['unfairness']:.4f} (Δ: {current_metrics['unfairness']-baseline_metrics['unfairness']:+.4f})")
        
        # Update best model if this is better
        current_fairness_score = calculate_fairness_score(current_metrics)
        if current_fairness_score < best_fairness_score:
            print("\n✨ NEW BEST MODEL!")
            best_model = clone_model(model_copy)
            best_model.set_weights(model_copy.get_weights())
            best_lr = lr
            best_metrics = current_metrics.copy()
            best_fairness_score = current_fairness_score
    
    # Final summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    
    if best_model is not None and best_lr is not None:
        print(f"\n✅ BEST LR: {best_lr}")
        print(f"\n📊 BEST METRICS:")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f} (Δ: {best_metrics['accuracy']-baseline_metrics['accuracy']:+.4f})")
        print(f"   EOD: {best_metrics['eod']:.4f} (Δ: {best_metrics['eod']-baseline_metrics['eod']:+.4f})")
        print(f"   CNT: {best_metrics['cnt']:.4f} (Δ: {best_metrics['cnt']-baseline_metrics['cnt']:+.4f})")
        print(f"   Causality: {best_metrics['causality']:.4f} (Δ: {best_metrics['causality']-baseline_metrics['causality']:+.4f})")
        print(f"   Unfairness: {best_metrics['unfairness']:.4f} (Δ: {best_metrics['unfairness']-baseline_metrics['unfairness']:+.4f})")
        return best_model, best_lr, results_log, best_metrics
    else:
        print("\n⚠️  No improvement found. Returning original model.")
        return original_model, None, results_log, baseline_metrics


def calculate_fairness_score(metrics):
    """
    Calculate composite fairness score (lower is better).
    Combines all fairness metrics into a single score.
    """
    # EOD: distance from 0
    eod_score = abs(metrics['eod'])
    
    # CNT: distance from 1
    cnt_score = abs(1 - metrics['cnt'])
    
    # Causality: distance from 0
    causality_score = metrics['causality']
    
    # Unfairness: distance from 0
    unfairness_score = metrics['unfairness']
    
    # Weighted sum (you can adjust weights)
    composite_score = (eod_score * 1.0 + 
                      cnt_score * 1.0 + 
                      causality_score * 1.0 + 
                      unfairness_score * 1.0)
    
    return composite_score


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    ORIGINAL_MODEL_NAME = "AC-12"        
    FAIRER_MODEL_NAME = "AC-12-Retrained"
    
    # GRID SEARCH LEARNING RATES - CUSTOMIZE THESE!
    LEARNING_RATES = [0.01, 0.001, 0.005]
    # LEARNING_RATES = [0.0001, 0.00005, 0.00001, 0.000005]
    EPOCHS_PER_ITERATION = 5
    MAX_ITERATIONS = 3
    
    print("="*80)
    print("FAIRNESS-AWARE MODEL RETRAINING WITH GRID SEARCH")
    print("="*80)
    
    # Load original model
    print("\n📥 Loading original model...")
    original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
    print(original_model.summary())
    
    # Load data
    print("\n📥 Loading data...")
    df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()
    feature_names = ['age', 'workclass', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    
    # Define constraint for unfairness metric
    constraint = np.array([
        [10, 100],    # age
        [0, 6],       # workclass
        [0, 15],      # education
        [1, 16],      # education-num
        [0, 6],       # marital-status
        [0, 13],      # occupation
        [0, 5],       # relationship
        [0, 4],       # race
        [0, 1],       # sex
        [0, 19],      # capital-gain
        [0, 19],      # capital-loss
        [1, 100],     # hours-per-week
        [0, 40]       # native-country
    ])
    
    # STEP 1: Apply Kamishima relabeling
    print("\n" + "="*80)
    print("STEP 1: Applying Kamishima Individual EOD relabeling")
    print("="*80)
    df_train_orig = pd.DataFrame(X_train_orig, columns=feature_names)
    df_train_orig['true_label'] = y_train_orig
    df_train_orig['output'] = original_model.predict(X_train_orig, verbose=0).flatten()
    df_train_orig['decision'] = (df_train_orig['output'] >= 0.5).astype(int)
    
    df_train_relabeled = relabel_by_lipschitz_individual_fairness(
        df_train_orig, 
        feature_cols=feature_names,
        sensitive_col='sex',
        true_label_col='true_label'
    )
    
    y_train_orig_relabeled = df_train_relabeled['decision'].values
    print(f"✅ Relabeled {len(y_train_orig_relabeled)} training samples")
    
    # STEP 2: Load counterexamples
    print("\n" + "="*80)
    print("STEP 2: Loading synthetic counterexamples")
    print("="*80)
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
    
    # Prepare CE data in pairs
    X_train_ce = []
    y_train_ce = []
    for i in range(0, len(X_synthetic)-1, 2):
        X_train_ce.append(X_synthetic[i])
        X_train_ce.append(X_synthetic[i+1])
        y_train_ce.append(y_synthetic[i])
        y_train_ce.append(y_synthetic[i+1])
    
    X_train_ce = np.array(X_train_ce)
    y_train_ce = np.array(y_train_ce)
    print(f"✅ Loaded {len(X_train_ce)} counterexample samples")
    
    # Mix relabeled original data with counterexamples
    X_train_mixed = np.vstack([X_train_orig, X_train_ce])
    y_train_mixed = np.hstack([y_train_orig_relabeled, y_train_ce])
    print(f"✅ Total training samples: {len(X_train_mixed)}")
    
    # STEP 3: Grid search with fairness-aware early stopping
    print("\n" + "="*80)
    print("STEP 3: Grid Search Retraining")
    print("="*80)
    
    best_model, best_lr, results_log, best_metrics = grid_search_retrain(
        original_model=original_model,
        X_train_mixed=X_train_mixed,
        y_train_mixed=y_train_mixed,
        X_test=X_test_orig,
        y_test=y_test_orig,
        feature_names=feature_names,
        constraint=constraint,
        learning_rates=LEARNING_RATES,
        epochs=EPOCHS_PER_ITERATION,
        max_iterations=MAX_ITERATIONS
    )
    
    # STEP 4: Save best model
    if best_model is not None and best_lr is not None:
        print("\n" + "="*80)
        print("STEP 4: Saving best model")
        print("="*80)
        best_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
        print(f"✅ Best model saved as {FAIRER_MODEL_NAME}.h5")
        print(f"✅ Best learning rate: {best_lr}")
    else:
        print("\n⚠️  No improvement found. Original model not modified.")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)