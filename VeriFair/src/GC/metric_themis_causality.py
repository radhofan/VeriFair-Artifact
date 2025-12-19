#!/usr/bin/env python3
"""
Themis++ Evaluation for German Credit Dataset
Uses advanced sampling (Sobol) and robust statistical testing
"""

from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
from scipy.stats import qmc
import tensorflow as tf
import numpy as np
import os
from typing import Dict, List, Tuple, Optional

def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  

class Input:
    def __init__(self, name, values, kind="categorical"):
        self.name = name
        self.values = [str(v) for v in values]
        self.kind = kind

    def get_random_value(self):
        """Return a random value from possible values."""
        return random.choice(self.values)

    def __str__(self):
        return f"Feature: {self.name}, Values: {self.values}"


class AdvancedCausalDiscriminationDetector:
    """
    Improved discrimination detector with advanced sampling and robust statistics.
    """
    def __init__(self, model_predict_fn, max_samples=5000, min_samples=500, 
                 random_seed=42, sampling_method='sobol'):
        """
        Args:
            model_predict_fn: Function that returns binary prediction
            max_samples: Maximum number of test samples
            min_samples: Minimum samples before statistical tests
            random_seed: Random seed for reproducibility
            sampling_method: 'uniform', 'sobol', 'stratified', or 'adaptive'
        """
        self.model_predict_fn = model_predict_fn
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.random_seed = random_seed
        self.sampling_method = sampling_method
        self.inputs = {}
        self.input_order = []
        self._cache = {}
        
        self.rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)
        
        # For Sobol sampling
        self._sobol_engine = None
        self._sobol_index = 0
        
        # For adaptive sampling
        self._discrimination_density = {}

    def add_feature(self, name, values, kind="categorical"):
        self.inputs[name] = Input(name, values, kind)
        self.input_order.append(name)

    def add_continuous_feature(self, name, min_val, max_val, num_values=10):
        values = [min_val + i * (max_val - min_val) / (num_values - 1) 
                 for i in range(num_values)]
        self.add_feature(name, values, "continuous")

    # ========================================================================
    # CONTRIBUTION #1: ADVANCED SAMPLING STRATEGIES
    # ========================================================================
    
    def _initialize_sobol_sampler(self, num_features):
        """Initialize Sobol quasi-random sequence generator."""
        if num_features > 0:
            self._sobol_engine = qmc.Sobol(d=num_features, scramble=True, seed=self.random_seed)
            self._sobol_index = 0
    
    def _generate_sobol_assignment(self, feature_names):
        """
        Generate assignment using Sobol sequences (quasi-Monte Carlo).
        Better space coverage than uniform random sampling.
        """
        if not feature_names:
            return {}
            
        if self._sobol_engine is None or self._sobol_engine.d != len(feature_names):
            self._initialize_sobol_sampler(len(feature_names))
        
        # Get next point from Sobol sequence
        sample = self._sobol_engine.random(1)[0]
        
        assignment = {}
        for i, fname in enumerate(feature_names):
            feature = self.inputs[fname]
            # Map [0,1] to discrete feature values
            idx = int(sample[i] * len(feature.values))
            idx = min(idx, len(feature.values) - 1)
            assignment[fname] = feature.values[idx]
        
        self._sobol_index += 1
        return assignment
    
    def _generate_stratified_assignment(self, feature_names, stratum_idx, num_strata):
        """
        Generate assignment using stratified sampling.
        Ensures all regions of input space are represented.
        """
        assignment = {}
        for i, fname in enumerate(feature_names):
            feature = self.inputs[fname]
            num_values = len(feature.values)
            
            stratum_size = max(1, num_values // num_strata)
            stratum_start = (stratum_idx % num_strata) * stratum_size
            stratum_end = min(stratum_start + stratum_size, num_values)
            
            if stratum_start >= num_values:
                stratum_start = 0
                stratum_end = min(stratum_size, num_values)
            
            idx = self.rng.randint(stratum_start, stratum_end - 1)
            assignment[fname] = feature.values[idx]
        
        return assignment
    
    def _generate_adaptive_assignment(self, feature_names, discrimination_rate):
        """
        Adaptive sampling: focus more on regions with high discrimination.
        """
        if not self._discrimination_density or self.rng.random() < 0.3:
            return self._generate_random_assignment(feature_names)
        
        density_keys = list(self._discrimination_density.keys())
        base_case_tuple = self.rng.choice(density_keys)
        base_case = dict(base_case_tuple)
        
        assignment = {}
        for fname in feature_names:
            if self.rng.random() < 0.7:
                assignment[fname] = base_case.get(fname, self.inputs[fname].get_random_value())
            else:
                assignment[fname] = self.inputs[fname].get_random_value()
        
        return assignment

    def _generate_random_assignment(self, feature_names):
        """Standard uniform random sampling (baseline)."""
        return {name: self.inputs[name].get_random_value() for name in feature_names}
    
    def _generate_assignment(self, feature_names, iteration, discrimination_rate=0.0):
        """
        Generate assignment based on selected sampling method.
        """
        if self.sampling_method == 'sobol':
            return self._generate_sobol_assignment(feature_names)
        elif self.sampling_method == 'stratified':
            num_strata = int(np.sqrt(self.max_samples))
            return self._generate_stratified_assignment(feature_names, iteration, num_strata)
        elif self.sampling_method == 'adaptive':
            return self._generate_adaptive_assignment(feature_names, discrimination_rate)
        else:
            return self._generate_random_assignment(feature_names)

    # ========================================================================
    # CONTRIBUTION #2: ROBUST STATISTICAL TESTING
    # ========================================================================
    
    def _bootstrap_confidence_interval(self, discrimination_counts, total_counts, 
                                       n_bootstrap=1000, confidence=0.95):
        """
        Bootstrap confidence interval for discrimination rate.
        """
        if total_counts == 0:
            return 0.0, 0.0, 0.0
        
        observed_rate = discrimination_counts / total_counts
        
        bootstrap_rates = []
        bootstrap_rng = np.random.RandomState(self.random_seed)
        for _ in range(n_bootstrap):
            bootstrap_sample = bootstrap_rng.binomial(1, observed_rate, total_counts)
            bootstrap_rates.append(np.mean(bootstrap_sample))
        
        bootstrap_rates = np.array(bootstrap_rates)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_rates, 100 * alpha / 2)
        upper = np.percentile(bootstrap_rates, 100 * (1 - alpha / 2))
        
        return observed_rate, lower, upper
    
    def _permutation_test(self, discrimination_counts, total_counts, 
                          n_permutations=1000, null_rate=0.0):
        """
        Permutation test for statistical significance.
        """
        if total_counts == 0:
            return 1.0
        
        observed_rate = discrimination_counts / total_counts
        
        null_distribution = []
        perm_rng = np.random.RandomState(self.random_seed + 1)
        for _ in range(n_permutations):
            perm_sample = perm_rng.binomial(1, null_rate, total_counts)
            null_distribution.append(np.mean(perm_sample))
        
        null_distribution = np.array(null_distribution)
        
        p_value = np.mean(np.abs(null_distribution - null_rate) >= 
                         np.abs(observed_rate - null_rate))
        
        return max(p_value, 1.0 / n_permutations)
    
    def _cohens_h(self, rate1, rate2):
        """
        Cohen's h effect size for difference between two proportions.
        """
        rate1_adj = np.clip(rate1, 0.001, 0.999)
        rate2_adj = np.clip(rate2, 0.001, 0.999)
        
        phi1 = 2 * np.arcsin(np.sqrt(rate1_adj))
        phi2 = 2 * np.arcsin(np.sqrt(rate2_adj))
        return phi1 - phi2
    
    def _minimum_detectable_effect(self, n, alpha=0.05, power=0.8):
        """
        Calculate minimum effect size detectable with given sample size.
        """
        z_alpha = st.norm.ppf(1 - alpha / 2)
        z_beta = st.norm.ppf(power)
        
        mde = (z_alpha + z_beta) * np.sqrt(2 / n)
        return mde
    
    def _bonferroni_correction(self, p_value, num_tests):
        """
        Bonferroni correction for multiple hypothesis testing.
        """
        return min(p_value * num_tests, 1.0)

    def _robust_stopping_condition(self, count, num_sampled, conf=0.95):
        """
        Robust stopping condition using multiple criteria.
        """
        if num_sampled < self.min_samples:
            return 0, False, {}
        
        discrimination_rate = count / num_sampled
        
        obs_rate, ci_lower, ci_upper = self._bootstrap_confidence_interval(
            count, num_sampled, confidence=conf
        )
        ci_width = ci_upper - ci_lower
        
        p_value = self._permutation_test(count, num_sampled)
        effect_size = self._cohens_h(discrimination_rate, 0.0)
        mde = self._minimum_detectable_effect(num_sampled)
        
        stats = {
            'discrimination_rate': discrimination_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'p_value': p_value,
            'effect_size': effect_size,
            'min_detectable_effect': mde,
            'n_samples': num_sampled
        }
        
        should_stop = (
            ci_width < 0.05 and 
            num_sampled >= self.min_samples * 2 and
            (discrimination_rate > 0.10 or p_value < 0.05)
        )
        
        return discrimination_rate, should_stop, stats

    # ========================================================================
    # MAIN DISCRIMINATION DETECTION
    # ========================================================================

    def causal_discrimination(self, protected_features, conf=0.95, 
                            num_permutations=1000):
        """
        Detect causal discrimination with advanced sampling and robust statistics.
        """
        assert protected_features
        count = 0
        test_cases = []
        causal_pairs = []
        
        fixed_features = [f for f in self.input_order if f not in protected_features]
        
        self._discrimination_density = {}
        
        for num_sampled in range(1, self.max_samples):
            current_rate = count / num_sampled if num_sampled > 0 else 0
            
            fixed_assignment = self._generate_assignment(
                fixed_features, num_sampled, current_rate
            )
            
            protected_assignment = self._generate_random_assignment(protected_features)
            
            original_case = {**fixed_assignment, **protected_assignment}
            test_cases.append(original_case.copy())
            
            original_prediction = self._get_prediction(original_case)
            
            discrimination_found = False
            for alt_protected_assignment in self._generate_all_assignments(protected_features):
                if alt_protected_assignment == protected_assignment:
                    continue
                    
                modified_case = {**fixed_assignment, **alt_protected_assignment}
                test_cases.append(modified_case.copy())
                
                if self._get_prediction(modified_case) != original_prediction:
                    count += 1
                    causal_pairs.append((original_case.copy(), modified_case.copy()))
                    discrimination_found = True
                    
                    density_key = tuple(sorted(fixed_assignment.items()))
                    self._discrimination_density[density_key] = \
                        self._discrimination_density.get(density_key, 0) + 1
                    # break
            
            discrimination_rate, should_stop, stats = self._robust_stopping_condition(
                count, num_sampled, conf
            )
            
            if should_stop:
                print(f"  Robust early stopping at {num_sampled} samples")
                print(f"  CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
                print(f"  p-value: {stats['p_value']:.4f}")
                print(f"  Effect size (Cohen's h): {stats['effect_size']:.4f}")
                break
        
        final_rate, ci_lower, ci_upper = self._bootstrap_confidence_interval(
            count, num_sampled, confidence=conf
        )
        p_value = self._permutation_test(count, num_sampled, n_permutations=num_permutations)
        effect_size = self._cohens_h(final_rate, 0.0)
        mde = self._minimum_detectable_effect(num_sampled)
        
        final_stats = {
            'discrimination_rate': final_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'p_value': p_value,
            'effect_size': effect_size,
            'min_detectable_effect': mde,
            'n_samples': num_sampled,
            'sampling_method': self.sampling_method,
            'is_significant': p_value < 0.05,
            'effect_interpretation': self._interpret_effect_size(effect_size)
        }
        
        return test_cases, final_rate, causal_pairs, final_stats

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _interpret_effect_size(self, cohens_h):
        """Interpret Cohen's h effect size."""
        abs_h = abs(cohens_h)
        if abs_h < 0.2:
            return "negligible"
        elif abs_h < 0.5:
            return "small"
        elif abs_h < 0.8:
            return "medium"
        else:
            return "large"

    def _generate_all_assignments(self, feature_names):
        if not feature_names:
            return [{}]
            
        feature_values = [self.inputs[name].values for name in feature_names]
        combinations = product(*feature_values)
        
        return [dict(zip(feature_names, combo)) for combo in combinations]

    def _get_prediction(self, assignment):
        cache_key = tuple(assignment[name] for name in self.input_order)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        prediction = self.model_predict_fn(assignment)
        self._cache[cache_key] = prediction
        
        return prediction


if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    from tensorflow.keras.models import load_model

    set_all_seeds(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)
    from utils.verif_utils import *

    # German Credit models
    model_1_path = 'Fairify/models/german/GC-8.h5'
    model_2_path = 'Fairify/models/german/GC-8-AdvDb.h5'
    model_3_path = 'Fairify/models/german/GC-8-Ruler.h5'
    model_4_path = 'Fairify/models/german/GC-8-Runner.h5'
    model_5_path = 'Fairify/models/german/GC-8-Neufair.h5'
    model_6_path = 'Fairify/models/german/GC-8-Retrained.h5'

    print("Loading models...")
    model_1 = load_model(model_1_path)
    model_2 = load_model(model_2_path)
    model_3 = load_model(model_3_path)
    model_4 = load_model(model_4_path)
    model_5 = load_model(model_5_path)
    model_6 = load_model(model_6_path)

    # Load German Credit data
    df, X_train, y_train, X_test, y_test, encoders = load_german()
    
    feature_names = [
        "status",
        "month",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "investment_as_income_percentage",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "installment_plans",
        "housing",
        "number_of_credits",
        "skill_level",
        "people_liable_for",
        "telephone",
        "foreign_worker",
        "sex"
    ]

    def build_predict_fn(model):
        def predict_fn(feature_dict):
            x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
            return int(model.predict(x, verbose=0)[0][0] > 0.5)
        return predict_fn

    def run_causal_eval(model_path, model, sampling_method='sobol'):
        print("\n" + "="*60)
        print(f"Model: {os.path.basename(model_path).replace('.h5', '')}")
        print(f"Sampling Method: {sampling_method}")
        print("="*60)
        
        detector = AdvancedCausalDiscriminationDetector(
            build_predict_fn(model),
            max_samples=10000,
            min_samples=500,
            sampling_method=sampling_method,
            random_seed=42
        )
        
        for fname in feature_names:
            unique_vals = sorted(set(df[fname]))
            detector.add_feature(fname, unique_vals)
        
        print(f"Running Advanced Causal Discrimination Check on 'sex' with {sampling_method} sampling...")
        _, rate, _, stats = detector.causal_discrimination(['sex'])
        
        print(f"\nRESULTS:")
        print(f"  Discrimination rate: {rate:.4f}")
        print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        print(f"  p-value: {stats['p_value']:.4f} {'(SIGNIFICANT)' if stats['is_significant'] else '(not significant)'}")
        print(f"  Effect size: {stats['effect_size']:.4f} ({stats['effect_interpretation']})")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Min Detectable Effect: {stats['min_detectable_effect']:.4f}")

    # Run all German Credit models with Sobol sampling
    print("\n" + "="*80)
    print("TESTING ALL GERMAN CREDIT MODELS WITH THEMIS++ (SOBOL SAMPLING)")
    print("="*80)
    
    run_causal_eval(model_1_path, model_1, sampling_method='sobol')
    run_causal_eval(model_2_path, model_2, sampling_method='sobol')
    run_causal_eval(model_3_path, model_3, sampling_method='sobol')
    run_causal_eval(model_4_path, model_4, sampling_method='sobol')
    run_causal_eval(model_5_path, model_5, sampling_method='sobol')
    run_causal_eval(model_6_path, model_6, sampling_method='sobol')