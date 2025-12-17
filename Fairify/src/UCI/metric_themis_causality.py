#!/usr/bin/env python3

from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
from scipy.stats import qmc
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import tensorflow as tf
import numpy as np
import os

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


class CausalDiscriminationDetector:
    def __init__(self, model_predict_fn, max_samples=1000, min_samples=100, random_seed=42):
        self.model_predict_fn = model_predict_fn
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.random_seed = random_seed
        self.inputs = {}
        self.input_order = []
        self._cache = {}
        
        self.rng = random.Random(self.random_seed)

    def add_feature(self, name, values, kind="categorical"):
        self.inputs[name] = Input(name, values, kind)
        self.input_order.append(name)

    def add_continuous_feature(self, name, min_val, max_val, num_values=10):
        values = [min_val + i * (max_val - min_val) / (num_values - 1) 
                 for i in range(num_values)]
        self.add_feature(name, values, "continuous")

    def causal_discrimination(self, protected_features, conf=0.999, margin=0.0001):
        assert protected_features
        count = 0
        test_cases = []
        causal_pairs = []
        
        fixed_features = [f for f in self.input_order if f not in protected_features]
        
        for num_sampled in range(1, self.max_samples):
            fixed_assignment = self._generate_random_assignment(fixed_features)
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
                    break
            
            discrimination_rate, should_stop = self._check_stopping_condition(
                count, num_sampled, conf, margin)
            
            if should_stop:
                break
        
        return test_cases, discrimination_rate, causal_pairs

    def discrimination_search(self, threshold=0.15, conf=0.99, margin=0.01):
        discriminatory_features = {}
        
        for combo_size in range(1, len(self.input_order)):
            for feature_combo in combinations(self.input_order, combo_size):
                if self._is_superset_discriminatory(discriminatory_features, feature_combo):
                    continue
                
                print(f"Testing feature combination: {feature_combo}")
                
                _, discrimination_rate, causal_pairs = self.causal_discrimination(
                    protected_features=list(feature_combo), 
                    conf=conf, 
                    margin=margin
                )
                
                if discrimination_rate > threshold:
                    discriminatory_features[feature_combo] = {
                        'rate': discrimination_rate,
                        'pairs': causal_pairs
                    }
                    print(f"  -> Discrimination found: {discrimination_rate:.1%}")
                else:
                    print(f"  -> No significant discrimination: {discrimination_rate:.1%}")
        
        return discriminatory_features

    def _generate_random_assignment(self, feature_names):
        return {name: self.inputs[name].get_random_value() for name in feature_names}

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

    def _check_stopping_condition(self, count, num_sampled, conf, margin):
        if num_sampled < self.min_samples:
            return 0, False
            
        discrimination_rate = count / num_sampled
        
        if discrimination_rate == 0 or discrimination_rate == 1:
            error = 0
        else:
            z_score = st.norm.ppf(conf)
            error = z_score * math.sqrt((discrimination_rate * (1 - discrimination_rate)) / num_sampled)
        
        return discrimination_rate, error < margin

    def _is_superset_discriminatory(self, discriminatory_features, feature_combo):
        for known_combo in discriminatory_features.keys():
            if set(known_combo).issubset(set(feature_combo)):
                return True
        return False

    def print_results(self, results):
        if not results:
            print("No discriminatory feature combinations found.")
            return
            
        print("\n" + "="*60)
        print("CAUSAL DISCRIMINATION RESULTS")
        print("="*60)
        
        for features, data in results.items():
            print(f"\nFeatures: {', '.join(features)}")
            print(f"Discrimination Rate: {data['rate']:.1%}")
            print(f"Number of discriminatory pairs: {len(data['pairs'])}")
            
            if data['pairs']:
                print("\nExample discriminatory cases:")
                for i, (orig, modified) in enumerate(data['pairs'][:3]):  
                    print(f"  Case {i+1}:")
                    print(f"    Original:  {orig}")
                    print(f"    Modified:  {modified}")
                if len(data['pairs']) > 3:
                    print(f"    ... and {len(data['pairs']) - 3} more")


def load_student():
    """
    Load and preprocess Student Performance dataset with discretization for easier CE generation.
    
    Returns:
        df, X_train, y_train, X_test, y_test
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import KBinsDiscretizer
    
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
    
    return (
        df,
        X_train.to_numpy(),
        y_train.to_numpy().astype('int'),
        X_test.to_numpy(),
        y_test.to_numpy().astype('int')
    )
                    
if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    from tensorflow.keras.models import load_model

    set_all_seeds(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)

    # model_1_path = 'Fairify/models/uci/UCI-5.h5'
    # model_2_path = 'Fairify/models/uci/UCI-5-AdvDb.h5'
    # model_3_path = 'Fairify/models/uci/UCI-5-Ruler.h5'
    # model_4_path = 'Fairify/models/uci/UCI-5-Runner.h5'
    # model_5_path = 'Fairify/models/uci/UCI-5-Neufair.h5'
    model_6_path = 'Fairify/models/uci/UCI-5-Retrained.h5'

    print("Loading models...")
    # model_1 = load_model(model_1_path)
    # model_2 = load_model(model_2_path)
    # model_3 = load_model(model_3_path)
    # model_4 = load_model(model_4_path)
    # model_5 = load_model(model_5_path)
    model_6 = load_model(model_6_path)

    df, X_train, y_train, X_test, y_test = load_student()
    
    # Feature names for UCI Student dataset
    feature_names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 
                     'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                     'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
                     'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
                     'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

    print("="*40)

    def array_to_feature_dict(arr):
        return {feature_names[i]: arr[i] for i in range(len(feature_names))}

    def build_predict_fn(model):
        def predict_fn(feature_dict):
            x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
            return int(model.predict(x, verbose=0)[0][0] > 0.5)
        return predict_fn

    def run_causal_eval(model_path, model):
        print("Model:", os.path.basename(model_path).replace('.h5', ''))
        print("Setting up detector...")
        detector = CausalDiscriminationDetector(build_predict_fn(model), max_samples=1000, min_samples=100)
        for fname in feature_names:
            unique_vals = sorted(set(df[fname]))
            detector.add_feature(fname, unique_vals)
        print("Running Causal Discrimination Check on 'sex'...\n")
        _, rate, _ = detector.causal_discrimination(['sex'])
        print(f"Discrimination rate: {rate:.4f}")
        print("="*40)

    # run_causal_eval(model_1_path, model_1)
    # run_causal_eval(model_2_path, model_2)
    # run_causal_eval(model_3_path, model_3)
    # run_causal_eval(model_4_path, model_4)
    # run_causal_eval(model_5_path, model_5)
    run_causal_eval(model_6_path, model_6)
    
# if __name__ == "__main__":
#     import sys
#     import os
    
#     set_all_seeds(42)

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
#     sys.path.append(src_dir)
#     from tensorflow.keras.models import load_model
#     import numpy as np

#     ORIGINAL_MODEL_NAME = "UCI-1"        
#     FAIRER_MODEL_NAME = "UCI-1-Ruler"
    
#     ORIGINAL_MODEL_PATH = f'Fairify/models/uci/{ORIGINAL_MODEL_NAME}.h5'
#     FAIRER_MODEL_PATH = f'Fairify/models/uci/{FAIRER_MODEL_NAME}.h5'

#     print("Loading models...")
#     original_model = load_model(ORIGINAL_MODEL_PATH)
#     fairer_model = load_model(FAIRER_MODEL_PATH)

#     df, X_train, y_train, X_test, y_test = load_student()
    
#     # Feature names for UCI Student dataset
#     feature_names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 
#                      'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
#                      'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
#                      'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
#                      'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

#     print("="*40)

#     def array_to_feature_dict(arr):
#         return {feature_names[i]: arr[i] for i in range(len(feature_names))}
    
#     def model_predict_fn_original(feature_dict):
#         x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
#         return int(original_model.predict(x, verbose=0)[0][0] > 0.5)

#     def model_predict_fn_fairer(feature_dict):
#         x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
#         return int(fairer_model.predict(x, verbose=0)[0][0] > 0.5)

#     print("Setting up detector...")
#     detector_orig = CausalDiscriminationDetector(model_predict_fn_original, max_samples=1000, min_samples=100)
#     detector_fair = CausalDiscriminationDetector(model_predict_fn_fairer, max_samples=1000, min_samples=100)

#     for fname in feature_names:
#         unique_vals = sorted(set(df[fname]))
#         detector_orig.add_feature(fname, unique_vals)
#         detector_fair.add_feature(fname, unique_vals)

#     print("Running Causal Discrimination Check on 'sex'...\n")
#     _, rate_orig, _ = detector_orig.causal_discrimination(['sex'])
#     _, rate_fair, _ = detector_fair.causal_discrimination(['sex'])

#     print(f"Discrimination rate on original model ({ORIGINAL_MODEL_NAME}): {rate_orig:.4f}")
#     print(f"Discrimination rate on fairer model   ({FAIRER_MODEL_NAME}): {rate_fair:.4f}")

#     print("="*40)