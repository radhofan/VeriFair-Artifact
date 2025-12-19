#!/usr/bin/env python3

from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
from scipy.stats import qmc
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def create_aif360_dataset(X, y, feature_names, protected_attribute='sex', 
                         favorable_label=1, unfavorable_label=0):
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    dataset = BinaryLabelDataset(
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        df=df,
        label_names=['label'],
        protected_attribute_names=[protected_attribute]
    )
    return dataset

def safe_metric_value(metric_value):
    """Safely extract scalar value from metric result."""
    if isinstance(metric_value, np.ndarray):
        if metric_value.size == 1:
            return metric_value.item()
        else:
            return np.mean(metric_value)
    return metric_value

def measure_fairness_aif360(model, X_test, y_test, feature_names, 
                           protected_attribute='sex', pa_col_idx=None):

    predictions = model.predict(X_test)
    pred_binary = (predictions > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, pred_binary)
    f1 = f1_score(y_test, pred_binary)
    
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    dataset_orig = create_aif360_dataset(X_test, y_test, feature_names, protected_attribute)
    dataset_pred = create_aif360_dataset(X_test, pred_binary, feature_names, protected_attribute)
    
    unprivileged_groups = [{protected_attribute: 0}]  # Female
    privileged_groups = [{protected_attribute: 1}]    # Male
    
    classified_metric = ClassificationMetric(
        dataset_orig, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    metric_pred = BinaryLabelDatasetMetric(
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    di = classified_metric.disparate_impact()
    spd = classified_metric.mean_difference()
    eod = classified_metric.equal_opportunity_difference()
    aod = classified_metric.average_odds_difference()
    erd = classified_metric.error_rate_difference()
    cnt = metric_pred.consistency()  
    ti = classified_metric.theil_index()
    
    print(f"\n=== FAIRNESS METRICS (AIF360) ===")
    print(f"Disparate Impact (DI):            {di:.3f}")
    print(f"Statistical Parity Difference:    {spd:.3f}")
    print(f"Equal Opportunity Difference:     {eod:.3f}")
    print(f"Average Odds Difference:          {aod:.3f}")
    print(f"Error Rate Difference:            {erd:.3f}")
    print(f"Consistency (CNT):                {float(cnt):.3f}")
    print(f"Theil Index:                      {ti:.3f}")
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'disparate_impact': di,
        'statistical_parity_diff': spd,
        'equal_opportunity_diff': eod,
        'average_odds_diff': aod,
        'error_rate_diff': erd,
        'consistency': float(cnt),
        'theil_index': ti
    }

def load_student():
    """
    Load and preprocess Student Performance dataset with discretization for easier CE generation.
    
    Returns:
        df, X_train, y_train, X_test, y_test
    """
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
    from tensorflow.keras.models import load_model

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)

    print("Loading models...")
    # model_1_path = 'Fairify/models/uci/UCI-5.h5'
    # model_2_path = 'Fairify/models/uci/UCI-5-AdvDb.h5'
    # model_3_path = 'Fairify/models/uci/UCI-5-Ruler.h5'
    # model_4_path = 'Fairify/models/uci/UCI-5-Runner.h5'
    # model_5_path = 'Fairify/models/uci/UCI-5-Neufair.h5'
    model_6_path = 'Fairify/models/uci/UCI-5-Retrained.h5'

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

    def evaluate_model(model_path, model):
        model_name = os.path.basename(model_path).replace('.h5', '')
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        print("Model:", model_name)
        print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        print("F1 Score:", round(f1_score(y_test, y_pred), 3))
        print("\n=== FAIRNESS METRICS (AIF360) ===")
        measure_fairness_aif360(model, X_test, y_test, feature_names, protected_attribute='sex')
        print("="*40)

    # evaluate_model(model_1_path, model_1)
    # evaluate_model(model_2_path, model_2)
    # evaluate_model(model_3_path, model_3)
    # evaluate_model(model_4_path, model_4)
    # evaluate_model(model_5_path, model_5)
    evaluate_model(model_6_path, model_6)


# if __name__ == "__main__":
#     import sys
#     import os
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

#     y_pred_orig = (original_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
#     y_pred_fair = (fairer_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

#     accuracy_orig = accuracy_score(y_test, y_pred_orig)
#     accuracy_fair = accuracy_score(y_test, y_pred_fair)

#     print(f"Original model accuracy: {accuracy_orig:.4f}")
#     print(f"Fairer model accuracy: {accuracy_fair:.4f}")

#     print("="*40)

#     print("\n=== ORIGINAL MODEL FAIRNESS (AIF360) ===")
#     original_metrics = measure_fairness_aif360(original_model, X_test, y_test, 
#                                             feature_names, protected_attribute='sex')

#     print("\n=== FAIRER MODEL FAIRNESS (AIF360) ===")
#     fairer_metrics = measure_fairness_aif360(fairer_model, X_test, y_test, 
#                                             feature_names, protected_attribute='sex')

#     print("="*40)