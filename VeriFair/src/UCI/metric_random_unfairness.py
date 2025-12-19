#!/usr/bin/env python3
import torch
import numpy as np
import itertools
from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
from scipy.stats import qmc
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


def clustering(data, c_num):
    from sklearn import cluster
    kmeans = cluster.KMeans(n_clusters=c_num)
    y_pred = kmeans.fit_predict(data)
    return [data[y_pred == n] for n in range(c_num)]


def clip(instance, constraint):
    return np.minimum(constraint[:, 1], np.maximum(constraint[:, 0], instance))


def random_pick(probability):
    random_number = np.random.rand()
    current_proba = 0
    for i in range(len(probability)):
        current_proba += probability[i]
        if current_proba > random_number:
            return i


def get_seed(clustered_data, X_len, c_num, cluster_i, fashion='RoundRobin'):
    if fashion == 'RoundRobin':
        index = np.random.randint(0, len(clustered_data[cluster_i]))
        return clustered_data[cluster_i][index]
    elif fashion == 'Distribution':
        pick_probability = [len(clustered_data[i]) / X_len for i in range(c_num)]
        x = clustered_data[random_pick(pick_probability)]
        index = np.random.randint(0, len(x))
        return x[index]


def similar_set(x, num_attribs, protected_attribs, constraint):
    similar_x = np.empty(shape=(0, num_attribs))

    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(constraint[i][0], constraint[i][1] + 1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for comb in all_combs:
        x_new = x.copy()
        for a, c in zip(protected_attribs, comb):
            x_new[a] = c
        similar_x = np.append(similar_x, [x_new], axis=0)
    return similar_x


def is_discriminatory(x, similar_x, model):
    x_input = np.array(x).reshape(1, -1)
    logit = model(x_input).numpy()
    
    if logit.shape[1] > 1:
        y_pred = np.argmax(logit, axis=1)[0]
    else:
        y_pred = (logit > 0.5).astype(int)[0][0]
    
    for x_new in similar_x:
        x_new_input = np.array(x_new).reshape(1, -1)
        logit_similar = model(x_new_input).numpy()
        
        if logit_similar.shape[1] > 1:
            y_pre_similar = np.argmax(logit_similar, axis=1)[0]
        else:
            y_pre_similar = (logit_similar > 0.5).astype(int)[0][0]
            
        if y_pre_similar != y_pred:
            return True
    return False


def max_diff(x, similar_x, model):
    x_input = np.array(x).reshape(1, -1)
    y_pred_proba = model(x_input).numpy()

    def distance(x_new):
        x_new_input = np.array(x_new).reshape(1, -1)
        return np.sum(np.square(y_pred_proba - model(x_new_input).numpy()))

    max_dist = 0.0
    x_potential_pair = x.copy()
    for x_new in similar_x:
        if distance(x_new) > max_dist:
            max_dist = distance(x_new)
            x_potential_pair = x_new.copy()
    return x_potential_pair


def find_pair(x, similar_x, model):
    pairs = np.empty(shape=(0, len(x)))
    x_input = np.array(x).reshape(1, -1)
    y_pred = (model(x_input).numpy() > 0.5)
    
    for x_pair in similar_x:
        x_pair_input = np.array(x_pair).reshape(1, -1)
        if (model(x_pair_input).numpy() > 0.5) != y_pred:
            pairs = np.append(pairs, [x_pair], axis=0)
    
    selected_p = random_pick([1.0 / pairs.shape[0]] * pairs.shape[0])
    return pairs[selected_p]


def normalization(grad1, grad2, protected_attribs, epsilon):
    gradient = np.zeros_like(grad1)
    grad1 = np.abs(grad1)
    grad2 = np.abs(grad2)
    for i in range(len(gradient)):
        saliency = grad1[i] + grad2[i]
        gradient[i] = 1.0 / (saliency + epsilon)
        if i in protected_attribs:
            gradient[i] = 0.0
    gradient_sum = np.sum(gradient)
    probability = gradient / gradient_sum
    return probability


def purely_random(num_attribs, protected_attribs, constraint, model, gen_num):
    gen_id = np.empty(shape=(0, num_attribs))
    for i in range(gen_num):
        x_picked = [0] * num_attribs
        for a in range(num_attribs):
            x_picked[a] = np.random.randint(constraint[a][0], constraint[a][1] + 1)
        similar_set_data = similar_set(x_picked, num_attribs, protected_attribs, constraint)
        if is_discriminatory(x_picked, similar_set_data, model):
            gen_id = np.append(gen_id, [x_picked], axis=0)
    return gen_id


def ids_percentage(sample_round, num_gen, num_attribs, protected_attribs, constraint, model):
    """
    Compute the percentage of individual discriminatory instances with 95% confidence
    """
    statistics = np.empty(shape=(0,))
    for i in range(sample_round):
        gen_id = purely_random(num_attribs, protected_attribs, constraint, model, num_gen)
        percentage = len(gen_id) / num_gen
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)
    return avg, interval


class FairnessEvaluator:
    """
    Simple fairness evaluator for individual discrimination using generation utilities
    """
    
    def __init__(self, model, constraint, protected_attribs=None, num_attribs=30):
        self.model = model
        self.constraint = constraint
        self.protected_attribs = protected_attribs if protected_attribs else [1]  # sex is at index 1
        self.num_attribs = num_attribs
    
    def evaluate_individual_fairness(self, sample_round=10, num_gen=100):
        """
        Evaluate individual fairness using generation utilities
        """
        return ids_percentage(sample_round, num_gen, self.num_attribs, 
                            self.protected_attribs, self.constraint, self.model)


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
    from utils.verif_utils import *
    
    # Define all model paths
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
    print("="*40)
    
    # Constraints for UCI Student dataset (30 features)
    constraint = np.array([
        [0, 1],      # school (0=MS, 1=GP)
        [0, 1],      # sex (0=F, 1=M)
        [0, 1],      # age (binned to 0-1)
        [0, 1],      # address (0=R, 1=U)
        [0, 1],      # famsize (0=LE3, 1=GT3)
        [0, 1],      # Pstatus (0=A, 1=T)
        [0, 4],      # Medu (0-4 scale)
        [0, 4],      # Fedu (0-4 scale)
        [0, 4],      # Mjob (job encoding)
        [0, 4],      # Fjob (job encoding)
        [0, 3],      # reason (0-3)
        [0, 2],      # guardian (0-2)
        [1, 4],      # traveltime (1-4 scale)
        [1, 4],      # studytime (1-4 scale)
        [0, 3],      # failures (0-3)
        [0, 1],      # schoolsup (binary)
        [0, 1],      # famsup (binary)
        [0, 1],      # paid (binary)
        [0, 1],      # activities (binary)
        [0, 1],      # nursery (binary)
        [0, 1],      # higher (binary)
        [0, 1],      # internet (binary)
        [0, 1],      # romantic (binary)
        [1, 5],      # famrel (1-5 scale)
        [1, 5],      # freetime (1-5 scale)
        [1, 5],      # goout (1-5 scale)
        [1, 5],      # Dalc (1-5 scale)
        [1, 5],      # Walc (1-5 scale)
        [1, 5],      # health (1-5 scale)
        [0, 3],      # absences (binned to 0-4)
    ])
    
    def evaluate_model(model_path, model):
        print(f"Model: {os.path.basename(model_path).replace('.h5', '')}")
        
        # Individual Fairness Evaluation
        evaluator = FairnessEvaluator(model, constraint, protected_attribs=[1], num_attribs=30)
        evaluator.evaluate_individual_fairness()
        
        print("="*40)
    
    print("Using FairnessEvaluator class:\n")
    
    # evaluate_model(model_1_path, model_1)
    # evaluate_model(model_2_path, model_2)
    # evaluate_model(model_3_path, model_3)
    # evaluate_model(model_4_path, model_4)
    # evaluate_model(model_5_path, model_5)
    evaluate_model(model_6_path, model_6)

# if __name__ == "__main__":
#     import sys
#     import os

#     set_all_seeds(42)

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
#     sys.path.append(src_dir)
    
#     from tensorflow.keras.models import load_model

#     ORIGINAL_MODEL_NAME = "UCI-1"        
#     FAIRER_MODEL_NAME = "UCI-1-Ruler"

#     ORIGINAL_MODEL_PATH = f'Fairify/models/uci/{ORIGINAL_MODEL_NAME}.h5'
#     FAIRER_MODEL_PATH = f'Fairify/models/uci/{FAIRER_MODEL_NAME}.h5'
    
#     print("Loading models...")
#     original_model = load_model(ORIGINAL_MODEL_PATH)
#     fairer_model = load_model(FAIRER_MODEL_PATH)
    
#     df, X_train, y_train, X_test, y_test = load_student()

#     print("="*40)
    
#     # Constraints for UCI Student dataset (30 features)
#     constraint = np.array([
#         [0, 1],      # school (0=MS, 1=GP)
#         [0, 1],      # sex (0=F, 1=M)
#         [0, 1],      # age (binned to 0-1)
#         [0, 1],      # address (0=R, 1=U)
#         [0, 1],      # famsize (0=LE3, 1=GT3)
#         [0, 1],      # Pstatus (0=A, 1=T)
#         [0, 4],      # Medu (0-4 scale)
#         [0, 4],      # Fedu (0-4 scale)
#         [0, 4],      # Mjob (job encoding)
#         [0, 4],      # Fjob (job encoding)
#         [0, 3],      # reason (0-3)
#         [0, 2],      # guardian (0-2)
#         [1, 4],      # traveltime (1-4 scale)
#         [1, 4],      # studytime (1-4 scale)
#         [0, 3],      # failures (0-3)
#         [0, 1],      # schoolsup (binary)
#         [0, 1],      # famsup (binary)
#         [0, 1],      # paid (binary)
#         [0, 1],      # activities (binary)
#         [0, 1],      # nursery (binary)
#         [0, 1],      # higher (binary)
#         [0, 1],      # internet (binary)
#         [0, 1],      # romantic (binary)
#         [1, 5],      # famrel (1-5 scale)
#         [1, 5],      # freetime (1-5 scale)
#         [1, 5],      # goout (1-5 scale)
#         [1, 5],      # Dalc (1-5 scale)
#         [1, 5],      # Walc (1-5 scale)
#         [1, 5],      # health (1-5 scale)
#         [0, 3],      # absences (binned to 0-4)
#     ])
    
#     print("Using FairnessEvaluator class:")
#     print("Original Model:")
#     original_evaluator = FairnessEvaluator(original_model, constraint, protected_attribs=[1], num_attribs=30)
#     original_evaluator.evaluate_individual_fairness()
    
#     print("\nFairer Model:")
#     fairer_evaluator = FairnessEvaluator(fairer_model, constraint, protected_attribs=[1], num_attribs=30)
#     fairer_evaluator.evaluate_individual_fairness()

#     print("="*40)