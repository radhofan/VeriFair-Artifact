#!/usr/bin/env python3
"""
Directed Combinatorial Mutation Detector with Gradient Guidance
Starts base cases from the real German dataset and uses gradient-directed
refinement of t-way combinatorial mutations to more efficiently find
discriminatory counterfactuals (IDI).

Notes:
- Assumes the Keras models produce a single probability output (sigmoid).
- Protected features are treated as categorical/discrete in projection step.
- Continuous features will be nudged then snapped to the closest allowed discrete value
  present in the provided Input.values grid.
- This is a self-contained script mirroring your prior pipeline but replaced
  input/value handling to preserve numeric types and added gradient guidance.
"""

from itertools import combinations, product
import math
import random
import scipy.stats as st
import tensorflow as tf
import numpy as np
import os

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

class Input:
    """Feature descriptor that preserves original value types (numbers or strings)."""
    def __init__(self, name, values, kind="categorical"):
        self.name = name
        # keep original types (no str coercion)
        # convert numpy scalars to native python
        clean = []
        for v in values:
            if isinstance(v, (np.generic,)):
                v = v.item()
            clean.append(v)
        # ensure unique-preserving order
        seen = set()
        uniq = []
        for v in clean:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        self.values = uniq
        self.kind = kind

    def get_random_value(self):
        return random.choice(self.values)

    def __str__(self):
        return f"Feature: {self.name}, Values: {self.values}"

def generate_pairwise_combinations(parameters):
    """Simple all-pairs generator (attempts randomized selection then exhaustive fallback)."""
    if not parameters:
        yield ()
        return
    n_params = len(parameters)
    if n_params == 1:
        for v in parameters[0]:
            yield (v,)
        return
    covered_pairs = set()
    all_pairs_needed = set()
    for i in range(n_params):
        for j in range(i+1, n_params):
            for vi in parameters[i]:
                for vj in parameters[j]:
                    all_pairs_needed.add((i, j, vi, vj))
    max_attempts = max(1000, len(parameters[0]) * len(parameters[1]) * 10)
    attempt = 0
    while covered_pairs != all_pairs_needed and attempt < max_attempts:
        combo = tuple(random.choice(p) for p in parameters)
        new_pairs = set()
        for i in range(n_params):
            for j in range(i+1, n_params):
                pair = (i, j, combo[i], combo[j])
                if pair not in covered_pairs:
                    new_pairs.add(pair)
        if new_pairs:
            yield combo
            covered_pairs.update(new_pairs)
        attempt += 1
    if covered_pairs != all_pairs_needed:
        for combo in product(*parameters):
            pairs_in_combo = set()
            for i in range(n_params):
                for j in range(i+1, n_params):
                    pairs_in_combo.add((i, j, combo[i], combo[j]))
            if not pairs_in_combo.issubset(covered_pairs):
                yield combo
                covered_pairs.update(pairs_in_combo)
                if covered_pairs == all_pairs_needed:
                    break

def generate_t_way_mutations_list(base_case, protected_features, all_features, t=2):
    """
    Enumerate combinatorial t-way mutations for protected_features starting from base_case.
    base_case and returned mutations are dicts mapping feature->value (native types).
    """
    if not protected_features:
        return []
    mutations = []
    protected_values = [all_features[f].values for f in protected_features]
    for feature_indices in combinations(range(len(protected_features)), min(t, len(protected_features))):
        selected_features = [protected_features[i] for i in feature_indices]
        selected_values = [protected_values[i] for i in feature_indices]
        for value_combo in product(*selected_values):
            mutated_case = base_case.copy()
            changed = False
            for feat_name, new_value in zip(selected_features, value_combo):
                if mutated_case.get(feat_name) != new_value:
                    changed = True
                mutated_case[feat_name] = new_value
            if changed:
                mutations.append(mutated_case)
    return mutations

class CausalDiscriminationDetector:
    """
    Detector that:
    - draws base cases from a real dataset (set_real_dataset)
    - generates combinatorial t-way mutations on protected features
    - refines each mutation using gradient guidance to push toward a prediction flip
    - batches predictions for efficiency
    """
    def __init__(self, model_predict_fn, model, feature_names,
                 max_samples=1000, min_samples=100, random_seed=42,
                 mutation_strength=2, batch_size=256, grad_steps=6, step_size=0.1):
        self.model_predict_fn = model_predict_fn
        self.model = model
        self.feature_names = list(feature_names)
        self.feature_index = {f: i for i, f in enumerate(self.feature_names)}
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.random_seed = random_seed
        self.mutation_strength = mutation_strength
        self.batch_size = batch_size
        self.grad_steps = grad_steps
        self.step_size = step_size
        self.inputs = {}
        self.input_order = []
        self._cache = {}
        self.real_dataset_rows = None
        random.seed(random_seed)

    def set_real_dataset(self, df):
        """
        Convert dataframe rows into list of dicts preserving numeric types.
        Expects df contains the exact columns in self.feature_names.
        """
        rows = []
        for _, row in df[self.feature_names].iterrows():
            d = {}
            for f in self.feature_names:
                v = row[f]
                if isinstance(v, (np.generic,)):
                    v = v.item()
                d[f] = v
            rows.append(d)
        self.real_dataset_rows = rows

    def add_feature(self, name, values, kind="categorical"):
        self.inputs[name] = Input(name, values, kind)
        self.input_order.append(name)

    def add_continuous_feature(self, name, min_val, max_val, num_values=10):
        vals = [min_val + i * (max_val - min_val) / (num_values - 1) for i in range(num_values)]
        self.add_feature(name, vals, "continuous")

    def _dict_to_array(self, feature_dict):
        """Return numpy 1D array float32 in order self.feature_names"""
        return np.array([float(feature_dict[f]) for f in self.feature_names], dtype=np.float32)

    def _batch_predict(self, feature_dicts):
        if not feature_dicts:
            return np.array([], dtype=int)
        X = np.array([self._dict_to_array(fd) for fd in feature_dicts], dtype=np.float32)
        preds = self.model.predict(X, verbose=0, batch_size=self.batch_size)
        return (preds.flatten() > 0.5).astype(int)

    def _tensor_from_dict(self, d):
        """Return a tf.Variable 1D tensor for gradient operations"""
        arr = np.array([float(d[f]) for f in self.feature_names], dtype=np.float32)
        return tf.Variable(arr, dtype=tf.float32)

    def _dict_from_tensor(self, t):
        """Convert tensor/numpy array back to feature dict (native python types)"""
        arr = t.numpy().astype(float).tolist()
        return {self.feature_names[i]: arr[i] for i in range(len(self.feature_names))}

    def _project_value_to_allowed(self, feature_name, raw_val):
        """
        For a feature, pick the allowed discrete value in Input.values closest to raw_val.
        If values are non-numeric (strings), choose the value whose float cast is closest
        if possible; else prefer exact match if raw_val equals one of allowed values.
        """
        allowed = self.inputs[feature_name].values
        # if allowed are numeric
        try:
            allowed_nums = [float(a) for a in allowed]
            rv = float(raw_val)
            # pick index with min absolute difference
            diffs = [abs(rv - an) for an in allowed_nums]
            idx = int(np.argmin(diffs))
            chosen = allowed[idx]
            # convert to native type if numpy scalar
            if isinstance(chosen, (np.generic,)):
                chosen = chosen.item()
            return chosen
        except Exception:
            # fallback: if raw_val equals an allowed choice, return it
            for a in allowed:
                if a == raw_val:
                    return a
            # else return a random allowed (should be rare)
            return random.choice(allowed)

    def gradient_guided_refine(self, mutated_case, base_case, protected_features, base_pred):
        """
        Refine a mutated_case using gradient guidance to increase chance of flipping.
        - mutated_case: dict (initial combinatorial mutation)
        - base_case: dict (original base case)
        - protected_features: list of names to allow updates on (others frozen)
        - base_pred: integer 0 or 1, prediction for base_case
        Returns a new dict (projected to allowed values).
        """
        # create variable for full input vector
        x = self._tensor_from_dict(mutated_case)
        # mask of indices that are protected (allowed to change)
        prot_indices = [self.feature_index[f] for f in protected_features]
        prot_mask = np.zeros(len(self.feature_names), dtype=np.bool_)
        for idx in prot_indices:
            prot_mask[idx] = True
        prot_mask_tf = tf.constant(prot_mask)

        # Determine optimization direction: if base_pred==0 we want to increase prob, else decrease
        for step in range(self.grad_steps):
            with tf.GradientTape() as tape:
                tape.watch(x)
                x_in = tf.reshape(x, (1, -1))
                prob = tf.squeeze(self.model(x_in, training=False))
                # clamp to scalar
                prob = tf.cast(prob, tf.float32)
                # loss: push probability away from base_pred towards flip
                if base_pred == 0:
                    loss = -prob  # minimize negative prob => maximize prob
                else:
                    loss = prob   # minimize prob
            grads = tape.gradient(loss, x)
            if grads is None:
                break
            grads_np = grads.numpy()
            # apply update only on protected indices
            update = np.zeros_like(grads_np, dtype=np.float32)
            # use sign step
            update[prot_mask] = self.step_size * np.sign(grads_np[prot_mask])
            # update variable
            x.assign_add(tf.convert_to_tensor(update))
            # Project protected coords back to nearest allowed discrete value each step
            x_np = x.numpy()
            for f in protected_features:
                idx = self.feature_index[f]
                proj = self._project_value_to_allowed(f, x_np[idx])
                x_np[idx] = float(proj)
            # Assign projected vector back
            x.assign(tf.convert_to_tensor(x_np, dtype=tf.float32))

        refined = self._dict_from_tensor(x)
        # Ensure non-protected features equal base_case (we freeze them)
        for f in self.feature_names:
            if f not in protected_features:
                refined[f] = base_case[f]
        # Cast discrete values to original types from inputs list where appropriate
        for f in protected_features:
            refined[f] = self._project_value_to_allowed(f, refined[f])
        return refined

    def causal_discrimination(self, protected_features, conf=0.999, margin=0.0001):
        assert protected_features, "Must specify protected features to test"
        assert self.real_dataset_rows is not None, "Call set_real_dataset(df) before running"

        count = 0
        test_cases = []
        causal_pairs = []

        fixed_features = [f for f in self.input_order if f not in protected_features]

        if fixed_features:
            fixed_feature_values = [self.inputs[f].values for f in fixed_features]
            base_case_generator = generate_pairwise_combinations(fixed_feature_values)
        else:
            base_case_generator = [()]

        base_cases_batch = []
        all_mutations_batch = []
        base_case_indices = []
        num_base_cases = 0
        current_index = 0
        should_stop = False

        for combo in base_case_generator:
            if num_base_cases >= self.max_samples:
                break
            if current_index >= len(self.real_dataset_rows):
                break

            # Use real dataset row as the base case (warm start)
            real_base = self.real_dataset_rows[current_index]
            current_index += 1
            num_base_cases += 1
            base_case = real_base.copy()

            # generate combinatorial t-way mutations for protected features
            mutations = generate_t_way_mutations_list(
                base_case, protected_features, self.inputs, t=self.mutation_strength
            )

            # For each combinatorial mutation, perform gradient-guided refinement
            guided_mutations = []
            # compute base prediction once
            base_pred = self._batch_predict([base_case])[0]  # 0 or 1
            for m in mutations:
                # refine guided candidate
                guided = self.gradient_guided_refine(m, base_case, protected_features, base_pred)
                # only add if different from base_case
                if guided != base_case:
                    guided_mutations.append(guided)

            if guided_mutations:
                base_cases_batch.append(base_case)
                all_mutations_batch.extend(guided_mutations)
                base_case_indices.append((len(base_cases_batch) - 1,
                                          len(all_mutations_batch) - len(guided_mutations),
                                          len(all_mutations_batch)))

            if len(all_mutations_batch) >= self.batch_size or num_base_cases >= self.max_samples:
                base_predictions = self._batch_predict(base_cases_batch)
                mutation_predictions = self._batch_predict(all_mutations_batch)

                for idx, (base_idx, mut_start, mut_end) in enumerate(base_case_indices):
                    base_case_local = base_cases_batch[base_idx]
                    base_pred_local = int(base_predictions[base_idx])
                    test_cases.append(base_case_local)

                    mut_preds = mutation_predictions[mut_start:mut_end]
                    mutations_for_base = all_mutations_batch[mut_start:mut_end]
                    discriminatory_idx = np.where(mut_preds != base_pred_local)[0]

                    if len(discriminatory_idx) > 0:
                        count += 1
                        first_disc_idx = int(discriminatory_idx[0])
                        mutated_case = mutations_for_base[first_disc_idx]
                        causal_pairs.append((base_case_local.copy(), mutated_case.copy()))
                        test_cases.extend(mutations_for_base[:first_disc_idx + 1])
                    else:
                        test_cases.extend(mutations_for_base)

                discrimination_rate, should_stop = self._check_stopping_condition(
                    count, len(base_cases_batch), conf, margin)

                if should_stop:
                    break

                base_cases_batch = []
                all_mutations_batch = []
                base_case_indices = []

        # Process remaining batch
        if base_cases_batch and not should_stop:
            base_predictions = self._batch_predict(base_cases_batch)
            mutation_predictions = self._batch_predict(all_mutations_batch)

            for idx, (base_idx, mut_start, mut_end) in enumerate(base_case_indices):
                base_case_local = base_cases_batch[base_idx]
                base_pred_local = int(base_predictions[base_idx])
                test_cases.append(base_case_local)

                mut_preds = mutation_predictions[mut_start:mut_end]
                mutations_for_base = all_mutations_batch[mut_start:mut_end]
                discriminatory_idx = np.where(mut_preds != base_pred_local)[0]

                if len(discriminatory_idx) > 0:
                    count += 1
                    first_disc_idx = int(discriminatory_idx[0])
                    mutated_case = mutations_for_base[first_disc_idx]
                    causal_pairs.append((base_case_local.copy(), mutated_case.copy()))
                    test_cases.extend(mutations_for_base[:first_disc_idx + 1])
                else:
                    test_cases.extend(mutations_for_base)

        discrimination_rate = count / num_base_cases if num_base_cases > 0 else 0
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

if __name__ == "__main__":
    # Example execution using your existing load_german and models
    import sys
    import os
    from tensorflow.keras.models import load_model
    set_all_seeds(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)
    from utils.verif_utils import *  # expected to expose load_german

    model_1_path = 'Fairify/models/german/GC-1.h5'
    model_2_path = 'Fairify/models/german/GC-1-AdvDb.h5'
    model_3_path = 'Fairify/models/german/GC-1-Ruler.h5'
    model_4_path = 'Fairify/models/german/GC-1-Runner.h5'
    model_5_path = 'Fairify/models/german/GC-1-Neufair.h5'
    model_6_path = 'Fairify/models/german/GC-1-Retrained.h5'

    print("Loading models...")
    model_1 = load_model(model_1_path)
    model_2 = load_model(model_2_path)
    model_3 = load_model(model_3_path)
    model_4 = load_model(model_4_path)
    model_5 = load_model(model_5_path)
    model_6 = load_model(model_6_path)

    df, X_train, y_train, X_test, y_test, encoders = load_german()

    feature_names = [
        "status","month","credit_history","purpose","credit_amount","savings","employment",
        "investment_as_income_percentage","other_debtors","residence_since","property","age",
        "installment_plans","housing","number_of_credits","skill_level","people_liable_for",
        "telephone","foreign_worker","sex"
    ]

    def build_predict_fn(model):
        def predict_fn(feature_dict):
            x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
            return int(model.predict(x, verbose=0)[0][0] > 0.5)
        return predict_fn

    def run_causal_eval(model_path, model):
        print("Model:", os.path.basename(model_path).replace('.h5', ''))
        detector = CausalDiscriminationDetector(
            build_predict_fn(model),
            model=model,
            feature_names=feature_names,
            max_samples=1000,
            min_samples=100,             # lower min for quicker runs; adjust as needed
            mutation_strength=3,        # 3-way combinatorial mutations
            batch_size=512,
            grad_steps=6,
            step_size=0.5               # step size for gradient nudging; tune as needed
        )

        for fname in feature_names:
            unique_vals = sorted(set(df[fname]))
            detector.add_feature(fname, unique_vals)

        detector.set_real_dataset(df)
        print("Running Causal Discrimination Check on 'age'...\n")
        _, rate, pairs = detector.causal_discrimination(['age'])
        print(f"Discrimination rate: {rate:.4f}")
        print(f"Number of IDI pairs: {len(pairs)}")
        print("="*40)

    run_causal_eval(model_1_path, model_1)
    run_causal_eval(model_2_path, model_2)
    run_causal_eval(model_3_path, model_3)
    run_causal_eval(model_4_path, model_4)
    run_causal_eval(model_5_path, model_5)
    run_causal_eval(model_6_path, model_6)
