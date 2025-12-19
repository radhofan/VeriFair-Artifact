# Test AIF360 Fair Classifiers - German Credit Dataset
if __name__ == "__main__":
    import sys
    import os
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import logging
    import warnings
   
    # === IMPORTANT: Disable eager execution BEFORE loading Keras model ===
    import tensorflow.compat.v1 as tf
    import tensorflow.keras as keras
    tf.disable_eager_execution()
    
    # === Load paths and project structure ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    sys.path.append(src_dir)
    from utils.verif_utils import *
    from metric_aif360 import measure_fairness_aif360
    from metric_themis_causality import CausalDiscriminationDetector, set_all_seeds
    from metric_random_unfairness import FairnessEvaluator, set_all_seeds
    from aif360.datasets import BinaryLabelDataset
   
    set_all_seeds(42)
   
    # === German Credit preprocessing function ===
    def german_custom_preprocessing(df):
        """Custom preprocessing for German Credit dataset."""
        return df
    
    def load_german(drop_personal_status=False):
        """
        Load German Credit dataset.
        
        Args:
            drop_personal_status: If True, drop personal_status column (for 19 features).
                                 If False, keep it (for 20 features).
        """
        filepath = 'Fairify/data/german/german.data'
        column_names = ['status', 'month', 'credit_history',
                    'purpose', 'credit_amount', 'savings', 'employment',
                    'investment_as_income_percentage', 'personal_status',
                    'other_debtors', 'residence_since', 'property', 'age',
                    'installment_plans', 'housing', 'number_of_credits',
                    'skill_level', 'people_liable_for', 'telephone',
                    'foreign_worker', 'credit']
        na_values=[]
        df = pd.read_csv(filepath, sep=' ', header=None, names=column_names, na_values=na_values)
        df['age'] = df['age'].apply(lambda x: float(x >= 26))
        df = german_custom_preprocessing(df)
        
        # Encode personal_status before potentially dropping it
        encoders = {}
        cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 
                    'other_debtors', 'property', 'installment_plans',
                    'housing', 'skill_level', 'telephone', 'foreign_worker', 'personal_status']
       
        for f in cat_feat:
            label = LabelEncoder()
            df[f] = label.fit_transform(df[f])
            encoders[f] = label
        
        # Optionally drop personal_status after encoding
        if drop_personal_status:
            df = df.drop(['personal_status'], axis=1)
          
        label_name = 'credit'
        
        # Convert credit labels to 0/1 format (German Credit uses 1=good, 2=bad)
        # We map: 1 (good credit) -> 1, 2 (bad credit) -> 0
        df[label_name] = df[label_name].apply(lambda x: 1 if x == 1 else 0)
        
        X = df.drop(labels = [label_name], axis = 1, inplace = False)
        y = df[label_name]
        
        seed = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)        
        return (df, X_train.to_numpy(), y_train.to_numpy().astype('int'), 
                X_test.to_numpy(), y_test.to_numpy().astype('int'), encoders)
   
    # === Prediction Wrapper for AIF360 models ===
    class PredictionWrapper:
        """Wrapper to make pre-computed predictions compatible with measure_fairness_aif360."""
        def __init__(self, predictions):
            self.predictions = predictions
        
        def predict(self, X):
            return self.predictions.reshape(-1, 1)
    
    # === Causal Discrimination Wrapper ===
    class AIF360CausalWrapper:
        """Wrapper to make AIF360 models work with causal discrimination detector."""
        def __init__(self, aif360_model, feature_names):
            self.model = aif360_model
            self.feature_names = feature_names
        
        def __call__(self, feature_dict):
            x = np.array([[feature_dict[f] for f in self.feature_names]], dtype=np.float32)
            x = np.vstack([x, x])
            
            df_mini = pd.DataFrame(x, columns=self.feature_names)
            df_mini["label"] = [0, 0]
            dataset = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df_mini,
                label_names=["label"],
                protected_attribute_names=["age"],
                privileged_protected_attributes=[[1]],
                unprivileged_protected_attributes=[[0]]
            )
            
            pred_dataset = self.model.predict(dataset)
            pred_df, _ = pred_dataset.convert_to_dataframe()
            return int(pred_df["label"].values[0])
    
    # === Random Unfairness Wrapper ===
    class AIF360NumpyWrapper:
        """Wrapper to make AIF360 models work with random unfairness evaluator."""
        def __init__(self, aif360_model, feature_names):
            self.model = aif360_model
            self.feature_names = feature_names
        
        def __call__(self, x_input):
            if len(x_input.shape) == 1:
                x_input = x_input.reshape(1, -1)
            
            df_data = pd.DataFrame(x_input, columns=self.feature_names)
            df_data["label"] = [0] * len(x_input)
            dataset = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df_data,
                label_names=["label"],
                protected_attribute_names=["age"],
                privileged_protected_attributes=[[1]],
                unprivileged_protected_attributes=[[0]]
            )
            
            pred_dataset = self.model.predict(dataset)
            pred_df, _ = pred_dataset.convert_to_dataframe()
            predictions = pred_df["label"].values.reshape(-1, 1).astype(np.float32)
            
            class NumpyWrapper:
                def __init__(self, array):
                    self.array = array
                def numpy(self):
                    return self.array
            
            return NumpyWrapper(predictions)
    
    # === Custom Adversarial Debiasing that uses original model weights ===
    class AdversarialDebiasingWithTransfer:
        """Custom adversarial debiasing that transfers weights from original model."""
        
        def __init__(self, original_model, unprivileged_groups, privileged_groups, 
                     scope_name='adv_deb_transfer', num_epochs=50, batch_size=128):
            self.original_model = original_model
            self.unprivileged_groups = unprivileged_groups
            self.privileged_groups = privileged_groups
            self.scope_name = scope_name
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.sess = None
            
        def fit(self, dataset):
            """Train adversarial debiasing model initialized with original model weights."""
            tf.reset_default_graph()
            self.sess = tf.Session()
            
            # Get original model architecture and weights
            orig_weights = self.original_model.get_weights()
            print(f"\nOriginal model architecture:")
            for i, layer in enumerate(self.original_model.layers):
                print(f"  Layer {i}: {layer.__class__.__name__} - {layer.output_shape}")
            
            # Extract features and labels
            features, labels = dataset.features, dataset.labels.ravel()
            n_features = features.shape[1]
            
            # Build computational graph
            with tf.variable_scope(self.scope_name):
                # Input placeholders
                self.input_ph = tf.placeholder(tf.float32, shape=[None, n_features])
                self.label_ph = tf.placeholder(tf.float32, shape=[None])
                self.protected_ph = tf.placeholder(tf.float32, shape=[None])
                
                # Classifier network (dynamically match original architecture)
                with tf.variable_scope('classifier'):
                    x = self.input_ph
                    
                    # Process all layers except the last one
                    for layer_idx in range(len(self.original_model.layers) - 1):
                        layer = self.original_model.layers[layer_idx]
                        W, b = orig_weights[layer_idx * 2], orig_weights[layer_idx * 2 + 1]
                        
                        W_var = tf.get_variable(f'W{layer_idx}', initializer=tf.constant(W, dtype=tf.float32))
                        b_var = tf.get_variable(f'b{layer_idx}', initializer=tf.constant(b, dtype=tf.float32))
                        
                        x = tf.matmul(x, W_var) + b_var
                        x = tf.nn.relu(x)
                    
                    # Output layer
                    output_layer_idx = len(self.original_model.layers) - 1
                    W_out = orig_weights[output_layer_idx * 2]
                    b_out = orig_weights[output_layer_idx * 2 + 1]
                    
                    W_out_var = tf.get_variable(f'W{output_layer_idx}', 
                                                initializer=tf.constant(W_out, dtype=tf.float32))
                    b_out_var = tf.get_variable(f'b{output_layer_idx}', 
                                                initializer=tf.constant(b_out, dtype=tf.float32))
                    
                    logits = tf.matmul(x, W_out_var) + b_out_var
                    self.pred_probs = tf.nn.sigmoid(logits)
                    self.predictions = tf.cast(self.pred_probs > 0.5, tf.float32)
                
                # Adversary network
                with tf.variable_scope('adversary'):
                    adv_h1 = tf.layers.dense(self.pred_probs, 100, activation=tf.nn.relu)
                    adv_logits = tf.layers.dense(adv_h1, 1)
                    adv_probs = tf.nn.sigmoid(adv_logits)
                
                # Losses
                clf_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_ph, logits=tf.squeeze(logits))
                )
                adv_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_ph, logits=tf.squeeze(adv_logits))
                )
                
                self.total_loss = clf_loss - adv_loss
                
                # Optimizers
                clf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{self.scope_name}/classifier')
                adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{self.scope_name}/adversary')
                
                self.clf_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
                    self.total_loss, var_list=clf_vars
                )
                self.adv_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
                    adv_loss, var_list=adv_vars
                )
            
            # Initialize variables
            self.sess.run(tf.global_variables_initializer())
            
            # Extract protected attribute
            protected = dataset.protected_attributes[:, 0]
            
            # Training loop
            n_samples = len(features)
            print(f"\nTraining adversarial debiasing (transferring from original model)...")
            print(f"Architecture: {[layer.units for layer in self.original_model.layers]}")
            print(f"Epochs: {self.num_epochs}, Batch size: {self.batch_size}")
            
            for epoch in range(self.num_epochs):
                indices = np.random.permutation(n_samples)
                epoch_clf_loss = 0
                epoch_adv_loss = 0
                n_batches = 0
                
                for start_idx in range(0, n_samples, self.batch_size):
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    batch_features = features[batch_indices]
                    batch_labels = labels[batch_indices]
                    batch_protected = protected[batch_indices]
                    
                    feed_dict = {
                        self.input_ph: batch_features,
                        self.label_ph: batch_labels,
                        self.protected_ph: batch_protected
                    }
                    
                    # Train adversary
                    _, adv_loss_val = self.sess.run([self.adv_optimizer, adv_loss], feed_dict)
                    
                    # Train classifier
                    _, clf_loss_val = self.sess.run([self.clf_optimizer, self.total_loss], feed_dict)
                    
                    epoch_clf_loss += clf_loss_val
                    epoch_adv_loss += adv_loss_val
                    n_batches += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}: "
                          f"Classifier Loss={epoch_clf_loss/n_batches:.4f}, "
                          f"Adversary Loss={epoch_adv_loss/n_batches:.4f}")
        
        def predict(self, dataset):
            """Make predictions using the trained model."""
            features = dataset.features
            
            feed_dict = {self.input_ph: features}
            predictions = self.sess.run(self.predictions, feed_dict)
            
            pred_dataset = dataset.copy()
            pred_dataset.labels = predictions.reshape(-1, 1)
            
            return pred_dataset
        
        def save_as_keras(self, output_path):
            """Extract weights and save as Keras model with exact original architecture."""
            clf_vars = [v for v in tf.global_variables() 
                        if f'{self.scope_name}/classifier' in v.name 
                        and 'Adam' not in v.name]
            
            print("\nDebug: TensorFlow variables found:")
            for v in sorted(clf_vars, key=lambda x: x.name):
                print(f"  {v.name}: {v.shape}")
            
            weight_vars = sorted([v for v in clf_vars if '/W' in v.name], 
                                key=lambda v: int(v.name.split('/W')[1].split(':')[0]))
            bias_vars = sorted([v for v in clf_vars if '/b' in v.name], 
                            key=lambda v: int(v.name.split('/b')[1].split(':')[0]))
            
            weight_values = self.sess.run(weight_vars)
            bias_values = self.sess.run(bias_vars)
            
            n_features = weight_values[0].shape[0] if len(weight_values) > 0 else len(feature_names)
            
            layers = []
            num_layers = len(self.original_model.layers)
            
            for layer_idx in range(num_layers):
                units = self.original_model.layers[layer_idx].units
                
                if layer_idx == 0:
                    if layer_idx == num_layers - 1:
                        layers.append(keras.layers.Dense(units, activation='sigmoid', input_shape=(n_features,)))
                    else:
                        layers.append(keras.layers.Dense(units, activation='relu', input_shape=(n_features,)))
                else:
                    if layer_idx == num_layers - 1:
                        layers.append(keras.layers.Dense(units, activation='sigmoid'))
                    else:
                        layers.append(keras.layers.Dense(units, activation='relu'))
            
            keras_model = keras.Sequential(layers)
            keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            for layer_idx in range(num_layers):
                if layer_idx < len(weight_values) and layer_idx < len(bias_values):
                    keras_model.layers[layer_idx].set_weights([weight_values[layer_idx], bias_values[layer_idx]])
            
            keras_model.save(output_path)
            return keras_model
   
    # === Load German Credit dataset ===
    print("Loading German Credit dataset...")
    # First, try to load the model to check its input shape
    ORIGINAL_MODEL_NAME = "GC-10"
    
    # Load model in graph mode (after tf.disable_eager_execution())
    temp_model = keras.models.load_model(f'Fairify/models/german/{ORIGINAL_MODEL_NAME}.h5')
    expected_features = temp_model.input_shape[1]
    print(f"Model expects {expected_features} input features")
    
    # Load data accordingly
    if expected_features == 20:
        print("Loading with personal_status (20 features)")
        df, X_train, y_train, X_test, y_test, encoders = load_german(drop_personal_status=False)
        feature_names = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', 
                         'savings', 'employment', 'investment_as_income_percentage', 'personal_status',
                         'other_debtors', 'residence_since', 'property', 'age',
                         'installment_plans', 'housing', 'number_of_credits',
                         'skill_level', 'people_liable_for', 'telephone', 'foreign_worker']
    else:
        print("Loading without personal_status (19 features)")
        df, X_train, y_train, X_test, y_test, encoders = load_german(drop_personal_status=True)
        feature_names = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', 
                         'savings', 'employment', 'investment_as_income_percentage', 
                         'other_debtors', 'residence_since', 'property', 'age',
                         'installment_plans', 'housing', 'number_of_credits',
                         'skill_level', 'people_liable_for', 'telephone', 'foreign_worker']
    
    print(f"Loaded dataset with {X_train.shape[1]} features")
    
    # Debug: Check label values
    print(f"Training label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    print(f"Unique training labels: {np.unique(y_train)}")
    print(f"Unique test labels: {np.unique(y_test)}")
    
    # Constraint for German Credit (adjust based on number of features)
    if len(feature_names) == 20:
        # With personal_status
        constraint = np.array([
            [0, 4],    # status
            [0, 72],   # month
            [0, 5],    # credit_history
            [0, 10],   # purpose
            [0, 20000], # credit_amount
            [0, 5],    # savings
            [0, 5],    # employment
            [0, 4],    # investment_as_income_percentage
            [0, 4],    # personal_status
            [0, 3],    # other_debtors
            [0, 4],    # residence_since
            [0, 4],    # property
            [0, 1],    # age (protected attribute, binary)
            [0, 3],    # installment_plans
            [0, 3],    # housing
            [0, 4],    # number_of_credits
            [0, 4],    # skill_level
            [0, 2],    # people_liable_for
            [0, 2],    # telephone
            [0, 2]     # foreign_worker
        ])
        protected_idx = 12  # age is at index 12 when personal_status is included
    else:
        # Without personal_status (19 features)
        constraint = np.array([
            [0, 4],    # status
            [0, 72],   # month
            [0, 5],    # credit_history
            [0, 10],   # purpose
            [0, 20000], # credit_amount
            [0, 5],    # savings
            [0, 5],    # employment
            [0, 4],    # investment_as_income_percentage
            [0, 3],    # other_debtors
            [0, 4],    # residence_since
            [0, 4],    # property
            [0, 1],    # age (protected attribute, binary)
            [0, 3],    # installment_plans
            [0, 3],    # housing
            [0, 4],    # number_of_credits
            [0, 4],    # skill_level
            [0, 2],    # people_liable_for
            [0, 2],    # telephone
            [0, 2]     # foreign_worker
        ])
        protected_idx = 11  # age is at index 11 when personal_status is dropped
    
    # === Create BinaryLabelDataset ===
    def to_aif360_dataset(X, y):
        df_data = pd.DataFrame(X, columns=feature_names)
        df_data["label"] = y
        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df_data,
            label_names=["label"],
            protected_attribute_names=["age"],
            privileged_protected_attributes=[[1]],
            unprivileged_protected_attributes=[[0]]
        )
        return dataset
   
    dataset_train = to_aif360_dataset(X_train, y_train)
    dataset_test = to_aif360_dataset(X_test, y_test)
   
    unprivileged_groups = [{"age": 0}]
    privileged_groups = [{"age": 1}]
    
    results = {}
    causal_results = {}
    random_unfairness_results = {}
   
    # ============================================================
    # LOAD ORIGINAL MODEL GC-X.h5
    # ============================================================
    print("\n" + "=" * 60)
    print("LOADING ORIGINAL MODEL")
    print("=" * 60)
    
    original_model = temp_model  # Use the model we already loaded
    input_model_path = f'Fairify/models/german/{ORIGINAL_MODEL_NAME}.h5'
    print(f"Loaded from: {input_model_path}")
    
    y_test_pred_orig = (original_model.predict(X_test) > 0.5).astype(int).flatten()
    
    from sklearn.metrics import accuracy_score
    acc_orig = accuracy_score(y_test, y_test_pred_orig)
    print(f"Original model accuracy: {acc_orig:.4f}")
    
    wrapper_orig = PredictionWrapper(y_test_pred_orig)
    result_orig = measure_fairness_aif360(
        wrapper_orig, X_test, y_test, feature_names, protected_attribute="age"
    )
    print(f"Original fairness metrics: {result_orig}")
    results["OriginalModel"] = result_orig
   
    # ============================================================
    # REPAIR MODEL WITH ADVERSARIAL DEBIASING
    # ============================================================
    print("\n" + "=" * 60)
    print("REPAIRING MODEL WITH ADVERSARIAL DEBIASING")
    print("(Transferring weights from original model)")
    print("=" * 60)
   
    adv_deb = AdversarialDebiasingWithTransfer(
        original_model=original_model,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        scope_name='adv_deb_transfer',
        num_epochs=50,
        batch_size=128
    )
   
    adv_deb.fit(dataset_train)
    dataset_adv_test = adv_deb.predict(dataset_test)
   
    df_adv, _ = dataset_adv_test.convert_to_dataframe()
    X_adv_test = df_adv.drop(columns=["label"]).values
    y_adv_pred = df_adv["label"].values
   
    # ============================================================
    # SAVE REPAIRED MODEL
    # ============================================================
    output_model_path = f"Fairify/models/german/{ORIGINAL_MODEL_NAME}-AdvDb.h5"
    print(f"\n{'='*60}")
    print(f"SAVING REPAIRED MODEL")
    print(f"{'='*60}")
    
    repaired_keras_model = adv_deb.save_as_keras(output_model_path)
    print(f"✓ SAVED REPAIRED MODEL TO: {output_model_path}")
    
    # Verify the saved model
    y_verify = (repaired_keras_model.predict(X_test) > 0.5).astype(int).flatten()
    acc_verify = accuracy_score(y_test, y_verify)
    print(f"\nVerification - Saved model accuracy: {acc_verify:.4f}")
    
    acc_repaired = accuracy_score(y_test, y_adv_pred)
    print(f"Repaired model accuracy: {acc_repaired:.4f} (change: {acc_repaired - acc_orig:+.4f})")
   
    print("\n=== Evaluating Adversarial Debiasing ===")
    wrapper_adv = PredictionWrapper(y_adv_pred)
    result_adv = measure_fairness_aif360(
        wrapper_adv, X_adv_test, y_test, feature_names, protected_attribute="age"
    )
    results["AdversarialDebiasing"] = result_adv
    
    # ============================================================
    # CAUSAL DISCRIMINATION TESTING
    # ============================================================
    print("\n" + "=" * 60)
    print("CAUSAL DISCRIMINATION ANALYSIS")
    print("=" * 60)
    
    if "AdversarialDebiasing" in results:
        print("\nTesting Adversarial Debiasing for causal discrimination on 'age'...")
        wrapper_adv_causal = AIF360CausalWrapper(adv_deb, feature_names)
        detector_adv = CausalDiscriminationDetector(
            wrapper_adv_causal, 
            max_samples=1000, 
            min_samples=100,
            random_seed=42
        )
        
        for fname in feature_names:
            unique_vals = sorted(set(df[fname]))
            detector_adv.add_feature(fname, unique_vals)
        
        _, rate_adv, pairs_adv = detector_adv.causal_discrimination(['age'])
        causal_results["AdversarialDebiasing"] = {
            'rate': rate_adv,
            'pairs': pairs_adv
        }
        print(f"Discrimination rate: {rate_adv:.4f} ({len(pairs_adv)} discriminatory pairs)")
    
    # ============================================================
    # RANDOM UNFAIRNESS TESTING
    # ============================================================
    print("\n" + "=" * 60)
    print("RANDOM UNFAIRNESS (INDIVIDUAL DISCRIMINATION) ANALYSIS")
    print("=" * 60)
    
    if "AdversarialDebiasing" in results:
        print("\nTesting Adversarial Debiasing for individual discrimination...")
        wrapper_adv_random = AIF360NumpyWrapper(adv_deb, feature_names)
        evaluator_adv = FairnessEvaluator(
            wrapper_adv_random,
            constraint,
            protected_attribs=[protected_idx],
            num_attribs=len(feature_names)
        )
        
        avg_discrimination, interval = evaluator_adv.evaluate_individual_fairness(
            sample_round=10,
            num_gen=100
        )
        
        random_unfairness_results["AdversarialDebiasing"] = {
            'avg_discrimination': avg_discrimination,
            'confidence_interval': interval
        }
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"\nInput model:  {input_model_path}")
    print(f"Output model: {output_model_path}")
    print(f"\nAccuracy: {acc_orig:.4f} → {acc_repaired:.4f} (Δ = {acc_repaired - acc_orig:+.4f})")
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Fairness metrics: {result}")
        if model_name in causal_results:
            print(f"  Causal discrimination rate: {causal_results[model_name]['rate']:.4f}")
        if model_name in random_unfairness_results:
            ru = random_unfairness_results[model_name]
            print(f"  Individual discrimination: {ru['avg_discrimination']:.4f} ± {ru['confidence_interval']:.4f}")
    
    adv_deb.sess.close()
    tf.reset_default_graph()
    print("\n✓ REPAIR COMPLETE")