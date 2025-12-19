# Test AIF360 Fair Classifiers - Default Credit Dataset
if __name__ == "__main__":
    import sys
    import os
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import KBinsDiscretizer
    import logging
    import warnings
    
    ORIGINAL_MODEL_NAME = "DF-5"
   
    # === Load paths and project structure ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    sys.path.append(src_dir)
    from utils.verif_utils import *
    from metric_aif360 import measure_fairness_aif360
    from metric_themis_causality import CausalDiscriminationDetector, set_all_seeds
    from metric_random_unfairness import FairnessEvaluator, set_all_seeds
    from aif360.datasets import BinaryLabelDataset
    import tensorflow.compat.v1 as tf
    import tensorflow.keras as keras
   
    set_all_seeds(42)
   
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
                protected_attribute_names=["SEX"],  # Changed to ordinal SEX
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
                protected_attribute_names=["SEX"],  # Changed to ordinal SEX
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
                    # Build network layer by layer to match original model
                    x = self.input_ph
                    
                    # Process all layers except the last one (which is output layer)
                    for layer_idx in range(len(self.original_model.layers) - 1):
                        layer = self.original_model.layers[layer_idx]
                        W, b = orig_weights[layer_idx * 2], orig_weights[layer_idx * 2 + 1]
                        
                        W_var = tf.get_variable(f'W{layer_idx}', initializer=tf.constant(W, dtype=tf.float32))
                        b_var = tf.get_variable(f'b{layer_idx}', initializer=tf.constant(b, dtype=tf.float32))
                        
                        x = tf.matmul(x, W_var) + b_var
                        x = tf.nn.relu(x)  # All hidden layers use ReLU
                    
                    # Output layer (no activation, we'll apply sigmoid later)
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
                
                # Adversary network (tries to predict protected attribute from predictions)
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
                
                # Combined loss: classifier tries to minimize clf_loss while MAXIMIZING adv_loss
                # This makes predictions independent of protected attribute
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
            
            # Create new dataset with predictions
            pred_dataset = dataset.copy()
            pred_dataset.labels = predictions.reshape(-1, 1)
            
            return pred_dataset
        
        def save_as_keras(self, output_path):
            """Extract weights and save as Keras model with exact original architecture."""
            # Get classifier variables - EXCLUDE Adam optimizer variables
            clf_vars = [v for v in tf.global_variables() 
                        if f'{self.scope_name}/classifier' in v.name 
                        and 'Adam' not in v.name]
            
            # Print debug info
            print("\nDebug: TensorFlow variables found:")
            for v in sorted(clf_vars, key=lambda x: x.name):
                print(f"  {v.name}: {v.shape}")
            
            # Separate weights and biases, sort by layer index extracted from name
            weight_vars = sorted([v for v in clf_vars if '/W' in v.name], 
                                key=lambda v: int(v.name.split('/W')[1].split(':')[0]))
            bias_vars = sorted([v for v in clf_vars if '/b' in v.name], 
                            key=lambda v: int(v.name.split('/b')[1].split(':')[0]))
            
            print("\nDebug: Sorted weights:")
            for v in weight_vars:
                print(f"  {v.name}: {v.shape}")
            print("\nDebug: Sorted biases:")
            for v in bias_vars:
                print(f"  {v.name}: {v.shape}")
            
            # Run session to get values
            weight_values = self.sess.run(weight_vars)
            bias_values = self.sess.run(bias_vars)
            
            print("\nDebug: Weight shapes after extraction:")
            for i, w in enumerate(weight_values):
                print(f"  Layer {i} weight: {w.shape}")
            print("\nDebug: Bias shapes after extraction:")
            for i, b in enumerate(bias_values):
                print(f"  Layer {i} bias: {b.shape}")
            
            # Recreate the exact original architecture
            n_features = weight_values[0].shape[0] if len(weight_values) > 0 else len(self.original_model.layers[0].input_shape) - 1
            
            # Build layers dynamically to match original model
            layers = []
            num_layers = len(self.original_model.layers)
            
            print(f"\nDebug: Building Keras model with {num_layers} layers:")
            for layer_idx in range(num_layers):
                units = self.original_model.layers[layer_idx].units
                print(f"  Layer {layer_idx}: {units} units")
                
                if layer_idx == 0:
                    # First layer needs input_shape
                    if layer_idx == num_layers - 1:
                        # If only one layer (output), use sigmoid
                        layers.append(keras.layers.Dense(units, activation='sigmoid', input_shape=(n_features,)))
                    else:
                        layers.append(keras.layers.Dense(units, activation='relu', input_shape=(n_features,)))
                else:
                    # Subsequent layers
                    if layer_idx == num_layers - 1:
                        # Output layer uses sigmoid
                        layers.append(keras.layers.Dense(units, activation='sigmoid'))
                    else:
                        # Hidden layers use relu
                        layers.append(keras.layers.Dense(units, activation='relu'))
            
            keras_model = keras.Sequential(layers)
            keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Transfer all weights (now properly matched)
            print("\nDebug: Setting weights...")
            for layer_idx in range(num_layers):
                if layer_idx < len(weight_values) and layer_idx < len(bias_values):
                    print(f"  Layer {layer_idx}: setting weight {weight_values[layer_idx].shape} and bias {bias_values[layer_idx].shape}")
                    keras_model.layers[layer_idx].set_weights([weight_values[layer_idx], bias_values[layer_idx]])
            
            keras_model.save(output_path)
            return keras_model
   
    # === Load Default Credit Dataset with NEW preprocessing ===
    def load_default(n_bins=5):
        """
        Load and preprocess UCI Credit Card dataset with discretization for easier CE generation.
        
        Key changes from original:
        1. Bins continuous features into discrete categories (reduces partition complexity)
        2. Uses ordinal encoding instead of one-hot (fewer dimensions)
        3. Simplifies payment status into 3 levels (on-time, slight delay, severe delay)
        """
        file_path = 'Fairify/data/default/default.csv'
        print("Loading default credit dataset...")
        df = pd.read_csv(file_path)
        df = df.rename(columns={"PAY_0": "PAY_1"})
        df = df.drop(columns=["ID"])
        
        # Bin continuous features to make CE generation easier
        continuous_features = [
            'LIMIT_BAL', 'AGE', 
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]
        
        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        df[continuous_features] = binner.fit_transform(df[continuous_features]).astype(int)
        
        # Simplify payment status: 0=on-time, 1=slight delay (1-2mo), 2=severe delay (3+mo)
        payment_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        for col in payment_cols:
            df[col] = pd.cut(df[col], bins=[-np.inf, 0, 2, np.inf], labels=[0, 1, 2]).astype(int)
        
        # Use ordinal encoding (not one-hot) - keeps dimensions lower
        df['SEX'] = df['SEX'] - 1
        df['EDUCATION'] = np.clip(df['EDUCATION'] - 1, 0, 3)
        df['MARRIAGE'] = np.clip(df['MARRIAGE'] - 1, 0, 2)
        
        label_name = "default.payment.next.month"
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
    
    print("Loading Default Credit dataset...")
    df, X_train, y_train, X_test, y_test = load_default()
    
    # Feature names after NEW preprocessing (ordinal encoding, no one-hot)
    feature_names = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Actual X_train shape: {X_train.shape}")
    
    # Define constraints for Default Credit dataset
    constraint = []
    for col in feature_names:
        min_val = df[col].min()
        max_val = df[col].max()
        constraint.append([min_val, max_val])
    constraint = np.array(constraint)
    
    # Find the index of the protected attribute (SEX - ordinal encoding)
    protected_attr_idx = feature_names.index('SEX') if 'SEX' in feature_names else None
    print(f"Protected attribute (SEX) index: {protected_attr_idx}")
    
    # === Create BinaryLabelDataset ===
    def to_aif360_dataset(X, y):
        df_data = pd.DataFrame(X, columns=feature_names)
        df_data["label"] = y
        dataset = BinaryLabelDataset(
            favorable_label=0,  # 0 = no default (favorable outcome)
            unfavorable_label=1,  # 1 = default
            df=df_data,
            label_names=["label"],
            protected_attribute_names=["SEX"],  # Changed to ordinal SEX
            privileged_protected_attributes=[[1]],  # Female (SEX=1)
            unprivileged_protected_attributes=[[0]]  # Male (SEX=0)
        )
        return dataset
   
    dataset_train = to_aif360_dataset(X_train, y_train)
    dataset_test = to_aif360_dataset(X_test, y_test)
   
    unprivileged_groups = [{"SEX": 0}]  # Changed to ordinal SEX
    privileged_groups = [{"SEX": 1}]
    
    results = {}
    causal_results = {}
    random_unfairness_results = {}
    
    tf.disable_eager_execution()
   
    # ============================================================
    # LOAD ORIGINAL MODEL DF-1.h5
    # ============================================================
    print("\n" + "=" * 60)
    print("LOADING ORIGINAL MODEL")
    print("=" * 60)
    
    original_model = load_model(f'Fairify/models/default/{ORIGINAL_MODEL_NAME}.h5')
    input_model_path = f'Fairify/models/default/{ORIGINAL_MODEL_NAME}.h5'
    print(f"Loaded from: {input_model_path}")
    
    y_test_pred_orig = (original_model.predict(X_test) > 0.5).astype(int).flatten()
    
    from sklearn.metrics import accuracy_score
    acc_orig = accuracy_score(y_test, y_test_pred_orig)
    print(f"Original model accuracy: {acc_orig:.4f}")
    
    wrapper_orig = PredictionWrapper(y_test_pred_orig)
    result_orig = measure_fairness_aif360(
        wrapper_orig, X_test, y_test, feature_names, protected_attribute="SEX"
    )
    print(f"Original fairness metrics: {result_orig}")
    results["OriginalModel"] = result_orig
   
    # ============================================================
    # REPAIR MODEL WITH ADVERSARIAL DEBIASING (TRANSFER WEIGHTS)
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
    # SAVE REPAIRED MODEL AS DF-1-AdvDb.h5
    # ============================================================
    output_model_path = f"Fairify/models/default/{ORIGINAL_MODEL_NAME}-AdvDb.h5"
    print(f"\n{'='*60}")
    print(f"SAVING REPAIRED MODEL")
    print(f"{'='*60}")
    
    repaired_keras_model = adv_deb.save_as_keras(output_model_path)
    print(f"✓ SAVED REPAIRED MODEL TO: {output_model_path}")
    
    # Verify the saved model works
    y_verify = (repaired_keras_model.predict(X_test) > 0.5).astype(int).flatten()
    acc_verify = accuracy_score(y_test, y_verify)
    print(f"\nVerification - Saved model accuracy: {acc_verify:.4f}")
    
    acc_repaired = accuracy_score(y_test, y_adv_pred)
    print(f"Repaired model accuracy: {acc_repaired:.4f} (change: {acc_repaired - acc_orig:+.4f})")
   
    print("\n=== Evaluating Adversarial Debiasing ===")
    wrapper_adv = PredictionWrapper(y_adv_pred)
    result_adv = measure_fairness_aif360(
        wrapper_adv, X_adv_test, y_test, feature_names, protected_attribute="SEX"
    )
    results["AdversarialDebiasing"] = result_adv
    
    # ============================================================
    # CAUSAL DISCRIMINATION TESTING
    # ============================================================
    print("\n" + "=" * 60)
    print("CAUSAL DISCRIMINATION ANALYSIS")
    print("=" * 60)
    
    if "AdversarialDebiasing" in results:
        print("\nTesting Adversarial Debiasing for causal discrimination on 'SEX'...")
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
        
        _, rate_adv, pairs_adv = detector_adv.causal_discrimination(['SEX'])
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
    
    if "AdversarialDebiasing" in results and protected_attr_idx is not None:
        print("\nTesting Adversarial Debiasing for individual discrimination...")
        wrapper_adv_random = AIF360NumpyWrapper(adv_deb, feature_names)
        evaluator_adv = FairnessEvaluator(
            wrapper_adv_random,
            constraint,
            protected_attribs=[protected_attr_idx],  # Use actual index of SEX
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