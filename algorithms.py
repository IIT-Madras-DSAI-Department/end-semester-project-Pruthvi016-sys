#XGBoost:
import numpy as np
import time
from sklearn.metrics import f1_score

class FastXGBoostTree:
    def __init__(self, max_depth=3, reg_lambda=1.0, gamma=0.0, max_bins=50):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_bins = max_bins
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.leaf_value = None

    def _calc_leaf_value(self, G, H):
        return -G / (H + self.reg_lambda)

    def _calc_gain(self, G_left, H_left, G_right, H_right, G_total, H_total):
        gain = 0.5 * (G_left*G_left/(H_left + self.reg_lambda) + 
                     G_right*G_right/(H_right + self.reg_lambda) - 
                     G_total*G_total/(H_total + self.reg_lambda)) - self.gamma
        return gain

    def _find_best_split(self, X, g, h, sample_indices, feature_thresholds):
        G_total = np.sum(g[sample_indices])
        H_total = np.sum(h[sample_indices])
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        # Only check random subset of features (like Random Forest)
        n_features = len(feature_thresholds)
        n_features_to_check = int(np.sqrt(n_features))  # Random Forest style
        
        feature_indices = np.random.choice(n_features, n_features_to_check, replace=False)
        
        for feat_idx in feature_indices:
            thresholds = feature_thresholds[feat_idx]
            if thresholds is None or len(thresholds) == 0:
                continue

            x_col = X[sample_indices, feat_idx]
            g_vals = g[sample_indices]
            h_vals = h[sample_indices]

            # Sort only once per feature
            sorted_idx = np.argsort(x_col)
            x_sorted = x_col[sorted_idx]
            g_sorted = g_vals[sorted_idx]
            h_sorted = h_vals[sorted_idx]

            G_left, H_left = 0.0, 0.0
            
            # Try fewer thresholds for speed
            step = max(1, len(thresholds) // 10)  # Check only 10 thresholds per feature
            for th in thresholds[::step]:
                # Find split position efficiently
                split_pos = np.searchsorted(x_sorted, th, side='right')
                if split_pos == 0 or split_pos == len(x_sorted):
                    continue

                # Update cumulative sums incrementally
                if split_pos > len(g_sorted) // 2:
                    # Calculate from the right side for better numerical stability
                    G_right = np.sum(g_sorted[split_pos:])
                    H_right = np.sum(h_sorted[split_pos:])
                    G_left = G_total - G_right
                    H_left = H_total - H_right
                else:
                    G_left = np.sum(g_sorted[:split_pos])
                    H_left = np.sum(h_sorted[:split_pos])
                    G_right = G_total - G_left
                    H_right = H_total - H_left

                gain = self._calc_gain(G_left, H_left, G_right, H_right, G_total, H_total)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = th

        return best_feature, best_threshold, best_gain

    def _fit_recursive(self, X, g, h, sample_indices, depth, feature_thresholds):
        n_samples = len(sample_indices)
        if n_samples == 0:
            return

        G = np.sum(g[sample_indices])
        H = np.sum(h[sample_indices])
        self.leaf_value = self._calc_leaf_value(G, H)

        # Stop early if no improvement
        if depth >= self.max_depth or n_samples < 20:
            return

        feature, threshold, gain = self._find_best_split(X, g, h, sample_indices, feature_thresholds)

        if feature is None or gain <= 0:
            return

        self.feature = feature
        self.threshold = threshold

        left_mask = X[sample_indices, feature] <= threshold
        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[~left_mask]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return

        # Grow left and right branches
        self.left = FastXGBoostTree(
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            max_bins=self.max_bins
        )
        self.left._fit_recursive(X, g, h, left_indices, depth + 1, feature_thresholds)

        self.right = FastXGBoostTree(
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            max_bins=self.max_bins
        )
        self.right._fit_recursive(X, g, h, right_indices, depth + 1, feature_thresholds)

    def fit(self, X, g, h, feature_thresholds):
        sample_indices = np.arange(X.shape[0])
        self._fit_recursive(X, g, h, sample_indices, 0, feature_thresholds)
        return self

    def predict(self, X):
        if self.feature is None:
            return np.full(X.shape[0], self.leaf_value)
        
        left_mask = X[:, self.feature] <= self.threshold
        preds = np.full(X.shape[0], self.leaf_value)
        
        if self.left:
            preds[left_mask] = self.left.predict(X[left_mask])
        if self.right:
            preds[~left_mask] = self.right.predict(X[~left_mask])
            
        return preds


class FastXGBoostClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0, max_bins=50, subsample=0.8, colsample=0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample  # Column subsampling
        self.trees = []
        self.base_score = 0.0

    def _sigmoid(self, x):
        # Faster sigmoid approximation
        x = np.clip(x, -20, 20)  # Tighter bounds for stability
        return 0.5 * (x / (1 + np.abs(x))) + 0.5

    def _precompute_feature_thresholds(self, X):
        thresholds = []
        n_samples = X.shape[0]
        
        for feat in range(X.shape[1]):
            col = X[:, feat]
            if np.all(col == col[0]):
                thresholds.append(None)
                continue
                
            # Use fewer bins for speed
            n_bins = min(self.max_bins, n_samples // 10)
            if len(np.unique(col)) > n_bins:
                percentiles = np.linspace(10, 90, n_bins - 1)  # Skip extremes
                ths = np.percentile(col, percentiles)
            else:
                unique_vals = np.unique(col)
                ths = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                
            thresholds.append(ths if len(ths) > 0 else None)
        return thresholds

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)  # Use float32 for speed
        y = np.asarray(y, dtype=np.float32)
        y_binary = y

        # Precompute global thresholds once
        feature_thresholds = self._precompute_feature_thresholds(X)

        n_samples, n_features = X.shape
        y_pred = np.full(n_samples, self.base_score, dtype=np.float32)

        for i in range(self.n_estimators):
            # Row subsampling
            if self.subsample < 1.0:
                subsample_size = int(self.subsample * n_samples)
                row_indices = np.random.choice(n_samples, subsample_size, replace=False)
            else:
                row_indices = np.arange(n_samples)

            # Column subsampling for extra speed and regularization
            if self.colsample < 1.0:
                n_cols = int(self.colsample * n_features)
                col_indices = np.random.choice(n_features, n_cols, replace=False)
                X_sub = X[row_indices][:, col_indices]
                feature_thresholds_sub = [feature_thresholds[i] for i in col_indices]
            else:
                col_indices = np.arange(n_features)
                X_sub = X[row_indices]
                feature_thresholds_sub = feature_thresholds

            y_sub = y_binary[row_indices]
            y_pred_sub = y_pred[row_indices]

            # Calculate gradients and hessians
            p = self._sigmoid(y_pred_sub)
            g = p - y_sub
            h = p * (1.0 - p) + 1e-6  # Smaller epsilon

            # Train tree
            tree = FastXGBoostTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                max_bins=self.max_bins
            )
            tree.fit(X_sub, g, h, feature_thresholds_sub)
            
            # Store column indices for prediction
            tree.col_indices = col_indices
            self.trees.append(tree)

            # Update predictions
            if self.colsample < 1.0:
                X_pred = X[:, col_indices]
            else:
                X_pred = X
                
            update = tree.predict(X_pred)
            y_pred += self.learning_rate * update

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        y_pred = np.full(X.shape[0], self.base_score, dtype=np.float32)
        
        for tree in self.trees:
            if hasattr(tree, 'col_indices'):
                X_sub = X[:, tree.col_indices]
            else:
                X_sub = X
            y_pred += self.learning_rate * tree.predict(X_sub)
            
        proba = self._sigmoid(y_pred)
        return (proba >= 0.5).astype(int)


class FastMultiClassXGBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, reg_lambda=1.0, 
                 gamma=0.0, max_bins=50, n_classes=10, subsample=0.8, colsample=0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_bins = max_bins
        self.n_classes = n_classes
        self.subsample = subsample
        self.colsample = colsample
        self.classifiers = []

    def fit(self, X, y):
        self.classifiers = []
        
        for class_idx in range(self.n_classes):
            print(f"Training XGBoost for class {class_idx+1}/{self.n_classes}...")
            
            y_binary = (y == class_idx).astype(int)
            
            xgb_binary = FastXGBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                max_bins=self.max_bins,
                subsample=self.subsample,
                colsample=self.colsample
            )
            
            xgb_binary.fit(X, y_binary)
            self.classifiers.append(xgb_binary)
            
        return self

    def predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], self.n_classes), dtype=np.float32)
        
        for class_idx, classifier in enumerate(self.classifiers):
            raw_scores = np.zeros(X.shape[0], dtype=np.float32)
            for tree in classifier.trees:
                if hasattr(tree, 'col_indices'):
                    X_sub = X[:, tree.col_indices]
                else:
                    X_sub = X
                raw_scores += classifier.learning_rate * tree.predict(X_sub)
            
            # Use fast sigmoid
            proba = 0.5 * (raw_scores / (1 + np.abs(raw_scores))) + 0.5
            probabilities[:, class_idx] = proba
            
        # Normalize
        prob_sums = np.sum(probabilities, axis=1, keepdims=True)
        probabilities /= np.maximum(prob_sums, 1e-8)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

#PCA-WEIGHTED-KNN
import numpy as np
import time
from collections import Counter
from sklearn.metrics import f1_score, classification_report

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]
        
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

class PCAWeightedKNN:
    def __init__(self, k=5, n_components=50, batch_size=100):
        self.k = k
        self.n_components = n_components
        self.batch_size = batch_size
        self.pca = PCA(n_components)
        self.X_train_pca = None
        self.y_train = None
        
    def fit(self, X, y):
        print(f"Applying PCA to reduce dimensions from {X.shape[1]} to {self.n_components}...")
        start_pca = time.time()
        self.pca.fit(X)
        self.X_train_pca = self.pca.transform(X.astype(np.float32))
        self.y_train = y.astype(np.int32)
        pca_time = time.time() - start_pca
        print(f"PCA completed in {pca_time:.2f}s")
        
    def predict(self, X):
        X_pca = self.pca.transform(X.astype(np.float32))
        predictions = np.empty(X_pca.shape[0], dtype=np.int32)
        
        # Precompute squared norms for training data in PCA space
        train_norms = np.sum(self.X_train_pca ** 2, axis=1)
        
        # Process in batches
        for batch_start in range(0, X_pca.shape[0], self.batch_size):
            batch_end = min(batch_start + self.batch_size, X_pca.shape[0])
            X_batch = X_pca[batch_start:batch_end]
            
            # Vectorized distance computation in PCA space
            test_norms = np.sum(X_batch ** 2, axis=1)
            dot_products = np.dot(self.X_train_pca, X_batch.T)
            squared_distances = train_norms[:, np.newaxis] + test_norms - 2 * dot_products
            
            # Get k nearest neighbors for each batch sample
            nearest_indices = np.argpartition(squared_distances, self.k, axis=0)[:self.k]
            nearest_squared_distances = np.take_along_axis(squared_distances, nearest_indices, axis=0)
            nearest_labels = self.y_train[nearest_indices]
            
            # Convert to actual distances and compute weights
            distances = np.sqrt(nearest_squared_distances + 1e-8)
            weights = 1.0 / (distances + 1e-8)
            
            # Weighted majority voting for each sample in batch
            for j in range(batch_end - batch_start):
                label_weights = {}
                for idx in range(self.k):
                    label = nearest_labels[idx, j]
                    weight = weights[idx, j]
                    label_weights[label] = label_weights.get(label, 0.0) + weight
                
                predictions[batch_start + j] = max(label_weights.items(), key=lambda x: x[1])[0]
                
        return predictions
#SVM:
class MultiClassSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, n_classes=10):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.W = None  # Weight matrix for all classes
        self.b = None  # Bias vector for all classes

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros((self.n_classes, n_features))
        self.b = np.zeros(self.n_classes)

        # One-vs-Rest approach
        for class_idx in range(self.n_classes):
            print(f"Training SVM for class {class_idx}...")
            # Convert to binary labels: current class vs all others
            y_binary = np.where(y == class_idx, 1, -1)
            
            w = np.zeros(n_features)
            b = 0

            for _ in range(self.n_iters):
                for i, x_i in enumerate(X):
                    condition = y_binary[i] * (np.dot(x_i, w) - b) >= 1
                    if condition:
                        w -= self.lr * (self.lambda_param * w)
                    else:
                        w -= self.lr * (self.lambda_param * w - np.dot(x_i, y_binary[i]))
                        b -= self.lr * y_binary[i]

            self.W[class_idx] = w
            self.b[class_idx] = b

    def predict(self, X):
        # For each sample, find the class with highest score
        scores = np.dot(X, self.W.T) - self.b
        return np.argmax(scores, axis=1)