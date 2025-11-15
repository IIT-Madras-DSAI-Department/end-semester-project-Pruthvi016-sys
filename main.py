import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import your optimized models
from algorithms import PCAWeightedKNN, MultiClassSVM, FastMultiClassXGBoost

class MajorityVotingEnsemble:
    def __init__(self, models, tie_breaker='knn'):
        self.models = models
        self.tie_breaker = tie_breaker  # Which model to trust in case of ties
    
    def fit(self, X, y):
        # Train all models
        print("Training ensemble models...")
        
        # Train PCA-KNN
        print("\n1. Training PCA-Weighted KNN...")
        start_time = time.time()
        self.models['knn'].fit(X, y)
        knn_time = time.time() - start_time
        print(f"   PCA-KNN training completed in {knn_time:.2f}s")
        
        # Train SVM
        print("\n2. Training SVM...")
        start_time = time.time()
        self.models['svm'].fit(X, y)
        svm_time = time.time() - start_time
        print(f"   SVM training completed in {svm_time:.2f}s")
        
        # Train XGBoost
        print("\n3. Training XGBoost...")
        start_time = time.time()
        self.models['xgb'].fit(X, y)
        xgb_time = time.time() - start_time
        print(f"   XGBoost training completed in {xgb_time:.2f}s")
        
        total_training_time = knn_time + svm_time + xgb_time
        print(f"\nTotal ensemble training time: {total_training_time:.2f}s")
        
        return self
    
    def predict(self, X):
        # Get predictions from all models
        knn_pred = self.models['knn'].predict(X)
        svm_pred = self.models['svm'].predict(X)
        xgb_pred = self.models['xgb'].predict(X)
        
        # Stack predictions for majority voting
        all_predictions = np.column_stack([knn_pred, svm_pred, xgb_pred])
        
        final_predictions = []
        
        for i in range(len(X)):
            # Get votes for this sample
            votes = all_predictions[i]
            
            # Count occurrences of each class
            vote_counts = Counter(votes)
            
            # Find the maximum count
            max_count = max(vote_counts.values())
            
            # Get all classes with maximum count
            most_common_classes = [cls for cls, count in vote_counts.items() if count == max_count]
            
            if len(most_common_classes) == 1:
                # Clear majority - use that class
                final_pred = most_common_classes[0]
            else:
                # Tie - use tie-breaker model
                if self.tie_breaker == 'knn':
                    final_pred = knn_pred[i]
                elif self.tie_breaker == 'svm':
                    final_pred = svm_pred[i]
                elif self.tie_breaker == 'xgb':
                    final_pred = xgb_pred[i]
                else:
                    # Default: use the first class in case of unexpected tie-breaker
                    final_pred = most_common_classes[0]
            
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)

# Load data
print("Loading data...")
train_data = pd.read_csv('MNIST_train.csv')
val_data = pd.read_csv('MNIST_validation.csv')

# Separate features and labels
X_train = train_data.iloc[:, 1:785].values / 255.0
y_train = train_data.iloc[:, 0].values

X_val = val_data.iloc[:, 1:785].values / 255.0
y_val = val_data.iloc[:, 0].values

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

print("\n" + "="*60)
print("MAJORITY VOTING ENSEMBLE")
print("="*60)

# Initialize models
models = {
    'knn': PCAWeightedKNN(k=5, n_components=30, batch_size=200),
    'svm': MultiClassSVM(learning_rate=0.0003, lambda_param=0.008, n_iters=120, n_classes=10),
    'xgb': FastMultiClassXGBoost(
        n_estimators=70,
        learning_rate=0.15,
        max_depth=9,
        reg_lambda=0.8,
        subsample=0.7,
        colsample=0.4,
        max_bins=32,
        n_classes=10
    )
}

# Create majority voting ensemble (tie goes to KNN since it's your best model)
ensemble = MajorityVotingEnsemble(models, tie_breaker='knn')

# Train ensemble
print("\nStarting ensemble training...")
ensemble_start = time.time()
ensemble.fit(X_train, y_train)
ensemble_training_time = time.time() - ensemble_start

print(f"\n" + "="*60)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*60)

# Evaluate individual models
individual_results = {}
for name, model in models.items():
    print(f"\nEvaluating {name.upper()}...")
    start_time = time.time()
    y_pred = model.predict(X_val)
    pred_time = time.time() - start_time
    
    f1 = f1_score(y_val, y_pred, average='weighted')
    accuracy = np.mean(y_pred == y_val)
    
    print(f"{name.upper()} Results:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    individual_results[name] = {
        'f1': f1,
        'accuracy': accuracy,
        'predictions': y_pred
    }

print(f"\n" + "="*60)
print("MAJORITY VOTING ENSEMBLE PERFORMANCE")
print("="*60)

# Evaluate ensemble
ensemble_start_pred = time.time()
y_pred_ensemble = ensemble.predict(X_val)
ensemble_pred_time = time.time() - ensemble_start_pred

f1_ensemble = f1_score(y_val, y_pred_ensemble, average='weighted')
accuracy_ensemble = np.mean(y_pred_ensemble == y_val)

print(f"Ensemble Results:")
print(f"  F1 Score: {f1_ensemble:.4f}")
print(f"  Accuracy: {accuracy_ensemble:.4f}")
print(f"  Total Training Time: {ensemble_training_time:.2f}s")


print(f"\n🎯 FINAL MAJORITY VOTING ENSEMBLE F1: {f1_ensemble:.4f}")