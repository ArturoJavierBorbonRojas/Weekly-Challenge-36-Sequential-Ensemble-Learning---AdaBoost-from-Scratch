import numpy as np

# Weekly Challenge 36: Sequential Ensemble Learning - AdaBoost from Scratch
# Author: Ing. Arturo Javier Borbon Rojas

print("🚀 Initializing AdaBoost (Adaptive Boosting) Engine...")
print("-" * 65)

# 1. Weak Learner: Decision Stump (A tree with only 1 split)
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None # The 'say' or weight this stump has in the final vote

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        
        # Predict -1 or 1 based on polarity and threshold
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

# 2. AdaBoost Classifier
class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights: All students start with the exact same weight (1/N)
        w = np.full(n_samples, (1 / n_samples))
        
        print(f"⚙️ Training {self.n_clf} sequential Decision Stumps:")
        
        for i in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            
            # Find the best threshold and feature to split on
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                
                for threshold in thresholds:
                    # Test both polarities (direction of the inequality)
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    
                    # Calculate weighted error
                    misclassified = w[y != predictions]
                    error = sum(misclassified)
                    
                    # Flip polarity if error is greater than 50%
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                        
                    # Save the best stump configuration
                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i

            # Calculate the stump's Alpha (amount of say in the final vote)
            EPS = 1e-10 # Prevent division by zero
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            
            # Generate predictions with the best stump
            predictions = clf.predict(X)
            
            # Update sample weights: Increase weight of misclassified students
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w) # Normalize weights to sum to 1
            
            self.clfs.append(clf)
            print(f"   ➔ Stump {i+1} | Feature: {clf.feature_idx} | Threshold: {clf.threshold:.2f} | Alpha (Say): {clf.alpha:.3f}")

    def predict(self, X):
        """Aggregate predictions from all stumps weighted by their Alpha."""
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)

# --- Synthetic Educational Dataset ---
# Features: [Grades (0-10), Absences (Total), Stress Level (1-10)]
X_train = np.array([
    [9.5, 2, 3], [8.0, 5, 4], [8.5, 3, 3], [7.5, 4, 5], # Safe students
    [6.0, 15, 8], [5.5, 12, 9], [4.0, 18, 7], [6.5, 14, 8], # Clear At-Risk students
    [8.0, 12, 4], [7.0, 3, 9] # Boundary/Confusing cases for the algorithm
])

# Labels strictly mapped to: -1 (Safe), 1 (At Risk)
y_train = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1])

# Train AdaBoost with 5 sequential stumps
boost = AdaBoost(n_clf=5)
boost.fit(X_train, y_train)

print("-" * 65)
print("📊 EVALUATING SEQUENTIAL ENSEMBLE:")

X_test = np.array([
    [9.0, 1, 2],   # Expected: -1 (Safe)
    [5.0, 16, 9],  # Expected: 1 (Risk)
    [7.5, 10, 8]   # Complex borderline case
])

predictions = boost.predict(X_test)

for i, (student, pred) in enumerate(zip(X_test, predictions)):
    status = "🚨 At Risk" if pred == 1 else "✅ Safe (Retained)"
    print(f"Profile {i+1} [Grades: {student[0]}, Absences: {student[1]:02.0f}, Stress: {student[2]}] ➔ {status}")