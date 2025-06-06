import numpy as np
from collections import Counter
import math

class XGBoostRegressor:
    def __init__(self, 
             n_estimators=100,
             max_depth=6,
             learning_rate=0.1,
             subsample=1.0,
             colsample_bytree=1.0,
             colsample_bylevel=1.0,
             reg_alpha=0.0,
             reg_lambda=1.0,
             gamma=0.0,
             min_child_weight=1.0,
             max_delta_step=0.0,
             random_state=None,
             objective='reg:squarederror',
             base_score=0.5,
             early_stopping_rounds=None,
             eval_metric=None):
    
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha  # L1 regularization
        self.reg_lambda = reg_lambda  # L2 regularization
        self.gamma = gamma  # Minimum loss reduction required to make split
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.random_state = random_state
        self.objective = objective
        self.base_score = base_score
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric if eval_metric else 'rmse'  # Default metric
        
        self.trees = []
        self.feature_importances_ = None
        self.n_features_ = None
        self.best_iteration = None
        
        if random_state is not None:
            np.random.seed(random_state)

    def _compute_eval_metric(self, y_true, y_pred):
        """Compute evaluation metric"""
        if self.eval_metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.eval_metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.eval_metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        else:
            raise ValueError(f"Unsupported eval_metric: {self.eval_metric}")

    def fit(self, X, y, eval_set=None, verbose=False):
        """Fit the XGBoost model with early stopping support"""
        
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Initialize predictions with base score
        y_pred = np.full(n_samples, self.base_score)
        
        # Feature importance tracking
        feature_importances = np.zeros(self.n_features_)
        
        # Early stopping variables
        best_score = np.inf
        best_iteration = 0
        early_stop_counter = 0
        eval_scores = []
        
        # Process evaluation set if provided
        eval_X, eval_y = None, None
        if eval_set is not None:
            eval_X, eval_y = eval_set
            eval_pred = np.full(eval_y.shape[0], self.base_score)
        
        # Build trees iteratively
        for i in range(self.n_estimators):
            # Sample rows (subsample)
            if self.subsample < 1.0:
                sample_indices = np.random.choice(n_samples, 
                                                int(n_samples * self.subsample), 
                                                replace=False)
                X_sampled = X[sample_indices, :]
                y_sampled = y[sample_indices]
                y_pred_sampled = y_pred[sample_indices]
            else:
                sample_indices = np.arange(n_samples)
                X_sampled = X
                y_sampled = y
                y_pred_sampled = y_pred
            
            # Sample features (colsample_bytree)
            if self.colsample_bytree < 1.0:
                n_features_tree = max(1, int(self.n_features_ * self.colsample_bytree))
                feature_indices = np.random.choice(self.n_features_, n_features_tree, replace=False)
            else:
                feature_indices = np.arange(self.n_features_)
            
            # Compute gradients and hessians
            gradients, hessians = self._compute_gradients_hessians(y_sampled, y_pred_sampled)
            
            # Build tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                colsample_bylevel=self.colsample_bylevel
            )
            
            tree.fit(X_sampled, gradients, hessians, feature_indices)
            self.trees.append((tree, feature_indices, sample_indices))
            
            # Update predictions
            tree_predictions = tree.predict(X_sampled)
            y_pred[sample_indices] += self.learning_rate * tree_predictions
            
            # Update feature importances
            for feature_idx, importance in tree.feature_importances.items():
                feature_importances[feature_idx] += importance
            
            # Evaluate on validation set if provided
            if eval_set is not None:
                eval_tree_pred = tree.predict(eval_X[:, feature_indices])
                eval_pred += self.learning_rate * eval_tree_pred
                current_score = self._compute_eval_metric(eval_y, eval_pred)
                eval_scores.append(current_score)
                
                if verbose:
                    print(f"[{i}] Validation score: {current_score:.4f}")
                
                # Check for early stopping
                if self.early_stopping_rounds is not None:
                    if current_score < best_score:
                        best_score = current_score
                        best_iteration = i
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= self.early_stopping_rounds:
                            if verbose:
                                print(f"Early stopping at iteration {i}, best iteration is {best_iteration}")
                            self.best_iteration = best_iteration
                            break
            
        # Normalize feature importances
        if np.sum(feature_importances) > 0:
            self.feature_importances_ = feature_importances / np.sum(feature_importances)
        else:
            self.feature_importances_ = feature_importances
        
        # If no early stopping, set best_iteration to last iteration
        if self.best_iteration is None:
            self.best_iteration = len(self.trees) - 1

    def predict(self, X, ntree_limit=None):
        """Make predictions with optional early stopping"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize with base score
        predictions = np.full(n_samples, self.base_score)
        
        # Determine how many trees to use
        if ntree_limit is None and hasattr(self, 'best_iteration'):
            ntree_limit = self.best_iteration + 1
        elif ntree_limit is None:
            ntree_limit = len(self.trees)
        
        # Add predictions from each tree up to ntree_limit
        for i in range(ntree_limit):
            tree, feature_indices, _ = self.trees[i]
            X_subset = X[:, feature_indices] if len(feature_indices) < self.n_features_ else X
            tree_predictions = tree.predict(X_subset)
            predictions += self.learning_rate * tree_predictions
        
        return predictions
    
    def _calculate_similarity_score(self, gradients, hessians):
        """Calculate similarity score for a node"""
        G = np.sum(gradients)
        H = np.sum(hessians)
        return (G ** 2) / (H + self.reg_lambda)
    
    def _calculate_gain(self, left_gradients, left_hessians, right_gradients, right_hessians, parent_gradients, parent_hessians):
        """Calculate gain from a split"""
        left_score = self._calculate_similarity_score(left_gradients, left_hessians)
        right_score = self._calculate_similarity_score(right_gradients, right_hessians)
        parent_score = self._calculate_similarity_score(parent_gradients, parent_hessians)
        
        gain = 0.5 * (left_score + right_score - parent_score) - self.gamma
        return gain
    
    def _calculate_leaf_weight(self, gradients, hessians):
        """Calculate optimal weight for a leaf node"""
        G = np.sum(gradients)
        H = np.sum(hessians)
        weight = -G / (H + self.reg_lambda)
        
        # Apply max_delta_step constraint if specified
        if self.max_delta_step != 0:
            weight = np.sign(weight) * min(abs(weight), self.max_delta_step)
        
        return weight
    
    def _find_best_split(self, X, gradients, hessians, feature_indices):
        """Find the best split for current node"""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        n_samples = len(X)
        
        for feature_idx in feature_indices:
            # Get unique values for this feature
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try splits between unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_gradients = gradients[left_mask]
                left_hessians = hessians[left_mask]
                right_gradients = gradients[right_mask]
                right_hessians = hessians[right_mask]
                
                # Check minimum child weight constraint
                if (np.sum(left_hessians) < self.min_child_weight or 
                    np.sum(right_hessians) < self.min_child_weight):
                    continue
                
                gain = self._calculate_gain(left_gradients, left_hessians,
                                          right_gradients, right_hessians,
                                          gradients, hessians)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = np.where(left_mask)[0]
                    best_right_indices = np.where(right_mask)[0]
        
        return best_gain, best_feature, best_threshold, best_left_indices, best_right_indices
    # Continue XGBoostRegressor implementation
    def _compute_gradients_hessians(self, y_true, y_pred):
        """Compute gradients and hessians for squared error loss"""
        if self.objective == 'reg:squarederror':
            gradients = 2 * (y_pred - y_true)
            hessians = np.full_like(gradients, 2.0)
        elif self.objective == 'reg:linear':  # Same as squared error
            gradients = 2 * (y_pred - y_true)
            hessians = np.full_like(gradients, 2.0)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")
        
        return gradients, hessians

    def get_feature_importance(self, importance_type='weight'):
        """Get feature importance"""
        if importance_type == 'weight':
            return dict(enumerate(self.feature_importances_))
        else:
            raise ValueError(f"Unsupported importance_type: {importance_type}")
        

class TreeNode:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.weight = None
        self.is_leaf = False

class DecisionTree:
    def __init__(self, max_depth, gamma, min_child_weight, reg_lambda, colsample_bylevel):
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.colsample_bylevel = colsample_bylevel
        self.root = None
        self.feature_importances = {}
    
    def _build_tree(self, X, gradients, hessians, depth=0, feature_indices=None):
        """Recursively build the decision tree"""
        node = TreeNode()
        
        # Base cases for stopping
        if (depth >= self.max_depth or 
            len(X) < 2 or 
            np.sum(hessians) < self.min_child_weight):
            node.is_leaf = True
            node.weight = self._calculate_leaf_weight(gradients, hessians)
            return node
        
        # Feature sampling for this level
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        n_features_level = max(1, int(len(feature_indices) * self.colsample_bylevel))
        sampled_features = np.random.choice(feature_indices, n_features_level, replace=False)
        
        # Find best split
        best_gain, best_feature, best_threshold, left_indices, right_indices = self._find_best_split(
            X, gradients, hessians, sampled_features)
        
        # If no good split found, make leaf
        if best_gain <= 0:
            node.is_leaf = True
            node.weight = self._calculate_leaf_weight(gradients, hessians)
            return node
        
        # Record feature importance
        if best_feature not in self.feature_importances:
            self.feature_importances[best_feature] = 0
        self.feature_importances[best_feature] += best_gain
        
        # Create split
        node.feature = best_feature
        node.threshold = best_threshold
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(
            X[left_indices], gradients[left_indices], hessians[left_indices], 
            depth + 1, feature_indices)
        node.right = self._build_tree(
            X[right_indices], gradients[right_indices], hessians[right_indices], 
            depth + 1, feature_indices)
        
        return node
    
    def _calculate_leaf_weight(self, gradients, hessians):
        """Calculate optimal weight for a leaf node"""
        G = np.sum(gradients)
        H = np.sum(hessians)
        return -G / (H + self.reg_lambda)
    
    def _find_best_split(self, X, gradients, hessians, feature_indices):
        """Find the best split for current node"""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_gradients = gradients[left_mask]
                left_hessians = hessians[left_mask]
                right_gradients = gradients[right_mask]
                right_hessians = hessians[right_mask]
                
                if (np.sum(left_hessians) < self.min_child_weight or 
                    np.sum(right_hessians) < self.min_child_weight):
                    continue
                
                gain = self._calculate_gain(left_gradients, left_hessians,
                                          right_gradients, right_hessians,
                                          gradients, hessians)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = np.where(left_mask)[0]
                    best_right_indices = np.where(right_mask)[0]
        
        return best_gain, best_feature, best_threshold, best_left_indices, best_right_indices
    
    def _calculate_gain(self, left_gradients, left_hessians, right_gradients, right_hessians, parent_gradients, parent_hessians):
        """Calculate gain from a split"""
        left_score = self._calculate_similarity_score(left_gradients, left_hessians)
        right_score = self._calculate_similarity_score(right_gradients, right_hessians)
        parent_score = self._calculate_similarity_score(parent_gradients, parent_hessians)
        
        gain = 0.5 * (left_score + right_score - parent_score) - self.gamma
        return gain
    
    def _calculate_similarity_score(self, gradients, hessians):
        """Calculate similarity score for a node"""
        G = np.sum(gradients)
        H = np.sum(hessians)
        return (G ** 2) / (H + self.reg_lambda)
    
    def fit(self, X, gradients, hessians, feature_indices=None):
        """Build the tree"""
        self.root = self._build_tree(X, gradients, hessians, feature_indices=feature_indices)
    
    def predict(self, X):
        """Make predictions"""
        predictions = np.zeros(len(X))
        for i, sample in enumerate(X):
            predictions[i] = self._predict_single(sample, self.root)
        return predictions
    
    def _predict_single(self, sample, node):
        """Predict single sample"""
        if node.is_leaf:
            return node.weight
        
        if sample[node.feature] <= node.threshold:
            return self._predict_single(sample, node.left)
        else:
            return self._predict_single(sample, node.right)

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    X = np.random.randn(n_samples, n_features)
    y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + np.random.randn(n_samples) * 0.1
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = XGBoostRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Feature Importances: {model.get_feature_importance()}")


