"""
Module: models_classical.py
Description: Object-oriented wrapper handling standard scikit-learn baseline 
             execution blocks and understanding checking loops.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import config

class ScratchSVM:
    """A dual-QP SVM implementation with RBF kernel to validate mechanism understanding[cite: 1171]."""
    def __init__(self, C=1.0, gamma='scale'):
        self.C = C
        self.gamma_param = gamma
        self.gamma = None
        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.b = 0.0

    def _rbf_kernel_matrix(self, X1, X2):
        dist_sq = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X, y):
        y_mapped = np.where(y == 0, -1, y)
        n_samples, n_features = X.shape
        
        if self.gamma_param == 'scale':
            self.gamma = 1.0 / (n_features * X.var())
        elif self.gamma_param == 'auto':
            self.gamma = 1.0 / n_features
        else:
            self.gamma = self.gamma_param
            
        self.X_train = X
        self.y_train = y_mapped
        K = self._rbf_kernel_matrix(X, X)
        
        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(alpha * y_mapped, K) * y_mapped) - np.sum(alpha)
            
        constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y_mapped)}
        bounds = [(0, self.C) for _ in range(n_samples)]
        init_alpha = np.zeros(n_samples)
        
        opt_res = minimize(objective, init_alpha, method='SLSQP', bounds=bounds, constraints=constraints)
        self.alpha = opt_res.x
        
        sv_idx = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)
        if np.any(sv_idx):
            K_sv = self._rbf_kernel_matrix(X, X[sv_idx])
            self.b = np.mean(y_mapped[sv_idx] - np.dot(self.alpha * y_mapped, K_sv))
        else:
            self.b = 0.0

    def decision_function(self, X):
        K = self._rbf_kernel_matrix(self.X_train, X)
        return np.dot(self.alpha * self.y_train, K) + self.b

    def predict(self, X):
        decisions = self.decision_function(X)
        return np.where(np.sign(decisions) <= 0, 0, 1)

    def predict_proba(self, X):
        decisions = self.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-decisions))
        return np.column_stack((1.0 - probs, probs))


class ClassicalBaselineManager:
    """Manages tuning and execution of classical baselines[cite: 1212]."""
    def __init__(self, seed=config.SEED):
        self.seed = seed

    def fit_rbf_svc(self, X_train, y_train):
        grid = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=self.seed), 
                            config.SVC_PARAM_GRID, cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    def fit_scratch_svm(self, X_train, y_train, optimal_params):
        model = ScratchSVM(C=optimal_params['C'], gamma=optimal_params['gamma'])
        model.fit(X_train, y_train)
        return model

    def fit_xgboost(self, X_train, y_train):
        grid = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=self.seed), 
                            config.XGB_PARAM_GRID, cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_