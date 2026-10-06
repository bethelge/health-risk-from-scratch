import numpy as np

class GaussianNaiveBayesScratch:
    """
    Gaussian Naive Bayes, implemented from first principles with NumPy only.

    For each class, we learn:
      - the prior P(class) = fraction of training patients in that class
      - the mean and variance of each feature, within that class

    Prediction uses log-space Bayes' theorem:
      log P(class | x) proportional to log P(class) + sum_i log P(x_i | class)
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        self.mean_ = np.zeros((len(self.classes_), n_features))
        self.var_ = np.zeros((len(self.classes_), n_features))
        self.priors_ = np.zeros(len(self.classes_))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

        return self

    def _log_gaussian_likelihood(self, class_idx: int, X: np.ndarray) -> np.ndarray:
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        log_coeff = -0.5 * np.log(2 * np.pi * var)
        exponent = -((X - mean) ** 2) / (2 * var)
        return log_coeff + exponent

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes_)))

        for idx in range(len(self.classes_)):
            log_prior = np.log(self.priors_[idx])
            log_likelihood = self._log_gaussian_likelihood(idx, X).sum(axis=1)
            log_probs[:, idx] = log_prior + log_likelihood

        return log_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        best_idx = np.argmax(log_probs, axis=1)
        return self.classes_[best_idx]
