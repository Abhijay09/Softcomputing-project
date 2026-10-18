"""Fuzzy rule-based character classifier.

Every class owns a few fuzzy rules (found with fuzzy c-means):
    IF feature1 is ~c1 AND feature2 is ~c2 AND ... AND feature177 is ~c177  THEN class = K
'is ~c' is a Gaussian membership function. AND = geometric mean of the memberships,
rules of the same class are combined with OR = max, and the answer is the class with the
largest membership (max-membership defuzzification)."""
import numpy as np
from fuzzy_logic import fuzzy_c_means

MIN_SIGMA_FRAC = 0.3   # a rule is never narrower than 30% of the feature's overall spread


class FuzzyClassifier:
    def __init__(self, rules_per_class=4):
        self.k = rules_per_class

    def fit(self, X, y):
        self.classes = np.unique(y)
        floor = MIN_SIGMA_FRAC * X.std(axis=0)
        C, S, R = [], [], []
        for ci, cls in enumerate(self.classes):
            Xc = X[y == cls]
            centres, U = fuzzy_c_means(Xc, self.k, seed=ci)
            Um = U ** 2
            for r in range(self.k):                      # spread of each rule = weighted std dev
                var = (Um[:, r:r + 1] * (Xc - centres[r]) ** 2).sum(0) / Um[:, r].sum()
                S.append(np.sqrt(var))
                C.append(centres[r])
                R.append(ci)
        self.centres, self.sigmas, self.rule_class = np.array(C), np.maximum(np.array(S), floor), np.array(R)
        return self

    def memberships(self, X):
        """Membership of each sample in each class, shape (n, n_classes)."""
        X = np.atleast_2d(X)
        out = np.zeros((len(X), len(self.classes)), np.float32)
        for i in range(0, len(X), 512):
            d = (X[i:i + 512, None, :] - self.centres[None]) ** 2 / (2 * self.sigmas[None] ** 2)
            fire = np.exp(-d.mean(axis=2))               # geometric mean of Gaussian memberships
            for ci in range(len(self.classes)):          # OR over the rules of one class
                out[i:i + 512, ci] = fire[:, self.rule_class == ci].max(axis=1)
        return out

    def predict(self, X):
        return self.classes[self.memberships(X).argmax(1)]

    def save(self, path):
        np.savez(path, centres=self.centres, sigmas=self.sigmas, rule_class=self.rule_class, classes=self.classes)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls()
        m.centres, m.sigmas, m.rule_class, m.classes = d["centres"], d["sigmas"], d["rule_class"], d["classes"]
        return m
