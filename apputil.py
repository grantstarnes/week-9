import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate):
        if estimate not in ['mean', 'median']:
            raise ValueError("The estimate must be either 'mean' or 'median.'")
        self.estimate = estimate
    
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a valid pandas DataFrame")
        if len(X) != len(y):
            raise ValueError("X and y should have the same number of rows")
        if pd.isnull(y).any():
            raise ValueError("y must not contain missing values")

    def predict(self, X):
        return None