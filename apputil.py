import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate):
        if estimate not in ['mean', 'median']:
            raise ValueError("The estimate must be either 'mean' or 'median.'")
        self.estimate = estimate
    
    def fit(self, X, y):
        return None

    def predict(self, X):
        return None