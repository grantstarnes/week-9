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
        
        df = X.copy()
        df["__target__"] = y

        if self.estimate == 'mean':
            group_estimates = df.groupby(list(X.columns))["__target__"].mean().reset_index()
        else:
            group_estimates = df.groupby(list(X.columns))["__target__"].median().reset_index()

        group_estimates.rename(columns = {"__target__": "estimate"}, inplace=True)

        self.group_estimates_ = group_estimates

    def predict(self, X):
        
        if not hasattr(self, "group_estimates_") or self.group_estimates_ is None:
            raise RuntimeError(".fit() must be ran before .predict()")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns = self.group_estimates_.columns[:-1])

        merged = X.merge(self.group_estimates_, on = list(self.group_estimates_.columns[:-1]), how = 'left')


        count_missing = merged["estimate"].isna().sum()

        if count_missing > 0:
            print(f"There are {count_missing} missing groups")

        return merged["estimate"].values