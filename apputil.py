import pandas as pd


class GroupEstimate(object):

    '''
    The GroupEstimate class is utilized here for estimating values, whether mean or median, based on the 
    grouping of categories in the DataFrame. It takes in an estimatation value, mean or median, and will store the values
    based on which estimation is called in a DataFrame.
    '''

    def __init__(self, estimate):

        '''
        Determines which estimate to calculate and raises a ValueError if the estimate is not
        either a mean or median.
        '''

        if estimate not in ['mean', 'median']:
            raise ValueError("The estimate must be either 'mean' or 'median.'")
        self.estimate = estimate


    
    
    def fit(self, X, y):

        '''
        This .fit method fits the GroupEstimate and computes the mean or median for each group/category. 
        It takes in X which holds all of the categories, as well as y which holds the values.
        If X is not a DataFrame, an error is raised stating so. If the length of X and y differ,
        an error is raised stating that they should have the same number of rows, and it currently differs.
        Lastly, if y contains missing values, an error will also be raised for that.
        '''

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

        '''
        This predict method is used to predict the estimated target values for the combined categories. 
        It will return an array for these predicted estimates, mean or median, for every row within X.
        If a group is missing, it will print the number of missing groups if there are any and replace the value
        with nan if so.
        '''
        
        if not hasattr(self, "group_estimates_") or self.group_estimates_ is None:
            raise RuntimeError(".fit() must be ran before .predict()")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns = self.group_estimates_.columns[:-1])

        merged = X.merge(self.group_estimates_, on = list(self.group_estimates_.columns[:-1]), how = 'left')


        count_missing = merged["estimate"].isna().sum()

        if count_missing > 0:
            print(f"There are {count_missing} missing groups")

        return merged["estimate"].values