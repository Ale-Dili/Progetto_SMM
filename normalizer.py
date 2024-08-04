import pandas as pd

class Normalizer:
    def __init__(self, n_bins=4, normalization="z-score", encoding=None):
        self.n_bins = n_bins
        if normalization in ["z-score", "minmax"]:
            self.normalization = normalization
        else:
            self.normalization = None
        if encoding in ['onehot']:
            self.encoding = encoding
        else:
            self.encoding = None

    def normalize(self, X):
        for colname in X.columns:
            if colname in ['cap-diameter', 'stem-height', 'stem-width']:
                X[colname] = self._numerical_norm(X[colname])
            else:
                X = self._categorical_norm(X, colname)
        return X

    # One hot encoding
    def _categorical_norm(self, X, colname):
        if self.encoding:
            dummies = pd.get_dummies(X[colname], prefix=colname)
            X = pd.concat([X, dummies], axis=1)
            X.drop(colname, axis=1, inplace=True)
        return X

    # Z-score or minmax normalization
    def _numerical_norm(self, column):
        if self.normalization == 'z-score':
            mean = column.mean()
            std = column.std()
            column = (column - mean) / std
        elif self.normalization == 'minmax':
            column = (column - column.min()) / (column.max() - column.min())
        return column

    #fix label 
    def _fix_label(self, y):
        y['class'] = y['class'].apply(lambda x: 1 if x == 'e' else 0)
        return y
