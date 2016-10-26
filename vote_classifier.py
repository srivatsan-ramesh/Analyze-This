import pandas as pd
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer, MinMaxScaler
from numpy import float64
from sklearn.preprocessing.label import _check_numpy_unicode_bug
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

fname = './trained_models/vote_full.pkl'

degree_map = {'Primary': 0, 'Diploma': 1, 'Degree': 2, 'Masters': 3}
loc_map = {}

l = 0


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, s, e=None):
        self.s = s
        self.e = e

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.e is None:
            return np.array(data_dict[:, self.s]).reshape(-1, 1)
        else:
            return np.array(data_dict[:, self.s:self.e])


class ItemMap(BaseEstimator, TransformerMixin):
    def __init__(self, s, tog, loc_map=None):
        self.s = s
        self.tog = tog
        self.loc_map = loc_map

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.tog:
            return np.array(pd.DataFrame(data_dict[:, self.s]).replace(degree_map).as_matrix()).reshape(-1, 1)
        else:
            return np.array(pd.DataFrame(data_dict[:, self.s]).replace(self.loc_map).as_matrix()).reshape(-1, 1)


class LabelEncoderNew(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        X = column_or_1d(X.ravel(), warn=True)
        _check_numpy_unicode_bug(X)
        self.classes_ = np.unique(X)
        if isinstance(self.classes_[0], np.float64):
            self.classes_ = self.classes_[np.isfinite(self.classes_)]
        return self

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y.ravel(), warn=True)

        classes = np.unique(y)
        if isinstance(classes[0], np.float64):
            classes = classes[np.isfinite(classes)]
        _check_numpy_unicode_bug(classes)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))
        return np.searchsorted(self.classes_, y).reshape(-1, 1)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if diff:
            raise ValueError("y contains new labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]


pipeline = Pipeline([
    ('features', FeatureUnion(
        transformer_list=[
            ('p1', Pipeline([
                ('f1', ItemSelector(0)),
                ('le1', LabelEncoderNew())
            ])),
            ('f2', ItemSelector(1, 27)),
            ('p3', Pipeline([
                ('f3', ItemSelector(27)),
                ('le3', LabelEncoderNew())
            ])),
            ('f4', ItemSelector(28, 30)),
            ('f5', ItemMap(30, True)),
            ('f6', ItemSelector(31, 32)),
            ('f7', ItemMap(32, False)),
            ('f8', ItemMap(33, False))
        ]
    )),
    ('imp', Imputer()),
    ('ohe1', OneHotEncoder(categorical_features=(0, 27, 30, 32, 33), dtype=np.float64, sparse=False,
                           handle_unknown='ignore')),
    ('ohe2', OneHotEncoder(categorical_features=(32, 33), dtype=np.float64, sparse=False, handle_unknown='ignore',
                           n_values=l)),
    ('sc', MinMaxScaler()),
    ('pca', PCA(n_components=500, svd_solver='arpack')),
    ('lr', LogisticRegression())

])

df_train = pd.read_csv('Training_Dataset.csv')
df_test = pd.read_csv('Leaderboard_Dataset.csv')

u_vals = np.unique(np.r_[df_train['mvar32'].as_matrix(), df_test['mvar32'].as_matrix(),
                         df_train['mvar33'].as_matrix(), df_test['mvar33'].as_matrix()])

l = u_vals.shape[0]
loc_map = dict((key, val) for (key, val) in np.c_[u_vals, range(l)])

pipeline.set_params(ohe2__n_values=l, features__f7__loc_map=loc_map, features__f8__loc_map=loc_map)

X, y = df_train.iloc[:, 1:-1].values, df_train.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y)
if not os.path.isfile(fname):
    pipeline.fit(X, y)
    s = joblib.dump(pipeline, fname)
else:
    pipeline = joblib.load(fname)

y_pred = pipeline.predict(df_test.iloc[:, 1:].values)

y_pred = le.inverse_transform(y_pred)

pd.DataFrame(np.c_[df_test.iloc[:, 0].values, y_pred]).to_csv('nelsonmurdock_IITMadras_1.csv', header=False, index=False)
