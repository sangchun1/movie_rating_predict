#!/usr/bin/env python3
import time

import codefast as cf
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFormer(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        start_time = time.time()
        res = self.inner_transform(X)
        end_time = time.time()
        period = round(end_time - start_time, 2)
        cf.info(f"{self.__class__.__name__} took {period} seconds")
        return res

    def inner_transform(self, X):
        raise NotImplementedError


class DataCollection(object):

    def __init__(self, X, y, test, sub, target):
        self.X = X
        self.y = y
        self.test = test
        self.sub = sub
        self.target = target
        self.models = {}
        self.y_preds = {}
        self.cv = 5
        self.scores = {}


class DefaultArgs(object):

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
