#!/usr/bin/env python
import timeit
from rich import print
from premium.data.datasets import downloader
import codefast as cf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import (Dense, Embedding, GlobalAveragePooling1D,
                                     TextVectorization)
from tensorflow.keras.models import Sequential

try:
    import fasttext
except ImportError as e:
    cf.error("import fasttext failed", e)


def myfastText(df: pd.DataFrame,
               embedding_dims: int = 100,
               ngrams: int = 2,
               max_features: int = 30000,
               maxlen: int = 400,
               batch_size: int = 32,
               epochs: int = 10):
    """ A simple implementation of fastText.
    """
    X, Xv, y, yv = train_test_split(df['text'], df['target'], random_state=0)
    args = {
        'dim': embedding_dims,
        'ngrams': ngrams,
        'max_features': max_features,
        'maxlen': maxlen,
        'batch_size': batch_size,
        'epochs': epochs
    }
    cf.info(args)
    vectorize_layer = TextVectorization(
        ngrams=ngrams,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=maxlen,
    )
    vectorize_layer.adapt(X.values)
    model = Sequential([
        vectorize_layer,
        Embedding(max_features + 1, embedding_dims),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(Xv, yv),
    )
    return model, history


class InvalidVectorException(Exception):
    pass


def check_pretrained_vector(vector_path: str) -> bool:
    """<pretrainedVectors> file must starts with a line contains the number of
    words in the vocabulary and the size of the vectors. E.g., 
    100000 200
    Refer: https://fasttext.cc/docs/en/english-vectors.html
    """
    with open(vector_path, 'r') as f:
        first_line = f.readline().strip()
        if ' ' not in first_line:
            raise InvalidVectorException('Invalid vector file')
        num_words, dim = first_line.split(' ')
        if not num_words.isdigit() or not dim.isdigit():
            raise InvalidVectorException('Invalid vector file')
        return True


def split_data(df: pd.DataFrame):
    assert 'text' in df.columns, 'text column not found'
    assert 'target' in df.columns, 'target column not found'
    df['target'] = '__label__' + df['target'].astype(str)
    df['text'] = df['text'].astype(str)
    df['label'] = df['target']
    msg = {'label_count': df['label'].value_counts()}
    cf.info(msg)

    fasttext_input = df[['target', 'text']].astype(str)
    size = int(fasttext_input.shape[0] * 0.8)
    fasttext_train = fasttext_input.sample(size, random_state=0)
    fasttext_valid = fasttext_input.drop(fasttext_train.index)
    cf.info('Train data size: {}'.format(len(fasttext_train)))
    cf.info('Valid data size: {}'.format(len(fasttext_valid)))

    fasttext_train.to_csv("/tmp/tt.train",
                          quotechar=" ",
                          header=False,
                          index=False)
    fasttext_valid.to_csv("/tmp/tt.test",
                          quotechar=" ",
                          header=False,
                          index=False)
    return fasttext_train, fasttext_valid


def baseline(train_file: str,
             test_file: str,
             dim: int = 200,
             pretrainedVectors: str = None,
             model_path: str = None,
             deprecate_split: bool = False,
             *args):
    """ 
    Inputs:
        deprecate_split: bool, do not split data again if True. 
    """
    model_path = '/tmp/pyfasttext.bin' if not model_path else model_path
    cf.info('start training')
    train_args = {
        'input': train_file,
        'dim': dim,
        'thread': 12,
    }

    if pretrainedVectors:
        check_pretrained_vector(pretrainedVectors)
        train_args['pretrainedVectors'] = pretrainedVectors
    model = fasttext.train_supervised(**train_args)
    model.save_model(model_path)

    # validate the model
    res = model.test(test_file)
    cf.info('validate result {}'.format(res))
    return model


def autotune(df: pd.DataFrame,
             dim: int = 200,
             pretrainedVectors: str = None,
             model_path: str = None,
             autotuneDuration: float = 300,
             *args):
    # Find the best possible hyperparameters
    model_path = '/tmp/pyfasttext.bin' if not model_path else model_path
    _, _ = split_data(df)
    cf.info('start training')
    train_args = {
        'input': '/tmp/tt.train',
        'dim': dim,
        'thread': 12,
        'autotuneValidationFile': '/tmp/tt.test',
        'autoTuneDuration': autotuneDuration,
    }

    if pretrainedVectors:
        check_pretrained_vector(pretrainedVectors)
        train_args['pretrainedVectors'] = pretrainedVectors
    cf.info('Auto tune started...')
    model = fasttext.train_supervised(**train_args)
    model.save_model(model_path)
    return model


def train_vector(df: pd.DataFrame, *kargs, **kwargs):
    pass


from codefast.utils import timeit_decorator
class AutoTunner(BaseEstimator, TransformerMixin):
    def __init__(self,
                 train_file,
                 test_file,
                 dim=200,
                 pretrainedVectors=None,
                 autotuneDuration=300,
                 save_model_path=None
                 ) -> None:
        self.model = None
        self.test_file = test_file
        self.train_params = {
            'input': train_file,
            'dim': dim,
            'thread': 12,
            'autotuneValidationFile': test_file,
            'autotuneDuration': autotuneDuration,
        }
        if pretrainedVectors:
            self.train_params['pretrainedVectors'] = pretrainedVectors
        self.save_model_path = save_model_path

    def transform(self, X, y=None):
        model = fasttext.train_supervised(**self.train_params)
        if self.save_model_path:
            model.save_model(self.save_model_path)
        self.model = model
        return self.model

from codefast.utils import timeit_decorator

class FastTextTrainer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 train_file,
                 test_file,
                 dim=200,
                 pretrainedVectors=None,
                 threads=12,
                 save_model_path=None):
        """ pretrainedVectors must starts with a line contains the number of 
        words in the vocabulary and the size of the vectors. E.g., 100000 200
        """
        self.model = None
        self.test_file = test_file
        self.train_params = {
            'input': train_file,
            'dim': dim,
            'thread': threads,
        }
        if pretrainedVectors:
            self.train_params['pretrainedVectors'] = pretrainedVectors
        self.save_model_path = save_model_path

    def fit(self, X, y=None):
        return self

    @timeit_decorator
    def transform(self, X, y=None):
        cf.info('start training')
        model = fasttext.train_supervised(**self.train_params)
        if self.save_model_path:
            model.save_model(self.save_model_path)
        cf.info('start validating')
        valid_result = model.test(test_file)
        cf.info('validate result {}'.format(valid_result))
        self.model = model
        return model


class TencentPretrainedVectorDownloader(BaseEstimator, TransformerMixin):
    def __init__(self, vector_name: str = 'tencent_cn_1M.txt'):
        self.vector_name = vector_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        downloader.get(self.vector_name)
        cf.info('[DONE] downloaded vector file: {}'.format(self.vector_name))


if __name__ == '__main__':
    # note, 中文需要分词
    train_file = '/tmp/train.txt'
    test_file = '/tmp/test.txt'
    vecfile = '/tmp/tencent_cn_1M.txt'
    pipeline = Pipeline([
        ('tencent', TencentPretrainedVectorDownloader()),
        ('ft', FastTextTrainer(train_file, test_file,
                               pretrainedVectors=vecfile)),
    ])
    pipeline.fit_transform(None)
