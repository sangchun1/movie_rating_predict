from typing import Tuple

import codefast as cf
import fasttext
import pandas as pd
from codefast.patterns.pipeline import BeeMaxin, Pipeline
from codefast.utils import timeit_decorator
from rich import print

from premium.data.datasets import downloader

# —--------------------------------------------


class AutoTunner(BeeMaxin):

    def __init__(self,
                 dim=200,
                 autotuneDuration=300,
                 save_model_path=None) -> None:
        super().__init__()
        self.model = None
        self.dim = dim
        self.autotuneDuration = autotuneDuration
        self.save_model_path = save_model_path
        self.save_model_path = save_model_path

    @timeit_decorator('Autotunner.transform')
    def process(self, inputs: Tuple[str, str, str]):
        train_file, test_file, pretrainedVectors = inputs
        train_params = {
            'input': train_file,
            'dim': self.dim,
            'thread': 3,
            'autotuneValidationFile': test_file,
            'autotuneDuration': self.autotuneDuration,
        }
        if pretrainedVectors:
            train_params['pretrainedVectors'] = pretrainedVectors

        model = fasttext.train_supervised(**train_params)
        if self.save_model_path:
            model.save_model(self.save_model_path)
        self.model = model
        return self.model


class FastTextTrainer(BeeMaxin):

    def __init__(self, dim=200, threads=12, save_model_path=None):
        """ pretrainedVectors must starts with a line contains the number of
        words in the vocabulary and the size of the vectors. E.g., 100000 200
        """
        super().__init__()
        self.model = None
        self.threads = threads
        self.dim = dim
        self.save_model_path = save_model_path

    @timeit_decorator('FastTextTrainer.transform')
    def process(self, inputs: Tuple[str, str, str]):
        train_file, test_file, pretrainedVectors = inputs
        cf.info({
            'step': 'start training',
            'train_file': train_file,
            'test_file': test_file
        })
        parmas = {
            'input': train_file,
            'dim': self.dim,
            'thread': self.threads,
        }

        if pretrainedVectors:
            cf.info({'step': 'loading pretrainedVectors', 'pretrainedVectors': pretrainedVectors})
            parmas['pretrainedVectors'] = pretrainedVectors

        model = fasttext.train_supervised(**parmas)

        if self.save_model_path:
            cf.info({'step': 'saving model', 'save_model_path': self.save_model_path})
            model.save_model(self.save_model_path)

        cf.info({'step': 'start validating', 'test_file': test_file})
        sample_count, accuracy, recall = model.test(test_file)
        cf.info({
            'sample_count': sample_count,
            'accuracy': accuracy,
            'recall': recall,
        })


class TencentPretrainedVectorDownloader(BeeMaxin):

    def __init__(self, vector_name: str = 'tencent_cn_1M.txt'):
        super().__init__()
        self.vector_name = vector_name

    def process(self):
        return downloader.get(self.vector_name)


class CsvLoader(BeeMaxin):

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path

    def read_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        import jieba
        df['text'] = df['text'].apply(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))))
        df['label'] = df['label'].apply(lambda x: '__label__positive' if x
                                        else '__label__negative')
        df = df.sample(frac=1, random_state=314159)
        df = df[['label', 'text']]
        return df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, test = df[:int(len(df) * 0.8)], df[int(len(df) * 0.8):]
        cf.info({
            'message': 'split dataset',
            'train size': len(train),
            'test size': len(test),
        })
        return train, test

    def dump_text(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False, header=False, sep=' ')
        return path

    def process(self, pretrainedVectors=None) -> Tuple[str, str]:
        df = self.read_csv(self.csv_path)
        train, test = self.split(df)
        return self.dump_text(train, '/tmp/train.txt'), self.dump_text(
            test, '/tmp/test.txt'), pretrainedVectors


if __name__ == '__main__':
    # 注：中文需要分词
    pipeline = Pipeline([
        ('vector_getter', TencentPretrainedVectorDownloader()),
        ('csv_loader', CsvLoader('/tmp/gender.csv')),
        ('ftt', FastTextTrainer()),
        # ('autotuner', AutoTunner())
    ])
    pipeline.gather()
