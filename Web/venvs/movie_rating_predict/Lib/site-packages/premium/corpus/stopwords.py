#!/usr/bin/env python
from typing import List

import codefast as cf

from .utils import LocalData


class Stopwords(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_tokens'):
            self._tokens = {}

    @classmethod
    def get_cn_tokens(cls) -> List[str]:
        instance = cls()
        return instance.cn

    @classmethod
    def get_en_tokens(cls) -> List[str]:
        instance = cls()
        return instance.en

    @property
    def cn(self) -> List[str]:
        if 'cn' not in self._tokens:
            file_path = LocalData('cn_stopwords.txt').fullpath()
            self._tokens['cn'] = cf.io.read(file_path)
        return self._tokens['cn']

    @property
    def en(self) -> List[str]:
        if 'en' not in self._tokens:
            file_path = LocalData('en_stopwords.txt').fullpath()
            self._tokens['en'] = cf.io.read(file_path)
        return self._tokens['en']
