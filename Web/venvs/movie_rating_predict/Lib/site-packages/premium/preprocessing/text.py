#!/usr/bin/env python
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np


def text2vector(text: str, embeddings: Dict) -> np.ndarray:
    """ Convert text to vector using word2vec embeddings.
    """
    import jieba
    words = jieba.lcut(text.strip())
    vectors = [embeddings.get(word, embeddings.get('unk')) for word in words]
    return np.mean(vectors, axis=0)
