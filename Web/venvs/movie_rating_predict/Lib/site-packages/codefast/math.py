from typing import Tuple, List


class Math(object):
    def __init__(self):
        ...

    @classmethod
    def n_fib(cls, n: int) -> int:
        '''https://stackoverflow.com/questions/4935957/fibonacci-numbers-with-an-one-liner-in-python-3'''
        return pow(2 << n, n + 1, (4 << 2 * n) - (2 << n) - 1) % (2 << n)

    @classmethod
    def fibs(cls, n: int) -> List[int]:
        return list(map(cls.n_fib, range(1, n + 1)))

    @classmethod
    def cosine_similarity(cls, vec1: List[int], vec2: List[int]) -> float:
        import numpy as np
        array1 = np.array(vec1)
        array2 = np.array(vec2)
        return np.dot(array1, array2) / (
            np.linalg.norm(array1) * np.linalg.norm(array2) + 1e-8)
    
    