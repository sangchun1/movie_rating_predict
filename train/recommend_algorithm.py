import pandas as pd
import MySQLdb
import numpy as np
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

class Recommend:
    def __init__(self, db):
        self.db = db

    