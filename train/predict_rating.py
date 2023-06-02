import pandas as pd
import numpy as np
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

class Predict:
    def __init__(self, model_name, db):
        self.model = load_model(model_name)
        self.db = db

    def text_preprocess(self, X):
        # 특수문자,기호 제거
        X = X.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]","")
        # 공백 제거
        X = X.replace('^ +', "")
        X.replace('', np.nan, inplace=True)

        # 불용어 사전
        with open("c:/data/movie_list/불용어사전.txt") as f:
            lines = f.read().splitlines()
        stopwords = []
        for line in lines:
            stopwords.append(line.split("\t")[0])

        # 형태소 분석
        okt = Okt()
        X_lis = []
        for sentence in X:
            temp_X = okt.morphs(sentence, stem=True) # 토큰화
            temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
            X_lis.append(temp_X)
        
        # 정수 인코딩
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_lis)

        # 독립변수 패딩
        X = pad_sequences(X, maxlen = len(X))

        return X

    def predict_rating(self, user_id, review_id):
        # db 연결
        conn = self.db
        # 커서 생성
        cursor = conn.cursor()

        # 테이블 불러오기
        sql = f'''
        select * from user_review
        where user_id = {user_id} 
        and review_id = {review_id}
        '''
        cursor.execute(sql)
        rows = cursor.fetchall()

        # 테이블을 데이터 프레임으로 저장
        review_init = pd.DataFrame(rows)
        print(review_init)

        # 평점 예측
        X = self.text_preprocess(review_init['review'])
        rating = float(self.model.predict(X))
        if rating <= 0: rating = 0
        elif rating >= 10: rating = 10 / 2
        else: rating = rating / 2

        # 테이블에 다시 넣기
        sql = f'''
        update user_review set rating_predict = {rating}
        where user_id = {user_id} 
        and review_id = {review_id} 
        '''

        # 종료
        cursor.close()
        conn.close()