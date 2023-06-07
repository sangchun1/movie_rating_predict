import pandas as pd
import numpy as np
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping
from keras.models import load_model
from datetime import datetime

class Recommend:
    def __init__(self, db, user_id):
        self.db = db
        self.user_id = user_id

    # 날짜를 넣으면 기준일로부터 개봉한지 몇일 됐는지 출력
    def change_date(self, date):
        now = datetime.now() # 현재 날짜와 시간 얻기

        if len(date) == 11: # 양식이 '2023.06.04.' 일때
            compare_date = datetime.strptime(date, '%Y.%m.%d.')
            time_difference = now - compare_date # 날짜 차이 계산
            day = time_difference.days # 차이를 일수로 변환

        elif len(date) == 10: # 양식이 '2023-06-04' 일때
            compare_date = datetime.strptime(date, '%Y-%m-%d')
            time_difference = now - compare_date
            day = time_difference.days

        elif len(date) == 8:
            add_day = date + '01.' # 양식이 '2023.06.' 일때
            compare_date = datetime.strptime(add_day, '%Y.%m.%d.')
            time_difference = now - compare_date
            day = time_difference.days
            
        else:
            add_day = date + '.01.01.' # 양식이 '2023' 일때
            compare_date = datetime.strptime(add_day, '%Y.%m.%d.')
            time_difference = now - compare_date
            day = time_difference.days
        return day
    
    # 국가 단순화
    def change_country(self, country):
        # 한국, 미국, 일본, 유럽, 아시아, 기타
        if country == '대한민국' or country == '한국':
            country = '한국'
        elif country == '미국':
            country = '미국'
        elif country == '일본':
            country = '일본'
        elif country == '대만' or country == '말레이시아' or country == '필리핀' or country == '베트남' or country == '타이' or country == '중국' or country == '미얀마' or country == '이란' or country == '홍콩' or country == '싱가포르' or country == '아프가니스탄' or country == '인도네시아':
            country = '아시아'
        elif country == '영국' or country == '벨기에' or country == '프랑스' or country == '덴마크' or country == '독일' or country == '독일(구 서독)' or country == '아일랜드' or country == '이탈리아' or country == '슬로베니아' or country == '에스토니아' or country == '스위스' or country == '크로아티아' or country == '슬로바키아' or country == '체코' or country == '루마니아' or country == '스페인' or country == '폴란드' or country == '네덜란드' or country == '리투아니아' or country == '스웨덴' or country == '불가리아' or country == '룩셈부르크' or country == '그리스' or country == '포르투갈' or country == '그린란드':
            country = '유럽'
        else:
            country = '기타'
        return country
    
    # 장르 단순화
    def change_genre(self, genres):
        # 9 종류 및 기타
        genre_list = []

        try:
            g_split = genres.split(', ')
        except:
            g_split = genres.split(',')
        finally:
            for g in g_split:
                if g == '어드벤처' or g == '모험' or g == '서사':
                    g = '모험'
                elif g == '범죄' or g == '느와르':
                    g = '범죄'
                elif g == '코미디' or g == '블랙코미디':
                    g = '코미디'
                elif g == 'SF':
                    g = 'SF'
                elif g == '판타지':
                    g = '판타지'
                elif g == '멜로/로맨스' or g == '드라마' or g == '가족':
                    g = '멜로/로맨스'
                elif g == '스릴러' or g == '서스펜스':
                    g = '스릴러'
                elif g == '공포' or g == '공포(호러)':
                    g = '공포'
                elif g == '액션' or g == '무협':
                    g = '액션'
                else:
                    g = '기타'
                genre_list.append(g)
                genres = list(set(genre_list)) # 리스트로 반환
                # genres = ', '.join(genres) # 문자열로 반환
        return genres
    
    # 배우
    def actor_model(self, actors):
        X = actors['actor']
        y = actors['rating']

        # 배우 리스트
        X_lis = X.split(', ')

        # 정수 인코딩
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_lis)
        X = tokenizer.texts_to_sequences(X_lis)

        # 모델 생성


        # 모델 학습
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3)
        model.fit(X, y, batch_size=64, epochs=10, validation_split=0.2, callbacks=es)
        
        # 결과 시리즈로 저장
        series = model.predict(X)
        
        return series
    
    # 줄거리
    def summary_model(self, summary):
        X = summary['summary']
        y = summary['rating']

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
        X = tokenizer.texts_to_sequences(X_lis)

        # 줄거리의 최대 길이
        max_len = max(len(l) for l in X)

        # 독립변수 패딩
        X = pad_sequences(X, maxlen=max_len)

        # 모델 생성


        # 모델 학습
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3)
        model.fit(X, y, batch_size=64, epochs=10, validation_split=0.2, callbacks=es)
        
        # 결과 시리즈로 저장
        series = model.predict(X)

        return series
    
    # 테이블을 조인하여 데이터 프레임으로 저장
    def table_to_df(self):
        # db 연결
        conn = self.db
        # 커서 생성
        cursor = conn.cursor()

        # 테이블을 조인하여 불러오기
        sql = f'''
        select u.movie_id movie_id, movie_totsale totsale, movie_attendance attendance, movie_screen screen, movie_screening screening, 
        movie_date date, movie_world country, movie_genre genre, movie_time time, movie_director director, movie_actor actor, movie_summary summary, rating
        from movie_info m, user_review u
        where u.movie_id = m.movie_id
        and user_id = {self.user_id}
        '''
        cursor.execute(sql)
        rows = cursor.fetchall()

        # 테이블을 데이터 프레임으로 저장
        df = pd.DataFrame(rows)

        return df
    
    # 학습을 위한 전처리
    def df_preprocess(self, df):
        for i in df.index:
            df.loc[i,'released_date'] = self.change_date(df.loc[i,'released_date'])
            df.loc[i,'country'] = self.change_country(df.loc[i,'country'])
            df.loc[i,'genre'] = self.change_genre(df.loc[i,'genre'])

        # 국가 원핫인코딩
        

        # 장르 원핫인코딩


        # 텍스트 컬럼은 미리 선호도 학습
        df['actor'] = self.actor_model(df[['actor', 'rating']])
        df['summary'] = self.summary_model(df[['summary', 'rating']])

        # 스케일링


        return df

    # 최종 선호도 학습
    def create_algorithm(self, df):
        # X, y로 분류
        X = df.loc[:-1]
        y = df['rating']
        # y = df['rating_predict']

        # 모델 생성
        model = models.Sequential()
        model.add(layers.Dense(units=32, activation="relu", input_shape=(X.shape[1],)))
        model.add(layers.Dense(units=16, activation="relu"))
        model.add(layers.Dense(units=1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
        
        # 모델 학습
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
        model.fit(X, y, batch_size=64, epochs=50, validation_split=0.2, callbacks=es)
        model.save(f'../model/recommend_for{self.user_id}.h5')

    # 생성한 알고리즘으로 다른 영화의 예상 선호도 출력
    def recomment_top5(self):
        model = load_model(f'../model/recommend_for{self.user_id}.h5')
        
        # 영화 정보 가져오기
        # db 연결
        conn = self.db
        # 커서 생성
        cursor = conn.cursor()

        # 모든 영화 정보 테이블 불러와 데이터프레임으로 저장
        sql = "select * from movie_info"
        cursor.execute(sql)
        rows = cursor.fetchall()
        df = pd.DataFrame(rows)

        # 사용자가 리뷰를 단 영화들을 보기 위한 데이터프레임
        sql = f'''
        select movie_id from user_review
        where user_id = {self.user_id}
        '''
        cursor.execute(sql)
        rows = cursor.fetchall()
        temp = rows

        # 아직 리뷰를 안 단 영화들로만 정리
        df = df[~df.isin(temp)].dropna()

        # 학습을 위한 전처리
        X = self.df_preprocess(self, df)

        # 예측
        lis = model.predict(X)

        # top5 리스트
        top5_lis = sorted(lis, reverse=True)[:5]
        indices = [i for i, value in enumerate(lis) if value in top5_lis]

        # top5 리스트의 영화 아이디까지 포함해 리스트로 생성
        top5 = []
        for i in indices:
            top5.append([df.loc[i, 'movie_id'], lis[i]])

        return top5