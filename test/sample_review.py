import pandas as pd
import random
import MySQLdb
import numpy as np
from train.predict_rating import Predict


# 데이터 불러오기
df1 = pd.read_csv('../data/review-rating1.csv')
df2 = pd.read_csv('../data/review-rating2.csv')
df3 = pd.read_csv('../data/review-rating3.csv')
df4 = pd.read_csv('../data/review-rating4.csv')
df = pd.concat([df1, df2, df3, df4])

# 중복값 제거
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True, drop=True)

# 랜덤으로 200개의 리뷰 가져오기
random_indices = random.sample(range(len(df)), k=500)
temp = df.loc[random_indices]
temp.drop_duplicates(subset=['영화ID'], inplace=True)
random_indices = random.sample(range(len(temp)), k=200)
df_random = temp.loc[random_indices]

df_random = df_random.drop(['영화명', '평균평점'], axis=1).rename({'실제평점':'rating', '리뷰':'review', '영화ID':'movie_id'}, axis=1)
df_random['created_at'] = '2023-06-02'
df_random['user_id'] = 1
df_random['rating'] = df_random['rating'] / 2
df_random['rating_predict'] = np.nan
df_random = df_random.astype({'created_at':'datetime64'})
df_random = df_random[['review', 'created_at', 'movie_id', 'user_id', 'rating', 'rating_predict']]

# MySQL 서버 연결
conn = MySQLdb.connect(host='localhost', port='3306', user='', passwd='', db='')

# 데이터 프레임을 db 테이블로 옮기기
table_name = ''
df_random.to_sql(name=table_name, con=conn, if_exists='append', index=False)

# 연결 종료
conn.close()

predict = Predict('../model/review_RNN.h5', conn)

for i in df_random.index:
    predict.predict_rating(1, i)