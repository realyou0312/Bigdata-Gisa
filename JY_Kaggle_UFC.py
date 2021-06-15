# ref) https://www.datamanim.com/dataset/01_ufc/main.html

import pandas as pd
pd.set_option('display.max_columns', 100)
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

ufc =pd.read_csv('https://raw.githubusercontent.com/Datamanim/ufc/main/ufc.csv')

print(ufc.head())

# 성별에 관한 컬럼 Gender를 추가한다. 남성의경우 male, 여성의경우 Female로 표기
# (Red, Blue 선수 상관없이 해당선수들의 성별을 하나의 컬럼에 입력)

ufc['Gender'] = ufc['weight_class'].map(lambda x: 'Female' if 'Women' in x else 'male')

print(ufc['Gender'][:5])

# Gender 컬럼의 value 발생 빈도확인

print(ufc['Gender'].value_counts())

# ‘year’ 컬럼에 해당 경기 발생 년도를 입력하라. 년도별 남성,여성각각의 경기숫자를 시각화 하여라

ufc['date'] = pd.to_datetime(ufc['date'].str[-4:]+'-'+ufc['date'].str[:2]+'-'+ufc['date'].str[3:5])
ufc['year'] = ufc['date'].dt.year

gender_cnt = ufc.groupby(['year', 'Gender']).count()['date'].reset_index()

print(gender_cnt)

# 몇년도 몇월에 가장 많은 경기가 있었는지 확인하라. ★★★

month_df = ufc.date.dt.strftime('%Y-%m').value_counts().sort_index()
print(month_df[month_df == month_df.max()].index[0])

# 타이틀매치(column : title_bout=True)에 가장 많은 경기를 심판(column : Referee)본 인물과 그 횟수는?

print(ufc[ufc.title_bout==True].Referee.value_counts().head(1))

# 년도에 따른 체급별 타이틀 매치에 관한 데이터를 아래와 같이 새롭게 구성하라 ★★★ 피벗테이블

pv_table = ufc[ufc.title_bout ==True][['weight_class','year']].value_counts().reset_index().\
                                        pivot(index='weight_class',columns='year').fillna(0).astype(int)

print(pv_table)