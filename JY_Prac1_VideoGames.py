import pandas as pd
pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 100)
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/video/master/video_games_sale.csv',index_col=0)

print(df.head())

# video_games_sales 데이터셋(video_games_sales.csv)의 출시년도(Year_of_Release) 컬럼을
# 10년단위(ex 1990~1999 : 1990)로 변환하여 새로운 컬럼(year_of_ten)에 추가하고
# 게임이 가장 많이 출시된 년도(10년단위)와 가장 적게 출시된 년도(10년단위)를 각각 구하여라.

df['Year_of_ten'] = df['Year_of_Release'].map(lambda x: x//10*10)

print(df.Year_of_ten.value_counts())
Max = int(df.Year_of_ten.value_counts().index[0])
Min = int(df.Year_of_ten.value_counts().index[-1])

print('Max:', Max, 'Min:', Min)

# 플레이스테이션 플랫폼 시리즈(PS,PS2,PS3,PS4,PSV)중 장르가 Action로 발매된 게임의 총 수는?

print(len(df.loc[(df.Platform.isin(['PS','PS2','PS3','PS4','PSV'])) & (df.Genre=='Action')]))

#게임이 400개 이상 출시된 플랫폼들을 추출하여 각 플랫폼의 User_Score 평균값을 구하여
# 데이터프레임을 만들고 값을 내림차순으로 정리하여 출력하라

over_platform = df.Platform.value_counts()[df.Platform.value_counts()>=400].index
print(over_platform)

answer = df.loc[df.Platform.isin(over_platform)].groupby('Platform')['User_Score'].mean()

answer2 = answer.sort_values(ascending=False).to_frame()

print(answer2)

#게임 이름에 Mario가 들어가는 게임을 3회 개발한 개발자(Developer컬럼)을 구하여라

target = df[df.Name.str.contains('Mario')].Developer.value_counts()

ans = list(target[target==3].index)
print(ans)

#PS2 플랫폼으로 출시된 게임들의 User_Score의 첨도를 구하여라
ans = df[df.Platform=='PS2'].User_Score.kurtosis()
print(ans)


# 각 게임별 NA_Sales,EU_Sales,JP_Sales,Other_Sales 값의 합은 Global_Sales와 동일해야한다.
# 소숫점 2자리 이하의 생략으로 둘의 값의 다른경우가 존재하는데, 이러한 케이스가 몇개 있는지 확인하라

ans = (df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum(axis=1) != df['Global_Sales']).sum()
print(ans)

# User_Count컬럼의 값이 120 이상인 게임들 중에서 User_Score의 값이 9.0이상인 게임의 수를 구하여라

ans = len(df[(df.User_Count >= 120) & (df.User_Score >= 9.0)])
print(ans)

# Global_Sales컬럼의 값들을 robust스케일을 진행하고 40이상인 데이터 수를 구하여라

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler_fit = scaler.fit_transform(df.Global_Sales.values.reshape(-1,1))
ans = len(scaler_fit[scaler_fit > 40])
print(ans)