## 기본 라이브러리 불러오기
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")

##데이터 불러오기
bike_df = pd.read_csv("./bike_train.csv")
test = pd.read_csv("./bike_test.csv")
train_test = pd.concat([bike_df,test], axis=0)
train_test = train_test.fillna(0)

##창보기 설정
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)

##shape, head, info
print(bike_df.shape)
print(bike_df.head())
print(bike_df.info())

##타입변환 object-> datetime
train_test['datetime'] = pd.to_datetime(train_test['datetime'])

##년,월,일,시간 추출
train_test['year'] = train_test['datetime'].apply(lambda x: x.year)
train_test['month'] = train_test['datetime'].apply(lambda x:x.month)
train_test['day'] = train_test['datetime'].apply(lambda x:x.day)
train_test['hour'] =train_test['datetime'].apply(lambda x:x.hour)

##datetime, casual, registered 삭제(casual+registered = count라 상관도가 높아 예측 저해할 우려가 있음.)
drop_columns = ['datetime','casual','registered']
train_test.drop(drop_columns, axis=1, inplace= True)
print(train_test)
## Scailing
#로그 스케일
# 왜도 측정
from scipy.stats import skew
skewed_feats = train_test.apply(lambda x:skew(x))
skewed_feats = skewed_feats[skewed_feats>0.75]
# skewed_feats = skewed_feats.index
# 왜도가 높은 변수 --> 로그 변환
train_test[skewed_feats.index] =  np.log1p(train_test[skewed_feats.index])
# skewed_feats = skewed_feats.drop("count")
# test[skewed_feats.index] =  np.log1p(test[skewed_feats.index])


# standard_scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_test[["temp","atemp","humidity"]] = scaler.fit_transform(train_test[["temp","atemp","humidity"]])
# test[["temp","atemp","humidity"]] = scaler.fit_transform(test[["temp","atemp","humidity"]])


##Encoding- get_dummies
time_col =['year','month','day','hour']
train_test=  pd.get_dummies(train_test, prefix =time_col, columns = time_col)
# test=  pd.get_dummies(test, prefix =time_col, columns = time_col)

##모델 적용
# train_test 분할
from sklearn.model_selection import train_test_split
train =train_test.iloc[:len(bike_df),:]
test_features = train_test.iloc[len(bike_df):,:].drop("count", axis=1)

y_target = train['count']
X_features= train.drop(['count'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size =0.3, random_state =36)#순서 유의하기

#rmse, rmsle 계산
from sklearn.metrics import mean_squared_error, mean_absolute_error
def rmse(y, pred):
    return np.sqrt(mean_squared_error(y,pred))

def rmsle(y, pred): # np.sqrt(mean_squared_log_error)는 log1p대신 log를 사용하기 때문에 오버플로나 언더플로를 발생하기 쉬움
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y- log_pred)**2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# #model1. linear Regression --점수 bad
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# lr = LinearRegression()
# #규제
# ridge =Ridge(alpha=10)
# lasso =Lasso(alpha=0.01)
#
# lr.fit(X_train, y_train)
# pred = lr.predict(X_test)
# #-----------------------로그변환 --> 원래대로
# pred= np.expm1(pred)
# y_test =np.expm1(y_test)
# print(rmse(y_test, pred))
# print(rmsle(y_test,pred))
# # coef = pd.Series(lr.coef_,X_features.columns)
# # print(abs(coef).sort_values(ascending= False)[:10])
#
# for model in [ridge,lasso]:
#     model.fit(X_train,y_train)
#     pred = model.predict(X_test)
#     pred = np.expm1(pred)
#     # y_test =np.expm1(y_test)
#     print(model,":")
#     print("rmse:",rmse(y_test,pred))
#     print("rmlse:", rmsle(y_test,pred))
#


#model2 .RandomForestRegressor, GradientBoostRegressor, XGBRegressor, LGBMRegrssor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
xgb = XGBRegressor(n_estimators = 500)
lgbm = LGBMRegressor(n_estimators = 500)
#
# models = [rf,gbr,xgb, lgbm]
#
# y_test = np.expm1(y_test)
# for model in models:
#     model.fit(X_train,y_train)
#     pred = model.predict(X_test)
#     pred = np.expm1(pred)
#
#     print(model,":")
#     print("rmse:",rmse(y_test,pred))
#     print("rmlse:", rmsle(y_test,pred))

#앙상블(voting)
voting_models = [('RF',rf),('XGB',xgb),('LGBM',lgbm)]
from sklearn.ensemble import VotingRegressor
vot = VotingRegressor(voting_models)
vot.fit(X_train,y_train)
pred = vot.predict(X_test)
pred = np.expm1(pred)
y_test = np.expm1(y_test)
print("rmse:",rmse(y_test,pred))
print("rmlse:", rmsle(y_test,pred))


#초매개변수 최적화
# from sklearn.model_selection import GridSearchCV, cross_val_score
# cross_val_score
# scores_list = cross_val_score(rf, X_train, y_train, cv=5, scoring= "neg_mean_squared_log_error")
# print(scores_list.mean())



#submission
bike_test = pd.read_csv("./bike_test.csv")
test_pred = vot.predict(test_features)
test_pred = np.expm1(test_pred)
sub_df= pd.DataFrame({"datetime":test.datetime,"count":test_pred})
sub_df.set_index('datetime', drop=True, inplace= True)
print(sub_df.head())
sub_df.to_csv("0530.csv")
# sub_df= pd.read_csv("sampleSubmission.csv")
# print(sub_df.head())