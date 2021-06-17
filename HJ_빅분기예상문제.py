#문제 1-----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#데이터 불러오기
data = pd.read_csv('data/mtcars.csv')
data.set_index('Unnamed: 0', inplace= True)
#스케일링
scaler = MinMaxScaler()
df_scaler = scaler.fit_transform(data)
sdf = pd.DataFrame(df_scaler, columns= data.columns, index= data.index)
print(len(sdf['qsec'][sdf['qsec']>0.5]))

# 문제 2------------------------------------------------------------------------------------------------------------------
# 출력을 원하실 경우 print() 활용
# 예) print(df.head())

# 답안 제출 예시
# 수험번호.csv 생성
# DataFrame.to_csv("0000.csv", index=False)

import pandas as pd
import numpy as np

X = pd.read_csv("data/X_train.csv", encoding ='euc-kr')
y = pd.read_csv("data/y_train.csv", encoding ='euc-kr')
test = pd.read_csv("data/X_test.csv", encoding ='euc-kr')
train = X.merge(y, how="outer", on="cust_id")

# shape, head, info
print(train.shape, test.shape)
print(train.info())
print(test.info())

##결측치 -- 환불금액


# print(train.isna().sum())
# print(test.isna().sum())
numeric_feature = train.dtypes[train.dtypes != "object"].index
obj_feature = train.dtypes[train.dtypes == "object"].index

# print(train[numeric_feature].corrwith(train['환불금액']).sort_values())
# -- 환불금액은 총구매액, 최대구매액과 연관성이 높다
# print(train['총구매액'].corr(train['최대구매액']))
# -- 총구매액과 최대구맥의 상관계수는 0.7

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
train_index = train.dropna().index
rf_reg.fit(train.loc[train_index, ['총구매액', '최대구매액', '내점일수']], train.loc[train_index, '환불금액'])

for df in [train, test]:
    train_index = df.dropna().index
    null_index = [i for i in df.index if i not in train_index]

    pred = rf_reg.predict(df.loc[null_index, ['총구매액', '최대구매액', '내점일수']])
    df.loc[null_index, '환불금액'] = pred

##이상치 파악하기
# outlier_idx = []
# for col in ['총구매액','최대구매액']:
# 	q1,q3 = np.percentile(train[col], [25,75])
# 	iqr = q3 - q1
# 	lower_bound = q1- (iqr*1.5)
# 	upper_bound = q3+ (iqr*1.5)

# 	outlier = train[col][(train[col]<lower_bound)|(train[col]>upper_bound)].index
# 	outlier_idx.extend(outlier)
# 	# print(col, len(outlier_idx))
# outlier_idx = set(outlier_idx)
# print(len(outlier_idx))


###왜도 파악하기 -> 높은 것은 scaling
from scipy.stats import skew

skewed_feature = train[numeric_feature].apply(lambda x: skew(x))
skewed_feature = skewed_feature[abs(skewed_feature) > 0.75].index
# print(train[skewed_feature][train[skewed_feature]<0].sum())	총구매액과 최대구매액에 음수값이 있음

# 총구매액 minus면 모두 여자
# minus = train[['총구매액','gender']][train['총구매액']<0]
# print(minus)
# minus = train[['최대구매액','gender']][train['최대구매액']<0]
# print(minus)
train['minus'] = train['총구매액'].apply(lambda x: 5 if x < 0 else 0)
test['minus'] = test['총구매액'].apply(lambda x: 5 if x < 0 else 0)

for df in [train, test]:
    df.loc[df['총구매액'] < 0, '총구매액'] = 1
    df.loc[df['최대구매액'] < 0, '최대구매액'] = 1

train[skewed_feature] = np.log1p(train[skewed_feature])
test[skewed_feature] = np.log1p(test[skewed_feature])

### Encoding
# train_test = pd.concat([train,test], axis=0)
# train_test = pd.get_dummies(train_test)
# X = train_test.iloc[:len(train),:].drop('gender', axis=1)
# y = train_test.iloc[:len(train),:]['gender']
# test = train_test.iloc[len(train):,:].drop('gender', axis=1)


for col in obj_feature:
    col_map = train.loc[train['gender'] == 1, col].value_counts().to_dict()
    train[col] = train[col].map(col_map)
    test[col] = test[col].map(col_map)
    train[col].fillna(0, inplace=True)
    test[col].fillna(0, inplace=True)

X = train.drop(['gender', 'cust_id'], axis=1)
y = train['gender']
test_X = test.drop('cust_id', axis=1)

#cluster
from sklearn.cluster import KMeans
# case_one = y.value_counts()[0]
# case_two = y.value_counts()[1]
# class_diff = abs(case_one-case_two)
# prev= 10
# for i in range(10):
#     clusters = len(y) // (i+1)
#     after = abs(clusters- class_diff)/ class_diff


# 모델 학습
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=121)

svm = SVC(probability=True)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()
lr = LogisticRegression()
gnb = GaussianNB()  # 0.64

# cross-val-score
# models = [svm,dt,xgb,rf, lr,gnb]
# for model in models:
# 	score = cross_val_score(model, X,y, scoring = 'roc_auc',cv = skfold)
# 	print(model, score.mean())


models = [gnb, lr, xgb]
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, proba[:, 1])
    matrix = confusion_matrix(y_test, pred)
    print(model, "---------------------")
    print(acc, roc)
    print(matrix)

# try:
# 	print(model.feature_importances_)
# except:
# 	continue


# Stacking
estimators = {('gnb', gnb), ('DT', dt), ("RF", rf)}
Stacking = StackingClassifier(estimators=estimators, final_estimator=lr)  # 0.62
Stacking.fit(X_train, y_train)
pred = Stacking.predict(X_test)
proba = Stacking.predict_proba(X_test)
acc = accuracy_score(y_test, pred)
roc = roc_auc_score(y_test, proba[:, 1])
matrix = confusion_matrix(y_test, pred)
print(acc, roc)
print(matrix)



# 초매개변수 최적화 - GridSearch CV (rf, lr)
# ---rf
my_param = {"n_estimators":[100,300,500],
					 "max_depth":[1,3,5]}
					 # "min_samples_split":[1,2,3]}

gcv_rf = GridSearchCV(rf, param_grid = my_param, scoring='roc_auc',
									refit =True, cv=5)
gcv_rf.fit(X_train,y_train)
pred= gcv_rf.predict(X_test)
proba= gcv_rf.predict_proba(X_test)
accuracy = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba[:,1])
print(accuracy, auc) #0.6657142857142857 0.6503026463963963

# -- lr
my_param = {"penalty":['elasticnet','none'],
						"max_iter":[100,120,150,200],
					 "l1_ratio":[0,0.25,0.5]}
gcv_lr = GridSearchCV(lr, param_grid = my_param, scoring='roc_auc',
									refit =True, cv=5)
gcv_lr.fit(X_train,y_train)
pred= gcv_lr.predict(X_test)
proba= gcv_lr.predict_proba(X_test)
accuracy = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba[:,1])
print(gcv_lr.best_params_)#{'l1_ratio': 0, 'max_iter': 200, 'penalty': 'none'}
print(accuracy, auc) #0.64 0.6479448198198198

gcv_gnb = GridSearchCV(gnb,cv=5)
gcv_gnb.fit(X_train,y_train)
pred= gcv_gnb.predict(X_test)
proba= gcv_gnb.predict_proba(X_test)
accuracy = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba[:,1])
print(gcv_gnb.best_params_)
print(accuracy, auc)

# test
sub = gcv_rf.predict(test_X)
sub_df = pd.DataFrame({'cust_id': test['cust_id'], 'gender': sub})
print(sub_df.head())
sub_df.to_csv("002001526.csv", index=False)
