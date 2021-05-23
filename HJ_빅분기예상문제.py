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

#문제 2------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import skew
import warnings

warnings.filterwarnings(action='ignore')

# 모델 관련 라이브러리
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# 모델들_분류알고리즘
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# 평가지표
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    roc_curve
# 초매개변수 최적화
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_validate
# 앙상블
from sklearn.ensemble import VotingClassifier

# 데이터 불러오기
train_X = pd.read_csv("data/X_train.csv")
train_y = pd.read_csv("data/y_train.csv")
test = pd.read_csv("data/X_test.csv")
train = pd.merge(train_X, train_y, left_on='cust_id', right_on='cust_id', how='inner')
train_test = pd.concat([train, test])

#창보기 설정
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 100)

# --shape, head, info
# print(train.shape, train.head())
print(train_test.info())

# print(test.shape, test.head())
# print(test.info())


# 데이터 전처리
numeric_feats = train_test.dtypes[train_test.dtypes != 'object'].index
obj_feats = train_test.dtypes[train_test.dtypes == 'object'].index

# ---결측치 처리
print(train_test.isna().sum())  # --환불금액 train 2295 test 1611
train_test['환불금액'].fillna(0, inplace=True)

# -- 스케일링
# -----왜도가 0.75 이상인 컬럼 -->로그스케일
skewed_feats = train_test[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 3]
skewed_feats = skewed_feats.index

scaler = StandardScaler()
scaler.fit(train_test[skewed_feats])
train_test[skewed_feats] = scaler.transform(train_test[skewed_feats])

# one-hot encoding
# train_test = pd.get_dummies(train_test, prefix=obj_feats, columns=obj_feats)

prod_values = train['주구매상품'].groupby(train['gender']).value_counts()[1]
prod_keys = train['주구매상품'].groupby(train['gender']).value_counts()[1].index
prod_dict = dict(zip(prod_keys, prod_values))

train_test['주구매상품'] = train_test['주구매상품'].map(prod_dict)
train_test['주구매상품'].fillna(0, inplace= True)
print(train_test.isna().sum())
train_test.drop('주구매지점', axis=1, inplace= True)

# 분류 모델 적용
test_X = train_test.iloc[len(train):, ].drop('gender', axis=1)
X_df = train_test.iloc[:len(train), ].drop('gender', axis=1)
y_df = train_test.iloc[:len(train), ]['gender']

# 훈련데이터와 검증데이터 분리
X_train, X_valid, y_train, y_valid = train_test_split(X_df, y_df, test_size=0.2, random_state=36, shuffle=False)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

# 모델 적용
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier()
# svm = svm()

models = [('DT', dt), ('RF', rf), ('LR', lr)]

# for model in models:
# 	model.fit(X_train, y_train)
# 	pred= model.predict(X_valid)
# 	proba = model.predict_proba(X_valid)
# 	accuracy = accuracy_score(y_valid, pred)
# 	auc = roc_auc_score(y_valid, pred)
# 	f1 = f1_score(y_valid, pred)
# 	recall = recall_score(y_valid, pred)
# 	precision =precision_score(y_valid, pred)
# 	conf_mtx = confusion_matrix(y_valid, pred)
# 	print(model,"-"*50)
# 	print(accuracy, auc, f1, recall, precision)
# 	print(conf_mtx)

# 	#cross_val_score
# 	scores_list = cross_val_score(model, X_train, y_train, cv=5, scoring= 'roc_auc')
# 	print(scores_list)
# 	print(scores_list.mean())


# 초매개변수 최적화 - GridSearch CV (rf, lr)
# ---rf
# my_param = {"n_estimators":[100,300,500],
# 					 "max_depth":[1,3,5]}
# 					 # "min_samples_split":[1,2,3]}

# gcv = GridSearchCV(rf, param_grid = my_param, scoring='roc_auc',
# 									refit =True, cv=5)
# gcv.fit(X_train,y_train)
# pred= gcv.predict(X_valid)
# proba= gcv.predict_proba(X_valid)
# accuracy = accuracy_score(y_valid, pred)
# auc = roc_auc_score(y_valid, pred)
# print(accuracy, auc) #0.6357142857142857 0.5596330275229358

# # -- lr
# my_param = {"penalty":['elasticnet','none'],
# 						"max_iter":[100,120,150,200],
# 					 "l1_ratio":[0,0.25,0.5]}
# gcv = GridSearchCV(lr, param_grid = my_param, scoring='roc_auc',
# 									refit =True, cv=5)
# gcv.fit(X_train,y_train)
# pred= gcv.predict(X_valid)
# proba= gcv.predict_proba(X_valid)
# accuracy = accuracy_score(y_valid, pred)
# auc = roc_auc_score(y_valid, pred)
# print(gcv.best_params_)#{'l1_ratio': 0, 'max_iter': 150, 'penalty': 'none'}
# print(accuracy, auc) #.6271428571428571 0.5512579927717542

# 앙상블
vot = VotingClassifier(models)
# vot.fit(X_train,y_train)
# pred = vot.predict(X_valid)
# accuracy = accuracy_score(y_valid, pred)
# auc = roc_auc_score(y_valid, pred)
# print(accuracy, auc)

# 앙상블-초매개변수 최적화
my_params = {'voting': ['hard', 'soft'],
             'weights': [[2, 1, 1], [1, 1, 1], [1, 2, 1], [1, 1, 2]]}
vot_gcv = GridSearchCV(vot, param_grid=my_params,
                       scoring='roc_auc', refit=True, cv=5)
vot_gcv.fit(X_train, y_train)
pred = vot_gcv.predict(X_valid)
proba = vot_gcv.predict_proba(X_valid)
accuracy = accuracy_score(y_valid, pred)
auc = roc_auc_score(y_valid, proba[:,1])
print(accuracy, auc)
print(vot_gcv.best_params_)

# submission
print(X_df.shape, test.shape)
pred_test = vot_gcv.predict_proba(test_X)

sub_df = pd.DataFrame({'cust_id': test['cust_id'], 'gender': pred_test[:,1]})
print(sub_df.tail())
sub_df.to_csv("0090.csv", index=False)