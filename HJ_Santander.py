##기본 라이브러리 불러오기
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

##데이터 불러오기
cust_df= pd.read_csv("./santander_train.csv", encoding ='latin-1')
test_df = pd.read_csv("./santander_test.csv", encoding = 'latin-1')

train_test = pd.concat([cust_df,test_df], axis=0)
##shape, info, head
print('dataset shape:', cust_df.shape)
print('dataset shape:', test_df.shape)
# print(cust_df.info())
# print(test_df.info())
#-------------------------------창보기 설정
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows',200)
# print(cust_df.head(3))

#결측치 처리
train_test["TARGET"] = train_test['TARGET'].fillna(10)
print(train_test.isna().sum().sum())

## TARGET 분포 -- SMOTE사용은 불가
print(cust_df['TARGET'].value_counts())
unsatisfied_cnt = cust_df[cust_df['TARGET']==1].TARGET.count()
total_cnt = cust_df.TARGET.count()
print('unsatisfied 비율은 {0:.2f}'.format((unsatisfied_cnt/ total_cnt)))

## .describe()
# print(train_test.describe())

print(train_test['num_var6_0'].value_counts().index[1])

## 이상치 처리 및 Scaling
multi_cols = []
for col in train_test.columns:
    if train_test[col].nunique() ==1:
        train_test.drop(col, axis=1, inplace= True)
    elif train_test[col].nunique() ==2:
        print(col)
        print(train_test[col].value_counts().index[1])
        train_test[col].replace(train_test[col].value_counts().index[1] , 1, inplace= True)
    else:
        print(train_test[col].nunique())
        multi_cols.append(col)

#왜도 측정
from scipy.stats import skew
#----skew
skewed_feats = train_test[multi_cols].apply(lambda x:skew(x))
skewed_feats = skewed_feats[skewed_feats>0.75]
#---------로그스케일
#----음수값 가지는 컬럼은 제외하기
is_neg = train_test[skewed_feats.index][train_test[skewed_feats.index]<0].any()
skewed_feats = is_neg[is_neg.values== False].index
train_test[skewed_feats] = np.log1p(train_test[skewed_feats])

#--------정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_test[multi_cols][train_test[multi_cols].columns.difference(skewed_feats)] = scaler.fit_transform(train_test[multi_cols][train_test[multi_cols].columns.difference(skewed_feats)])


##ID DROP
train_test.drop('ID', axis=1, inplace= True)

##이상치 처리
# def outliers_iqr(data):
#     q1, q3 = np.percentile(data, [25,75])
#     iqr = q3- q1
#     lower_bound = q1 - (iqr* 1.5)
#     upper_bound = q3 + (iqr* 1.5)
#
#     return np.where((data> upper_bound)|(data<lower_bound))

## train_test 분리
from sklearn.model_selection import train_test_split
train = train_test.iloc[:len(cust_df),:]
test = train_test.iloc[len(cust_df):,:].drop("TARGET", axis=1)
X_features = train.drop("TARGET",axis=1)
y_target = train["TARGET"]

X_train, X_valid, y_train, y_valid = train_test_split(X_features, y_target, test_size= 0.3, random_state= 36)
# #-----------------------------target 분포 비율 확인- 비슷함
# print(y_train.value_counts(normalize= True))
# print(y_valid.value_counts(normalize= True))

## 모델 적용
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix #----정확도, roc-auc, confusion 매트릭스
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


rf = RandomForestClassifier()
xgb = XGBClassifier(n_estimators = 500, random_state= 36, eval_metric = 'auc')
lr = LogisticRegression()
models = [rf, xgb, lr]

for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    proba =model.predict_proba(X_valid)
    accuracy = accuracy_score(y_valid, pred)
    roc_auc = roc_auc_score(y_valid, pred)
    confusion = confusion_matrix(y_valid, pred)
    print(model)
    print(accuracy,roc_auc)
    print(confusion)

#xgb: 0.7859516421462801

#Submission
test_pred = xgb.predict(test)
sub_df =pd.DataFrame({"ID":test_df.ID, "TARGET":test_pred})
sub_df.to_csv("0531.csv")