# 라이브러리
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#데이터 불러오기
df = pd.read_csv('creditcard.csv')

#shape, info, head
print(df.shape)
print(df.info())
print(df.head())

#결측치 처리------------결측치 없음
# print(df.isna().sum())

#이상치 처리-----------너무 많아서 제거 x
# for col in df.columns:
#     q1,q3 = np.percentile(df[col],[25,75])
#     outlier = df[col][(df[col]<q1)|(df[col]>q3)]
#     print(col)
#     print(outlier.shape)

#카테고리 변수 찾기--- 없음
# for col in df.columns:
#     print(col, df[col].nunique())

#왜도----------------높은 변수는 표준화
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
skewed_features = df.apply(lambda x: skew(x))
skewed_features = skewed_features[skewed_features>0.75]
print(skewed_features.index)

scaler =StandardScaler()
df[skewed_features.index[:-2]] = scaler.fit_transform(df[skewed_features.index[:-2]])


#분류하기
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

X = df.drop("Class", axis=1)
y = df['Class']
X_train,X_valid,y_train,y_valid =train_test_split(X,y,test_size =0.3, shuffle= True, random_state= 121)

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svc = SVC()
xgb = XGBClassifier()

models = [dt,rf,svc,xgb]

for model in models:
    model.fit(X_train,y_train)
    pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, pred)
    f1 = f1_score(y_valid, pred)
    recall = recall_score(y_valid, pred)
    cf_matrix = confusion_matrix(y_valid,pred)
    roc = roc_auc_score(y_valid, pred)
    print(model)
    print("acc:" ,acc, "f1:",f1, "recall:",recall,"roc_auc_score:",roc )
    print("confusion_matrix",cf_matrix)
