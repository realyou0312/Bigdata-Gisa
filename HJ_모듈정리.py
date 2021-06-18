##군집 -Kmeans,실루엣 계수
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

#--------모델 적용
#n : 군집 개수(임의 지정)
kmeans = KMeans(n_clusters = n, init = 'k-means++', max_iter=200, random_state=121)
kmeans.fit(X)
X['cluster'] = kmean.labels_
#데이터별 실루엣 계수 구하기
silhouette_coef = silhouetter_samples(X.drop('cluster', axis=1), X['cluster'])
X['silhouette_coef']= silhouette_coef
#각 실루엣 계수의 평균 구하기
silhouette_average = silhouette_score(X.drop('cluster', axis=1), X['cluster'])

##분류 - 모델들, 평가지표, GridSearchCV

#--------모델 불러오기
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomFroestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score

X = train.drop('target',axis=1)
y= train['target']
test_X = test.copy()

X_train,y_train,X_test,y_test = train_test_split(X,y, test_size = 0.1, shuffle =True, random_state = 121)
dt = DecisionTreeClassifier()
svc = SVC(probability = True)
gnb = GaussianNB()
lr = LogisticRegression()
rf = RandomFroestClassifier()
xgb = XGBClassifier()

models = [dt,svc, gnb, lr, rf, xgb]
for model in models:
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  proba = model.predict_proba(X_test) # SVC는 predict_proba 사용하려면 옵션을 걸어야 함. 
  
  #--------target 종류가 2개인 경우
  acc = accuracy_score(y_test, pred) #y_true, y_pred
  f1  = f1_score(y_test, pred)
  recall = recall_score(y_test, pred)
  precision = precision_score(y_test, pred)
  confusion_matrix = confusion_matrix(y_test,pred)
  roc = roc_auc_score(y_test, proba[:,1]) # roc_auc는 proba가 들어가야 함!
  
  #-------target 종류가 3개인 경우
  acc = accuracy_sore(y_test, pred)
  recall = recall_score(y_test, pred, average = 'macro') #macro, micro, weighted, None,samples
  #macre: 각 label에 대한 recall을 구하고 평균을 내는 것
  #micro : 전체 데이터에 대한 recall을 구하는 것(한번에 합산)
  #weighted : 각 label에 대한 recall을 구하고, 정확히 맞춘 사례 비중을 가중치로 잡아서 평균을 내는 것
  #None: 각 label에 대한 recall을 구하는 것
  #samples : 각 instance에 대한 recall을 구하고 평균을 냄. --아직 이해가 되지 않음. 
  precision = precision_score(y_test, pred, average = 'macro')
  f1 = f1_score(y_test, pred, avverage= 'macro')
  roc = roc_auc_score(y_test, proba, average='macro', multi_class='ovr')
  #ovo: one vs one
  #ovr: one vs rest
  
#GridSearchCV -- fold를 사용하기 때문에 전체 데이터를 cv로 나눠서 학습 가능함. 
print(model.get_params()) #parameter출력
my_params = {}
skf = StratiFiedKFold(n_splits= 5, shuffle= False)
kf = KFold(n_splits= 5, shuffle= False)

gcv = GridSearchCV(model, param_gird = my_params, scoring = 'roc_auc' ,refit= True , cv= skf) #kf 
#scoring : 'accuracy', 'f1', 'f1_macro', 'neg_log_loss', 'precision', 'recall', 'roc_auc', 'roc_auc_ovr' --분류
#         'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2', 'neg_mean_squared_log_error'
gcv.fit(X, y)#주의주의
print(gcv.best_params_)
print(gcv.best_score_)

#parameter별로 train,test의 각 split 점수 알아내기
cv_result = pd.DataFrame(gcv.cv_results_)
cv_result.set_index('params')
print(cv_result.sort_values(by='rank_test_score')
     
pred = gcv.predict(test_X)
prboa = gcv.predict_proba(test_X)

#cross_val_score ---GridSearchCV(파라미터 튜닝)할 시간이 없다면, 점수 내기만 하자- scoring은 하나만 가능
cv_score = cross_val_scroe(model, X,y, cv= skf, scoring= 'roc_auc')
print(cv_score)
print(cv_score.mean())
      

#Stacking
from sklearn.ensemble import StackingClassifier



  
  
  
  
  


