import pandas as pd
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()

iris_data = iris.data
iris_label = iris.target
print('iris target값:', iris_label)
print('iris target명:', iris.target_names)


iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head(3))


X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf2 = RandomForestClassifier(random_state=42)

clf.fit(X_train, y_train)
clf2.fit(X_train, y_train)


pred = clf.predict(X_test)
pred2 = clf2.predict(X_test)

print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))

print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred2)))
