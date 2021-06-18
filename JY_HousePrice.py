import pandas as pd
pd.set_option('max_columns', 100)
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df_copy = df.copy()
print(df_copy.head(3))

df_isnull = df.isnull().sum()
print(df_isnull[df_isnull >0].sort_values(ascending=False))

df_isnull = test.isnull().sum()
print(df_isnull[df_isnull >0].sort_values(ascending=False))

print(df['SalePrice'].skew())
print(df['SalePrice'].kurt())

df['SalePrice'] = np.log1p(df['SalePrice'])

df.drop(['Id', 'PoolQC', 'MiscFeature', 'Fence', 'Alley', 'FireplaceQu'], axis=1, inplace=True)
test.drop(['Id', 'PoolQC', 'MiscFeature', 'Fence', 'Alley', 'FireplaceQu'], axis=1, inplace=True)

df.fillna(df.mean(), inplace=True)
test.fillna(df.mean(), inplace=True)
null_columns_count = df.isnull().sum()[df.isnull().sum() > 0]
print(df.dtypes[null_columns_count.index])

print(df.isnull().sum().sort_values(ascending=False))

print(df.isnull().sum().sum())
print(test.isnull().sum().sum())

numerical_feats = df.dtypes[df.dtypes != 'object'].index
print(len(numerical_feats))

categorical_feats = df.dtypes[df.dtypes == 'object'].index
print(len(categorical_feats))


train_test = pd.concat([df,test])
train_test = pd.get_dummies(train_test, prefix = categorical_feats, columns = categorical_feats)


house_df = train_test.iloc[:df.shape[0],:]
test = train_test.iloc[house_df.shape[0]:,:].drop('SalePrice', axis=1)

X = house_df.drop("SalePrice", axis=1)
y= house_df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

ridge = Ridge()
ridge.fit(X_train, y_train)

lasso = Lasso()
lasso.fit(X_train, y_train)

models = [lr, ridge, lasso]

rmses = []
for model in models:
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    rmses.append(rmse)

print(rmses)

from xgboost import XGBRegressor

def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=10)
    grid_model.fit(X, y)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 10 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__, np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_

xgb_params = {'n_estimators':[100]}
xgb_reg = XGBRegressor(n_estimators=500, learning_rate=0.05)
best_xgb = print_best_params(xgb_reg, xgb_params)

xgb_reg.fit(X, y)
xgb_pred = xgb_reg.predict(test)

xgb_pred = np.expm1(xgb_pred)
print(xgb_pred)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.iloc[:, 1] = xgb_pred
# sample_submission.to_csv('수험번호.csv', index=False)