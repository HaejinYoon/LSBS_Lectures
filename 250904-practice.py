import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./data/problem1.csv')

df.info()
df.describe()

df['Chol'] = df['Chol'].fillna(df['Chol'].mean(numeric_only=True))

X_train = df.drop(columns='DBP')
y_train = df['DBP']

train_X, test_X, train_y, test_y = train_test_split(
                   X_train,
                   y_train,
                   test_size = 0.3,
                   random_state = 2025, 
                   shuffle = True, 
                   stratify = None 
                   )

train_ratio = len(train_X) / len(df)
test_ratio = len(test_X) / len(df)
print(f'훈련 데이터 비율: {train_ratio:.1%}')
print(f'테스트 데이터 비율: {test_ratio:.1%}')

train_X
train_y

test_X
test_y

# ============================================
# LinearRegression
# ============================================
lr = LinearRegression()

lr.fit(train_X, train_y)
lr.coef_
lr.intercept_

y_pred_lr = lr.predict(test_X)

def cal_rmse(y, y_hat):
    import numpy as np
    result = np.sqrt(np.mean((y-y_hat)**2))
    return result

rmse_Lr = cal_rmse(test_y, y_pred_lr) # 10.189505000668806



# ============================================
# Elastic
# ============================================

from sklearn.linear_model import ElasticNet
# alpha == lamda (패널티 가중치 패러미터)

elasticnet = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}
# 파라미터 확인 
ElasticNet().get_params()

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)

# 그리드서치
elastic_search = GridSearchCV(estimator=elasticnet, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

elastic_search.fit(train_X, train_y)

# # 그리드서치 파라미터 성능 확인
# print(pd.DataFrame(elastic_search.cv_results_))

# # best prameter
print(elastic_search.best_params_)

y_pred_el = elastic_search.predict(test_X)

def cal_rmse(y, y_hat):
    import numpy as np
    result = np.sqrt(np.mean((y-y_hat)**2))
    return result

rmse_Ela = cal_rmse(test_y, y_pred_el) # 10.186990376944342

# ============================================
# KNN
# ============================================

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()

knn_params = {'n_neighbors' : np.arange(1, 50)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)

# 그리드서치
knn_search = GridSearchCV(estimator=knn,
                              param_grid=knn_params,
                              cv = cv, 
                              scoring='neg_mean_squared_error')

knn_search.fit(train_X, train_y)
knn_search.best_params_

y_pred_knn = knn_search.predict(test_X)
rmse_Knn = cal_rmse(test_y, y_pred_knn) # 10.267946586805296

# ============================================
# lasso
# ============================================
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.00001) 
lasso.fit(train_X, train_y)
y_pred_la_base = lasso.predict(test_X)
cal_rmse(test_y, y_pred_la_base) # 10.189504142177716
lasso.coef_
lasso.intercept_

from sklearn.linear_model import LassoCV
alphas = np.linspace(0.0001, 20, 2000) # 200개 구간 분할

lasso_cv = LassoCV(alphas=alphas, cv=10, random_state=2025)
lasso_cv.fit(train_X, train_y)
lasso_cv.alpha_
y_pred_la = lasso_cv.predict(test_X)
rmse_Lasso = cal_rmse(test_y, y_pred_la) # 10.186997500904468


rmse_Lr
rmse_Ela
rmse_Knn
rmse_Lasso

# 딕셔너리 → DataFrame
rmse_dict = {
    "LinearRegression": rmse_Lr,
    "ElasticNet": rmse_Ela,
    "KNN": rmse_Knn,
    "Lasso": rmse_Lasso
}

rmse_df = pd.DataFrame(list(rmse_dict.items()), columns=["Model", "RMSE"])

# RMSE 기준 오름차순 정렬
rmse_df_sorted = rmse_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
rmse_df_sorted

import matplotlib.pyplot as plt

df.hist();
plt.tight_layout();
plt.show();