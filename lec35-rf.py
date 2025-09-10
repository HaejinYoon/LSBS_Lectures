import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/houseprice/train.csv')
test_df = pd.read_csv('./data/houseprice/test.csv')

#train_df = train_df.select_dtypes(include=['number'])

# 칼럼 선택
num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop("SalePrice")
cat_columns = train_df.select_dtypes(include=['object']).columns

# 결측치 채우기 (간단히 처리)
# train_df = train_df.dropna()
from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')

train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
freq_impute.statistics_

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
onehot = OneHotEncoder(handle_unknown='ignore', 
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")

train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])

train_df_all = pd.concat([train_df_num, train_df_cat], axis = 1)

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df_all
y_train = np.log1p(train_df['SalePrice'])

from sklearn.ensemble import RandomForestRegressor
# alpha == lamda (패널티 가중치 패러미터)

rf = RandomForestRegressor(oob_score=True)
rf_grid = {
    "n_estimators": [300],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [0.3, 0.5, 0.7],
    "min_samples_split": [2, 5, 10, 20],
    "max_leaf_nodes": [None, 64, 128, 256, 512],
}


# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)

# 그리드서치
rf_search = GridSearchCV(estimator=rf, 
                              param_grid=rf_grid, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

rf_search.fit(X_train, y_train)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(rf_search.cv_results_))

# best prameter
print(rf_search.best_params_)

# 교차검증 best score 
print(-rf_search.best_score_)

# elastic = ElasticNet(alpha=0.1, 
#                      l1_ratio=0.75)
# elastic.fit(X_train, y_train)
# 최종 예측 

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
# test_df = test_df.select_dtypes(include=['number'])
# test_df = test_df.fillna(train_df.mean())

test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])

test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])

test_df_all = pd.concat([test_df_num, test_df_cat], axis = 1)

# 예측
#y_pred = elastic.predict(test_df)
y_pred = rf_search.predict(test_df_all)
submit = pd.read_csv('./data/houseprice/sample_submission.csv')
submit["SalePrice"]=np.expm1(y_pred)

# CSV로 저장
submit.to_csv('./data/houseprice/rf_search.csv', index=False)

