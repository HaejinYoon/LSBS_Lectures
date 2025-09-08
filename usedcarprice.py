import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/usedcarprice/train.csv')
test_df = pd.read_csv('./data/usedcarprice/test.csv')

train_df.info()

train_df[(train_df['brand'] == 'Lincoln') & (train_df['model'] == 'Town Car Signature')].sort_values('model_year', ascending=False)

# 칼럼 선택
num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop("id")

cat_columns = train_df.select_dtypes(include=['object']).columns

from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')

train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])

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
y_train = np.log1p(train_df['price'])

from sklearn.linear_model import ElasticNet
# alpha == lamda (패널티 가중치 패러미터)

elastic = ElasticNet()
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

elastic_search.fit(X_train, y_train)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(elastic_search.cv_results_))

# best prameter
print(elastic_search.best_params_)

elastic_search.fit(X_train, y_train)

test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])

test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])

test_df_all = pd.concat([test_df_num, test_df_cat], axis = 1)

# 예측
#y_pred = elastic.predict(test_df)
y_pred = elastic_search.predict(test_df_all)
submit = pd.read_csv('./data/usedcarprice/sample_submission.csv')
submit["price"]=np.expm1(y_pred)

# CSV로 저장
submit.to_csv('./data/usedcarprice/elasticnet_grid.csv', index=False)