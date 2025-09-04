import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/houseprice/train.csv')
test_df = pd.read_csv('./data/houseprice/test.csv')

train_df = train_df.select_dtypes(include=['number'])

# 결측치 제거 (간단히 처리)
train_df = train_df.dropna()

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df.drop(columns='SalePrice')
y_train = train_df['SalePrice']

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
# alpha == lamda (패널티 가중치 패러미터)

elastic = ElasticNet(alpha=0.1,
                     l1_ratio=0.6)
elastic.fit(X_train, y_train)
elastic.coef_
elastic.intercept_

ridge = Ridge(alpha=0.00001) 
ridge.fit(X_train, y_train)
ridge.coef_
ridge.intercept_

lasso = Lasso(alpha=0.00001) 
lasso.fit(X_train, y_train)
lasso.coef_
lasso.intercept_

from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

alphas = np.linspace(0.0001, 20, 200) # 200개 구간 분할

lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)

lasso_cv.fit(X_train, y_train)

# 최적 alpha 값
best_alpha = lasso_cv.alpha_
print("최적 alpha:::", best_alpha)


lr = LinearRegression
# 모델 학습
lr.fit(X_train, y_train)
lr.coef_
lr.intercept_

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
test_df = test_df.select_dtypes(include=['number'])
test_df = test_df.fillna(train_df.mean())

# 예측
y_pred = lr.predict(test_df)
submit = pd.read_csv('./data/houseprice/sample_submission.csv')
submit["SalePrice"]=y_pred

# CSV로 저장
submit.to_csv('./data/houseprice/baseline.csv', index=False)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X_train,  
                   y_train, 
                   test_size = 0.2,                     
                   random_state = 0, 
                   shuffle = True, 
                   stratify = None 
                   )

train_X, test_X, train_y, test_y = train_test_split(
                   X_train,  
                   y_train, 
                   test_size = 0.3,
                   random_state = 2025, 
                   shuffle = True, 
                   stratify = None 
                   )
# 1) train_df 총 행수 확인
# 2) 벡터가 주어졌을 때 랜덤으로 순서를 섞는 함수 알아낼 것
#     1:n 벡터를 섞어주세요!
# 3) 앞의 80% 인덱스는 train으로, 뒤 20%는 test로 설정
import random
shuffled_X = random.shuffle(X_train)

train_df.iloc[np.array([2, 3]), :]
n=train_df.shape[0]
idx_vec = np.arange(n)
np.random.shuffle(idx_vec)

k1=int(n*0.33)
k2=int(n*0.66)

df1_Xy = train_df.iloc[idx_vec[:k1]]
df2_Xy = train_df.iloc[idx_vec[k1:k2]]
df3_Xy = train_df.iloc[idx_vec[k2:]]

def df_concat(df1, df2):
    import pandas as pd
    result=pd.concat([df1, df2], axis=0)
    return result

df3_Xy
def Xy_split(df_Xy):
    X = df_Xy.drop(columns='SalePrice')
    y = df_Xy['SalePrice']
    return X, y


def cal_rmse(y, y_hat):
    import numpy as np
    result = np.sqrt(np.mean((y-y_hat)**2))
    return result

# valid_set
from sklearn.model_selection import train_test_split

train_X_sub, valid_X, train_y_sub, valid_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=1
)

# 모의고사 세트(validation set) 3개를 활용해서 모델 성능 점수를 계산
lr = LinearRegression()
result = np.array([0, 0, 0])
# 모의고사1
tr_df = df_concat(df1_Xy, df2_Xy)
val_df = df3_Xy

tr_X, tr_y = Xy_split(tr_df)
val_X, val_y = Xy_split(val_df)

#학습
lr.fit(tr_X, tr_y)
#성능평가
val_y_hat = lr.predict(val_X)
result[0] = cal_rmse(val_y, val_y_hat)

# 모의고사2
tr_df = df_concat(df1_Xy, df3_Xy)
val_df = df2_Xy

tr_X, tr_y = Xy_split(tr_df)
val_X, val_y = Xy_split(val_df)

#학습
lr.fit(tr_X, tr_y)
#성능평가
val_y_hat = lr.predict(val_X)
result[1] = cal_rmse(val_y, val_y_hat)

# 모의고사1
tr_df = df_concat(df2_Xy, df3_Xy)
val_df = df1_Xy

tr_X, tr_y = Xy_split(tr_df)
val_X, val_y = Xy_split(val_df)

#학습
lr.fit(tr_X, tr_y)
#성능평가
val_y_hat = lr.predict(val_X)
result[2] = cal_rmse(val_y, val_y_hat)

result.mean()

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)

# RMSE 기준으로 판단해보자

result2 = np.array([0, 0, 0])
# 모의고사1
tr_df = df_concat(df1_Xy, df2_Xy)
val_df = df3_Xy

tr_X, tr_y = Xy_split(tr_df)
val_X, val_y = Xy_split(val_df)

#학습
knn.fit(tr_X, tr_y)
#성능평가
val_y_hat = knn.predict(val_X)
result2[0] = cal_rmse(val_y, val_y_hat)

# 모의고사2
tr_df = df_concat(df1_Xy, df3_Xy)
val_df = df2_Xy

tr_X, tr_y = Xy_split(tr_df)
val_X, val_y = Xy_split(val_df)

#학습
knn.fit(tr_X, tr_y)
#성능평가
val_y_hat = knn.predict(val_X)
result2[1] = cal_rmse(val_y, val_y_hat)

# 모의고사1
tr_df = df_concat(df2_Xy, df3_Xy)
val_df = df1_Xy

tr_X, tr_y = Xy_split(tr_df)
val_X, val_y = Xy_split(val_df)

#학습
knn.fit(tr_X, tr_y)
#성능평가
val_y_hat = knn.predict(val_X)
result2[2] = cal_rmse(val_y, val_y_hat)

result2.mean()