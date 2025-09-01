import pandas as pd
from sklearn.linear_model import LinearRegression

train_df = pd.read_csv("./data/houseprice/train.csv")
test_df = pd.read_csv("./data/houseprice/test.csv")
submit_csv = pd.read_csv("./data/houseprice/sample_submission.csv")

train_df.info()
test_df.info()
train_df["SalePrice"]
#test_df["SalePrice"]

train_df.head()
test_df.head()

train_df = train_df.select_dtypes(include=['int64', 'float64'])

# drop na 말고 다른 건 없나
train_df = train_df.dropna()

model = LinearRegression()
# 의미없는 변수들을 뺴면 어떨까?
# 
# x와 y 설정
train_X = train_df.drop(["Id", "SalePrice"], axis=1)
train_y = train_df["SalePrice"]

# 모델 학습
model.fit(train_X, train_y)
model.coef_
model.intercept_
# import numpy as np
# model.intercept_ + np.sum(model.coef_ * test_df.iloc[0, :])

test_df = test_df.select_dtypes(include=['int64', 'float64'])

# 모든 NA 값을 0으로 채움
# NA말고 다른 값으로 채울 수 있는 칼럼없어?
test_df = test_df.fillna(0)

result = model.predict(test_df)
result

# train_df의 수치형 변수를 선택해서 train_df 업데이트
# 모든 변수 사용해서 선형회귀분석 fit(계수찾기)
# test_df의 정보를 사용해서 집값 예측
# 예측한 값을 사용해서 submit_csv의 집값 채우기

submit_csv["SalePrice"] = result
submit_csv

submit_csv.to_csv("./data/houseprice/output.csv", index=False, encoding='utf-8-sig')

# =======================================================================================
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')

y = dat.grade
X = dat.drop(['grade'], axis = 1)

train_X, test_X, train_y, test_y = train_test_split(
                   X,  
                   y, 
                   test_size = 0.2,
                   random_state = 0,
                   shuffle = True,
                   stratify = None
                   )
train_X.value_counts("school")
train_X.value_counts("school")

import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=2)
train_y.hist(ax=axs[0], color='blue', alpha=0.7)
axs[0].set_title('histogram of train y')
test_y.hist(ax=axs[1], color='red', alpha=0.7)
axs[1].set_title('histogram of test y')
plt.tight_layout(); 
plt.show();

train_X, test_X, train_y, test_y = train_test_split(
                    X, 
                    y, 
                    test_size = 0.2,
                    stratify = X['school'], 
                    random_state = 0)

train_X.value_counts("school")
train_X.value_counts("school")
train_y.value_counts("school")
test_y.value_counts("school")

fig, axs = plt.subplots(nrows=1, ncols=2)
train_y.hist(ax=axs[0], color='blue', alpha=0.7)
axs[0].set_title('histogram of train y')
test_y.hist(ax=axs[1], color='red', alpha=0.7)
axs[1].set_title('histogram of test y')
plt.tight_layout(); 
plt.show();

from sklearn.impute import SimpleImputer

y = dat.grade
X = dat.drop(['grade'], axis = 1)
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X,  
                   y, 
                   test_size = 0.2,                     
                   random_state = 0, 
                   shuffle = True, 
                   stratify = None 
                   )

train_X3 = train_X.copy()
test_X3 = test_X.copy()
imputer_mode = SimpleImputer(strategy = 'most_frequent')
train_X3['goout'] = imputer_mode.fit_transform(train_X3[['goout']])
test_X3['goout'] = imputer_mode.transform(test_X3[['goout']])
print('학습 데이터 goout 변수 결측치 확인 :', train_X3['goout'].isna().sum())


train_X3['goout'].isna().sum()

from sklearn.impute import KNNImputer
train_X5 = train_X.copy()
test_X5 = test_X.copy()
train_X5_num = train_X5.select_dtypes('number')
test_X5_num = test_X5.select_dtypes('number')
train_X5_cat = train_X5.select_dtypes('object')
test_X5_cat = test_X5.select_dtypes('object')

print('학습 데이터 goout 변수 결측치 확인 :', train_X5['goout'].isna().sum())

knnimputer = KNNImputer(n_neighbors = 5)
train_X5_num_imputed = knnimputer.fit_transform(train_X5_num)
test_X5_num_imputed = knnimputer.transform(test_X5_num)
                       
train_X5_num_imputed = pd.DataFrame(train_X5_num_imputed, 
                                    columns=train_X5_num.columns, 
                                    index = train_X5.index)
test_X5_num_imputed = pd.DataFrame(test_X5_num_imputed, 
                                   columns=test_X5_num.columns, 
                                   index = test_X5.index)
train_X5 = pd.concat([train_X5_cat, train_X5_num_imputed], axis = 1)
test_X5 = pd.concat([test_X5_cat, test_X5_num_imputed], axis = 1)
print('학습 데이터 goout 변수 결측치 확인 :', train_X5['goout'].isna().sum())

# ===========================================
submit_csv = pd.read_csv("./data/houseprice/sample_submission.csv")


train_df = pd.read_csv("./data/houseprice/train.csv")
test_df = pd.read_csv("./data/houseprice/test.csv")

train_df = train_df.select_dtypes(include=['int64','float64'])
test_df = test_df.select_dtypes(include=['int64','float64'])
# 1

y= train_df['SalePrice']
X= train_df.drop(columns=['SalePrice'])




train_X, valid_X, train_y, valid_y = train_test_split(
                   X,  
                   y, 
                  test_size = 0.3,
                   random_state = 0,
                   shuffle = True,
                   stratify = None
                   )


# 2 
train_df.isna().sum(axis=0)

from sklearn.impute import KNNImputer

train_X5 = train_X.copy()
valid_X5 = valid_X.copy()
test_X5 = test_df.copy()
# train_X5_num = train_X5.select_dtypes('number')
# valid_X5_num = valid_X5.select_dtypes('number')

knnimputer = KNNImputer(n_neighbors = 5)
train_X5_num_imputed = knnimputer.fit_transform(train_X5)
valid_X5_num_imputed = knnimputer.transform(valid_X5)
test_X5_num_imputed = knnimputer.transform(test_X5)
                       
train_X5 = pd.DataFrame(train_X5_num_imputed, 
                                    columns=train_X5.columns, 
                                    index = train_X5.index)

valid_X5 = pd.DataFrame(valid_X5_num_imputed, 
                                   columns=valid_X5.columns, 
                                   index = valid_X5.index)

test_X5 = pd.DataFrame(test_X5_num_imputed, 
                                   columns=test_X5.columns, 
                                   index = test_X5.index)



print('학습 데이터 goout 변수 결측치 확인 :', train_X5['LotFrontage'].isna().sum())

# 3 모델적합
model = LinearRegression()
model.fit(train_X5,train_y)
model.predict(valid_X5)

from sklearn.metrics import mean_squared_error

# 예측
y_pred = model.predict(valid_X5)

# RMSE 계산

rmse_manual = np.sqrt(((valid_y - y_pred) ** 2).mean())
print("RMSE (manual):", rmse_manual)


import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
import warnings
np.warnings = warnings

bike_data = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/bike_train.csv")

import matplotlib.pyplot as plt
bike_data['count'].hist();
plt.show();

box_tr = PowerTransformer(method = 'box-cox')
bike_data['count_boxcox'] = box_tr.fit_transform(
    bike_data[['count']])
print('lambda : ', box_tr.lambdas_)

# =====================
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

imputer = SimpleImputer(strategy="mean")
stdscaler = StandardScaler()

cat_columns = train_X.select_dtypes('object').columns
num_columns = train_X.select_dtypes('number').columns

mc_transformer = make_column_transformer(
    (imputer, num_columns),
    (stdscaler, num_columns),
    reainder="passthrough"
).set_output(transform="pandas")