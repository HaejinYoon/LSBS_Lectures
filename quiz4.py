import pandas as pd
import numpy as np

#======================================================================================================
# 회귀모델 학습 (Petal_Length ~ Petal_Width)
#======================================================================================================
import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.ols("Petal_Length ~ Petal_Width + C(Species)", data=iris).fit()

#======================================================================================================
# 피어슨 상관계수와 p-value
#======================================================================================================
from scipy.stats import pearsonr
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 13])

import scipy.stats as stats
corr_coeff, p_value = stats.pearsonr(x, y)
print(corr_coeff)
print(p_value)


#======================================================================================================
# One-hot-encoding => drop_first=False
# Dummy-coding => drop_first=True
#======================================================================================================
from palmerpenguins import load_penguins
df = load_penguins()

penguins=df.dropna()
penguins_dummies = pd.get_dummies(
    penguins, 
    columns=['species'],
    drop_first=True
    )

#======================================================================================================
# ANOVA
#======================================================================================================
 # 일원 분산분석(One-way ANOVA) 수행
from scipy.stats import f_oneway
f_statistic, p_value = f_oneway(lavender, rosemary, peppermint)

#======================================================================================================
# 사후분석: Tukey’s HSD로 집단 간 차이 확인
#======================================================================================================
from statsmodels.stats.multicomp import pairwise_tukeyhsd
odors = ['Lavender', 'Rosemary', 'Peppermint']
minutes_lavender = [10, 12, 11, 9, 8, 12, 11, 10, 10, 11]
minutes_rosemary = [14, 15, 13, 16, 14, 15, 14, 13, 14, 16]
minutes_peppermint = [18, 17, 18, 16, 17, 19, 18, 17, 18, 19]
anova_data = pd.DataFrame({
'Odor': np.repeat(odors, 10),
'Minutes': minutes_lavender + minutes_rosemary + minutes_peppermint
})

tukey = pairwise_tukeyhsd(
endog=anova_data['Minutes'],
groups=anova_data['Odor'],
alpha=0.05)
print(tukey)

#======================================================================================================
# RMSE 계산
#======================================================================================================
rmse_manual = np.sqrt(((valid_y - y_pred) ** 2).mean())

#======================================================================================================
# [해설]선형회귀분석
#======================================================================================================

#======================================================================================================
# ROC 커브 ===> 이진 분류지표 이해하기
#======================================================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# y_true: 실제 값 (0/1), y_score: 예측 확률
# 예시 데이터
y_true = np.array([0,0,1,1,0,1,0,1,0,1])
y_score = np.array([0.1,0.4,0.35,0.8,0.2,0.85,0.3,0.7,0.6,0.9])

# ROC 커브 계산
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Youden’s J 통계량 최적 cut-off
J = tpr - fpr
ix = np.argmax(J)
best_threshold = thresholds[ix]
print(f"최적 cut-off (Youden J 기준): {best_threshold:.3f}")

# ROC 커브 그리기
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, marker='o', label="ROC curve")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.scatter(fpr[ix], tpr[ix], color="red", label=f"Best cutoff={best_threshold:.2f}")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title(f"ROC Curve (AUC={roc_auc_score(y_true, y_score):.3f})")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

#======================================================================================================
# 로지스틱 회귀 오즈비 및 이탈 확률 계산 ===> 파이썬에서 로지스틱 회귀분석 수행하기
#======================================================================================================

import pandas as pd
import joblib as jb

lasso = jb.load("./data/quiz5/lasso_model.pkl")

df1 = pd.read_csv("./data/quiz5/datasetSalaries.csv")

df12 = pd.read_csv("./data/quiz5/problem2.csv")

df = pd.read_csv("./data/quiz5/problem4_33.csv")

df16 = pd.read_csv("./data/quiz5/problem15.csv")

df = pd.read_csv("./data/quiz5/problem19_test.csv")

df19 = pd.read_csv("./data/quiz5/problem19.csv")


from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
df1
#1
group1 = df1[df1['sex'] == 'Male']["salary"]
group2 = df1[df1['sex'] == 'Female']["salary"]
t_stat, p_val = ttest_ind(group1, group2, equal_var=False)

#2
df1['rank'].unique()
group1 = df1[df1['rank'] == 'Professor']["salary"]
group2 = df1[df1['rank'] == 'Assistant Professor']["salary"]
group3 = df1[df1['rank'] == 'Associate Professor']["salary"]
from scipy.stats import f_oneway
f_statistic, p_value = f_oneway(group1, group2, group3)

#3
from scipy.stats import shapiro
stat_m1, p_m = shapiro(group1)
stat_m2, p_m = shapiro(group2)
stat_m3, p_m = shapiro(group3)

(stat_m1 + stat_m2 + stat_m3)/3


#5한 음료 기계는 사용설명서에 따르면 25초 작동 시 평균 2.0온스의 음료가 추출된다고 알려져 있다. 이를 검증하기 위해 한 직원이 직접 8번의 실험을 진행한 결과, 다음과 같은 샘플 데이터를 얻었다.
sample = [1.95, 1.80, 2.10, 1.82, 1.75, 2.01, 1.83, 1.90]
t_stat, p_val = ttest_1samp(sample, popmean=100)

#90% 신뢰구간


#7 
#관측빈도
#이 공정은 평균 100g, 표준편차 5g의 정규분포를 따른다고 알려져 있다. 다음과 같은 4개 구간으로 나누어 관측빈도를 계산하시오(위 표의 빈칸 순서대로 관측 빈도 산출)
sample = [96, 95, 103.2, 101.0, 100.7, 99.9, 98.6, 100.1, 97.3, 98.4, 99.5, 100.2, 101.4, 100.9, 102.0, 96.8, 99.1]
import numpy as np
import scipy.stats as stats



#9
old_env = np.array([72, 68, 74, 70, 65, 69, 71, 73, 67, 66])

new_env = np.array([78, 70, 76, 74, 69, 72, 75, 77, 70, 72])

from scipy.stats import ttest_rel

t_stat, p_val = ttest_rel(new_env, old_env)

print("대응표본 t-검정")
print("t 통계량:",t_stat)
print("p-value:", p_val)

#12
df12

#14
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

data = fetch_openml(name="energy_efficiency", version=1, as_frame=True)

X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
lr = LinearRegression()

lr.fit(X_train, y_train)
lr.coef_
lr.intercept_

y_pred = lr.predict(X_test)
y_pred
rmse_manual = np.sqrt(((y_test - y_pred) ** 2).mean())

lasso_params = {'alpha': np.arange(0.1, 1.0, 0.1)}
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
lasso_search = GridSearchCV(estimator=lasso, 
                              param_grid=lasso_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')
lasso_search.fit(X_train, y_train)
lasso_search.get_params()
print(lasso_search.best_estimator_)
print(lasso_search.best_params_)

#16
df16 = pd.read_csv("./data/quiz5/problem15.csv")
df16.info()
sum(df16["age"]>=24)
sum(df16["absences"]>=100)
df16.isna().sum(axis=0)
df16["absences"].hist()

#17
df16.dropna(subset='absences', inplace=True)
X = df16
y = df16["age"]
X_train, X_test, y_train, y_test = train_test_split(df16, df16["age"], test_size=0.3, random_state=0)

from sklearn.impute import KNNImputer
knnimputer = KNNImputer(n_neighbors = 5)
X_train["medu"] = knnimputer.fit_transform(X_train[["medu"]])
X_train[['traveltime', 'studytime', 'freetime', 'famrel']] = X_train[['traveltime', 'studytime', 'freetime', 'famrel']].astype('object')

cat_columns = X_train.select_dtypes(include=['object']).columns

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(handle_unknown='ignore', 
                       sparse_output=False).set_output(transform="pandas")
X_train_cat_columns = onehot.fit_transform(X_train[cat_columns])
X_train_full = pd.concat([X_train.drop(columns=cat_columns), X_train_cat_columns], axis=1)
X_train_full.mean()


#19
df19 = pd.read_csv("./data/quiz5/problem19.csv")
df19
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
X = df19
y = df19['absences']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


dct = DecisionTreeRegressor()
lasso = Lasso()
dct.get_params()
dct_params = {'ccp_alpha': np.arange(0.1, 1.0, 0.1)}
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                              param_grid=dct_params, 
                              cv = cv, 
                              scoring='neg_mean_absolute_error')
dct_search.fit(X_train, y_train)
dct_search.get_params()
print(dct_search.best_estimator_)
print(dct_search.best_params_)
print(dct_search.best_score_)


lasso.get_params()
lasso_params = {'alpha': np.arange(0.1, 1.0, 0.1)}
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
lasso_search = GridSearchCV(estimator=lasso, 
                              param_grid=lasso_params, 
                              cv = cv, 
                              scoring='neg_mean_absolute_error')
lasso_search.fit(X_train, y_train)
lasso_search.get_params()
lasso_search.fit(X_train, y_train)
lasso_search.get_params()
print(lasso_search.best_estimator_)
print(lasso_search.best_params_)

from sklearn.linear_model import LassoCV
alphas = np.linspace(0.0001, 20, 2000) # 200개 구간 분할

lasso_cv = LassoCV(alphas=alphas, cv=10, random_state=2025)
lasso_cv.fit(train_X, train_y)
lasso_cv.alpha_
y_pred_la = lasso_cv.predict(test_X)
rmse_Lasso = cal_rmse(test_y, y_pred_la) # 10.186997500904468




#23
df23 = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_3_2.csv")
df23.info()
sum(df23["Phone_Service"] == 0) & sum(df23["churn"] == 1)
sum(df23["Phone_Service"] == 1) & sum(df23["churn"] == 1)

(51/193)/(1-51/193)
#25
from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)
train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
model = ols(
    "bill_length_mm ~ bill_depth_mm + species + bill_depth_mm:species", data=train_data
).fit()
print(model.summary())