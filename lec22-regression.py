import numpy as np
import pandas as pd
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# 1. Iris 데이터 로드
df_iris = load_iris()

# 2. pandas DataFrame으로 변환
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] #컬럼명 변경시

# 3. 타겟(클래스) 추가
iris["Species"] = df_iris.target

# 4. 클래스 라벨을 실제 이름으로 변환 (0: setosa, 1: versicolor, 2: virginica)
iris["Species"] = iris["Species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

iris.info()

import statsmodels.api as sm
import statsmodels.formula.api as smf
# 4. 회귀모델 학습 (Petal_Length ~ Petal_Width)
model = smf.ols("Petal_Length ~ Petal_Width + C(Species)", data=iris).fit()
print(model.summary())
model.params
# 예측용 데이터
new_data = pd.DataFrame({
    "Petal_Width" : [0.5],
    "Species": ["virginica"]
})
#예측
prediction = model.predict(new_data)
prediction

sm.stats.anova_lm(model)

import statsmodels.api as sm
from statsmodels.formula.api import ols


model1 = ols('Petal_Length ~ Petal_Width', data=iris).fit() #mod1
model2 = ols('Petal_Length ~ Petal_Width + Sepal_Length + Sepal_Width',
data=iris).fit() #mod2
table = sm.stats.anova_lm(model1, model2) #anova
print(table)


intercept, slope = model.params

# 5. 산점도 + 회귀직선 그리기
plt.figure(figsize=(8,6))
sns.scatterplot(data=iris, x="Petal_Width", y="Petal_Length", hue="Species", palette="Set1")

# 회귀직선 계산
x_vals = np.linspace(iris["Petal_Width"].min(), iris["Petal_Width"].max(), 100)
y_vals = model.params["Intercept"] + model.params["Petal_Width"] * x_vals

plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Regression Line")

plt.xlabel("Petal Width (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Iris Petal Width vs Length with Regression Line")
plt.legend()
plt.show()

model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Length + C(Species)", data=iris).fit()
print(model.summary())


# =============================================================================
import pandas as pd
import numpy as np
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
penguins

np.random.seed(2022)
train_index=np.random.choice(penguins.shape[0],200)
train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
train_data.head()

# 1 팔머펭귄 데이터의 부리길이를 종속변수,
# 부리 깊이를 독립변수로 설정하여 회귀직선을 구하시오.
# 1) 산점도를 그린 후, 구해진 직선을 시각화 해보세요.
model = smf.ols("bill_length_mm ~ bill_depth_mm", data=train_data).fit()
print(model.summary())

intercept, slope = model.params

# 3. 예측값 및 잔차
train_data["y_pred"] = model.predict(train_data["bill_depth_mm"])
train_data["residual"] = train_data["bill_depth_mm"] - train_data["y_pred"]

# 5. 산점도 + 회귀직선 그리기
plt.figure(figsize=(8,6))
sns.scatterplot(data=train_data, x="bill_depth_mm", y="bill_length_mm", hue="species", palette="Set2")

# 회귀직선 계산
x_vals = np.linspace(train_data["bill_depth_mm"].min(),train_data["bill_depth_mm"].max(), 100)
y_vals = model.params["Intercept"] + model.params["bill_depth_mm"] * x_vals

plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Regression Line")

x = train_data["bill_depth_mm"]
y = -0.7062 * x + 55.4110 

np.sum((train_data["bill_length_mm"] - y)**2)
SSE = np.sum((train_data["bill_length_mm"] - y)**2)

plt.text(
    x=train_data["bill_depth_mm"].min()+0.2, 
    y=train_data["bill_length_mm"].max()-0.5,
    s=f"SSE = {SSE:.2f}",
    fontsize=12, color="blue", bbox=dict(facecolor="white", alpha=0.7)
)
plt.xlabel("Bill Depth (mm)")
plt.ylabel("Bill Length (mm)")
plt.title("Penguins Bill Length vs Width with Regression Line")
plt.legend()
plt.show()

#2) 독립변수와 종속변수의 관계를 직선계수를 사용해서 해석해보세요.
#부리깊이가 단위길이 만큼(1mm) 넓어지면 부리길이가 0.7만큼씩 짧아진다

#3) 계수 유의성을 통해 해석 가능성을 이야기 해보세요.
# 0.000으로 매우 작음으로 계수가 유의미하다
# 부리깊이에 대응하는 계수의 유의확률이 0.05보다 작으므로, 부리깊이 계수가 0이 아니라는 통계적 근거가 충분하다. 따라서, 위 해석은 타당하다.

# 넘파이를 사용해서, 직선과 주어진 점들의 수직거리를 한 변으로하는 사각형들의 넓이의 합을 계산하세요.

X = np.array([0, 1])
X_prob = np.array([0.5, 0.5])
Y = np.array([2, 4, 6])
Y_prob = np.array([0.3, 0.3, 0.4])

EX = np.sum(X * X_prob)
EY = np.sum(Y * Y_prob)

EX2 = np.sum((X**2) * X_prob)
EY2 = np.sum((Y**2) * Y_prob)

VX = EX2 - (EX**2)
VY = EY2 - (EY**2)

x=np.array([0, 0, 0, 1, 1, 1])
y=np.array([2, 4, 6, 2, 4, 6])

(x - 0.5) * (y - 4.2)

px=np.array([0.15, 0.15, 0.2, 0.15, 0.15, 0.2])

CovXY = np.sum((x - 0.5) * (y - 4.2) * px)
CovXY / (np.sqrt(0.25) * np.sqrt(2.76))

x=np.array([0, 0, 0, 1, 1, 1])
y=np.array([2, 4, 6, 2, 4, 6])

px=np.array([0.15, 0.15, 0.2, 0.15, 0.15, 0.2])


CovXY = np.sum((x - 0.5) * (y - 4.2) * px)
CovXY / (np.sqrt(0.25) * np.sqrt(2.76))



x_vals=np.array([0, 0, 0, 1, 1, 1])
y_vals=np.array([2, 4, 6, 2, 4, 6])

probs=np.array([0.15, 0.15, 0.2, 0.15, 0.15, 0.2])

N = 300
idx = np.random.choice(len(x_vals), size=N, p=probs)
x = np.array([x_vals[i] for i in idx])
y = np.array([y_vals[i] for i in idx])

upper = np.sum((x - x.mean()) * (y - y.mean()))
lower_l=np.sqrt(np.sum((x - x.mean())**2))
lower_r=np.sqrt(np.sum((y - y.mean())**2))
upper / (lower_l * lower_r)

import scipy.stats as stats
corr_coeff, p_value = stats.pearsonr(x, y)

import numpy as np
x = np.array([10, 20, 30, 40, 50])
y = np.array([5, 15, 25, 35, 48]).reshape(-1, 1)

x = x.reshape(-1, 1)
X = np.hstack([np.ones((x.shape[0], 1)), x])

beta = np.array([2.5, 3.2]).reshape(-1, 1)
beta

def ssr(beta_vec):
    return (y - X @ beta_vec).transpose() @ (y -  X @ beta_vec)

ssr(beta)

from scipy.optimize import minimize


# SSR 함수 정의
def ssr(beta_vec):
    beta_vec = beta_vec.reshape(-1, 1)
    return float((y - X @ beta_vec).T @ (y - X @ beta_vec))

# 초기값
beta0 = np.array([0.0, 0.0])

# 최적화 실행
result = minimize(ssr, beta0, method="BFGS")
print("최적 베타:", result.x)
print("최소 SSR:", result.fun)


import pandas as pd
import numpy as np
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
print(penguins.head())

np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)
#1 train_index 를 사용하여 펭귄 데이터에서 인덱스에 대응하는 표본들을 뽑아서 train_data를 만드세요. (단, 결측치가 있는 경우 제거)
train_data = penguins.iloc[train_index]
train_data = train_data.dropna()

#2 train_data의 펭귄 부리길이 (bill_length_mm)를 부리 깊이 (bill_depth_mm)를 사용하여 산점도를 그려보세요.
plt.figure(figsize=(8,6))
sns.scatterplot(data=train_data, x="bill_depth_mm", y="bill_length_mm", hue="species", palette="Set2")

#3. 펭귄 부리길이 (bill_length_mm)를 부리 깊이 (bill_depth_mm)의 상관계수를 구하고, 두 변수사이에 유의미한 상관성이 존재하는지 검정해보세요.
model = smf.ols("bill_length_mm ~ bill_depth_mm", data=train_data).fit()
print(model.summary()) 

from scipy.stats import pearsonr

# 두 변수 선택
x = train_data["bill_length_mm"]
y = train_data["bill_depth_mm"]

# 피어슨 상관계수와 p-value
corr, pval = pearsonr(x, y)

print("상관계수 r =", round(corr, 3))
print("p-value =", pval)

#4 
# 회귀직선 계산
x_vals = np.linspace(train_data["bill_depth_mm"].min(),train_data["bill_depth_mm"].max(), 200)
y_vals = model.params["Intercept"] + model.params["bill_depth_mm"] * x_vals

plt.figure(figsize=(8,6))
sns.scatterplot(data=train_data, x="bill_depth_mm", y="bill_length_mm", hue="species", palette="Set2")
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Regression Line")

#5 F-statistics 값으로 판단

#6 R^2 = 0.062
# 데이터 전체 변동성의 약 6.2%정도를 회귀 모델이 설명하고 있다.

#7 
model.params
#bill_depth_mm의 계수 -0.706191
# 부리깊이가 1mm 증가하면, 부리길이는 평균적으로 0.7mm 감소하는 경향을 보인다.

model.resid

import scipy.stats as stats

residuals = model2.resid
fitted_values = model2.fittedvalues

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(fitted_values, residuals);

plt.subplot(1,2,2)
stats.probplot(residuals, plot=plt);
plt.show()

model2 = smf.ols("bill_length_mm ~ bill_depth_mm + C(species)", data=train_data).fit()

#8

#[연습문제] 선형 회귀분석
#1
import numpy as np
from scipy.stats import pearsonr
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 13])

import scipy.stats as stats
corr_coeff, p_value = stats.pearsonr(x, y)
print(corr_coeff)
print(p_value)

#2
x = np.array([1, 2, 3, 4, 10, 11, 12])
y = np.array([2, 4, 6, 8, 100, 200, -100])
corr_coeff, p_value = stats.pearsonr(x, y)
print(corr_coeff)
print(p_value)
 

#3
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 9, 12, 15])

#4

#5
import pandas as pd
from sklearn.datasets import fetch_california_housing
cal = fetch_california_housing(as_frame=True)
df = cal.frame

# 독립변수와 종속변수
X = df[["AveRooms", "AveOccup"]]
y = df["MedHouseVal"]

# 상수항 추가 (절편 포함)
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
1.6919 + 0.0708 * X["AveRooms"] - 0.0026 * X["AveOccup"]

#6
import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.ols('MedHouseVal ~ AveRooms + AveOccup', data=df).fit()
print(model.summary())
# AveRooms 22.074, 0.000

#7
df['IncomeLevel'] = pd.qcut(df['MedInc'], q=3, labels=['Low', 'Mid', 'High'])
model = smf.ols('MedHouseVal ~ AveRooms + AveOccup + C(IncomeLevel)', data=df).fit()
print(model.summary())

#8
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(model.resid)
print(dw_stat)

from sklearn.datasets import load_diabetes
# 데이터 불러오기 및 DataFrame 변환
diabetes = load_diabetes(as_frame=True)
df2 = diabetes.frame

model = smf.ols('target ~ bmi+ bp + s1', data=df2).fit()
print(model.summary())
model.rsquared_adj

ci_df = model.conf_int()
ci_df.iloc[1,:]

from statsmodels.formula.api import ols
model2 = ols("target ~ bmi + bp + s1 + s2", data=df2).fit()
print("model2>>> ", model2.summary())

#15
model3 = ols("target ~ bmi + bp + s1 + s2 + s3", data=df2).fit()
print("model3>>>", model3.summary())

#16
import seaborn as sns
penguins = sns.load_dataset("penguins").dropna()
model_peng = ols("body_mass_g ~ bill_length_mm + flipper_length_mm + C(species)", data=penguins).fit()
print("model_peng>>>>", model_peng.summary())
# 독립변수와 종속변수
X = penguins[["bill_length_mm", "flipper_length_mm"]]
y = penguins["body_mass_g"]

# 절편항 추가
X = sm.add_constant(X)

model_peng = sm.OLS(y, X).fit()
print(model_peng.summary())

r2 = model.rsquared
print("결정계수 R² =", r2)

#18
model_peng = ols("body_mass_g ~ bill_length_mm + flipper_length_mm + C(sex) + C(species)", data=penguins).fit()
print("model_peng>>>>", model_peng.summary())


import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols("Petal_Length ~ Petal_Width", data=iris).fit()
sm.stats.anova_lm(model)
