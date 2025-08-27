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
model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Length", data=iris).fit()
print(model.summary())
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