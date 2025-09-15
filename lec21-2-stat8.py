import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 자유도 설정
df = 4  # 자유도(degrees of freedom)
x = np.linspace(0, 20, 500)  # x 범위
pdf = chi2.pdf(x, df)        # 카이제곱분포 pdf 계산

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, label=f"Chi-square (df={df})", color="blue")
plt.fill_between(x, pdf, alpha=0.2, color="blue")
plt.title("Chi-square Distribution PDF")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

from scipy.stats import chi2
from scipy.stats import norm

X = chi2(df=3)
1-X.cdf(8)

Y = norm(loc=3, scale=2)

data_set = Y.rvs(500 * 15).reshape(500, -1)
s_2=data_set. var(ddof=1, axis=1)#각 행별로 S**2을 구한 것
statistics = s_2 * (15-1)/2**2

# 5. 히스토그램 + 이론적 카이제곱 PDF
x = np.linspace(0, max(statistics), 500)
pdf = chi2.pdf(x, df=14)  # 자유도 = n-1 = 14

plt.figure(figsize=(7,5))
plt.hist(statistics, bins=30, density=True, alpha=0.6, label="Simulated")
plt.plot(x, pdf, "r-", lw=2, label="Chi2 PDF (df=14)")
plt.title("Sample variance → Chi-square distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 예제
# 베어링 제조 회사의 품질 관리를 맡고 있는 정부 기관의 규정에 따르면, 생산되는 제품의 금속 재질
# 함유량 분산이 1.3 을 넘으면 생산 부적격이라고 판단한다. 다음은 A 회사 제품의 금속 함유량를 검
# 사한 데이터이다. 데이터를 기준으로 해당회사의 생산 부적격 검정을 수행하시오. 단, 유의 수준은
# 5%로 설정하시오.
# 위 문제에 해당하는 귀무가설과 대립가설은 다음과 같이 설정할 수 있습니다.
X = np.array([10.67, 9.92, 9.62, 9.53, 9.14, 9.74, 8.45, 12.65, 11.47, 8.62])
stat_t = (10-1) * X.var(ddof=1)/1.3

1- chi2.cdf(stat_t, df = 9)
# 유의확률인 0.235가 유의수준 5%보다 크므로 귀무가설을 기각하지 못한다.
# 현재 생산 적격이라 판단함.

# 1표본 t 검정 => 모평균이 특정값인지를 검정
# 1표본 카이제곱 검정 => 모분산이 특정값인지를 검정

1-chi2.cdf(15.55, df=1)

# 독립성 검정 예제
from scipy.stats import chi2_contingency

table = np.array([[14, 4],
                  [0, 10]])
chi2, p, df, expected = chi2_contingency(table, correction=False)
print('X-squared:', 
      chi2.round(3), 
      'df:', df,
      'p-value:', 
      p.round(3))

# 데이터 설정: 교차표
table = np.array([[50, 30, 20], # 도시 X
[45, 35, 20]]) # 도시 Y
chi2, p, df, expected = chi2_contingency(table, correction=False)

1-chi2.cdf(0.6478, df=2)





o_i = np.array([13, 23, 24, 20, 27, 18, 15])
e_i = np.repeat(20, 7)
sum((o_i - e_i)**2 / e_i)

1-chi2.cdf(7.6, df=6)

from scipy.stats import chisquare
observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)

statistic, p_value = chisquare(observed, f_exp=expected)
print("Test statistic: ", statistic.round(3))








from palmerpenguins import load_penguins
import pandas as pd
df = load_penguins()
penguins=df.dropna()

x = penguins["bill_length_mm"]
x.mean()
x.std(ddof=1)

# 정규분포 평균 43.993, 표준편차 5.468 따르는지 체크하고 싶다.
X = norm(loc=43.993, scale=5.468)
X.cdf(52.725) - X.cdf(45.85) # 104개 정도 예측
X.cdf(45.85) - X.cdf(38.975) # 150개 정도 예측
len(x)
pd.cut(x, bins=4).value_counts()





#======================================================================================================
# [연습문제] 카이제곱 검정 이해하기
#======================================================================================================

#문제 1
# 데이터 불러오기
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
# 임신 유무 파생변수 생성
dat['Pregnancy_status'] = (dat['Pregnancies'] > 0).astype(int)

dat.info()
#1-1. 귀무가설 : 임신 유무와 당뇨병 여부는 관계가 있다.
#     대립가설 : 임신 유무와 당뇨병 여부는 관계가 없다.

#1-2. 

#1-3.
ob1 = [
    [sum((dat["Pregnancy_status"] == 0) & (dat["Outcome"] == 0)), sum((dat["Pregnancy_status"] == 0) & (dat["Outcome"] == 1))],
    [sum((dat["Pregnancy_status"] == 1) & (dat["Outcome"] == 0)), sum((dat["Pregnancy_status"] == 1) & (dat["Outcome"] == 1))]
    ]
chi21, p1, df1, expected1 = chi2_contingency(ob1, correction=False)
1-chi2.cdf(chi21, df=1)

#1-5. 귀무가설 채택한다. 유의수준이 0.847로 기각하지 못한다.

# 문제2
#2-1. 귀무가설: 연령대와 당뇨병은 관계가 있다.
#     대립가설: 연령대와 당뇨병은 관계가 없다.

#2-2. 시각화?

#2-3.
ob2 = [
    [sum((dat["Age"] < 40) & (dat["Outcome"] == 0)), sum((dat["Age"] < 40) & (dat["Outcome"] == 1))],
    [sum((dat["Age"] >= 40) & (dat["Outcome"] == 0)), sum((dat["Age"] >= 40) & (dat["Outcome"] == 1))]
    ]

chi22, p2, df2, expected2 = chi2_contingency(ob2, correction=False)


