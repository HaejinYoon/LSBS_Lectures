#QQ plot
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot

data_x = dat[dat['Gender'] == 'Male']['Salary']
plt.figure(figsize=(6, 4));
probplot(data_x, dist="norm", plot=plt);
plt.title(f'Q-Q Plot of Salary (Male)');
plt.grid(True);
plt.show();

#정규성 검정, shapiro 검정
from scipy.stats import shapiro
stat_m, p_m = shapiro(data_x)
#stat_m 검정통계량
#p_m : 유의 확률 (p-value)

# 확률질량함수 (pmf) / 확률밀도함수 (pdf)
# 누적분포함수 (cdf)
# 퀀타일 함수 (ppf)
# 랜덤샘플함수 (rvs)

# 표준 오차
se = np.std(data, ddof=1) / np.sqrt(n)

#t-value
x = [4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
11.97, 2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9]
t_value = (np.mean(x)-7) / (np.std(x, ddof=1) / np.sqrt(len(x)))

#Z 검정
from scipy.stats import norm

# 주어진 값
mu = 100           # 모집단 평균
sigma = 15         # 모집단 표준편차 (알고 있음)
x_bar = 105        # 표본 평균
n = 36             # 표본 크기

# Z 검정 통계량
z = (x_bar - mu) / (sigma / (n ** 0.5))

# p값 (양측 검정)
p_value = 2 * (1 - norm.cdf(abs(z)))

print("Z 검정 통계량:", round(z, 4))
print("p-value:", round(p_value, 4))

# T 검정(단일 표본 평균 검정)
from scipy.stats import t

# 표본 데이터
sample = [102, 98, 101, 105, 100, 99, 97]
n = len(sample)
x_bar = sum(sample) / n
s = (sum((x - x_bar)**2 for x in sample) / (n - 1))**0.5  # 표본 표준편차
mu = 100  # 모집단 평균

# T 검정 통계량
t_stat = (x_bar - mu) / (s / (n ** 0.5))

# p값 (양측 검정)
p_value = 2 * (1 - t.cdf(abs(t_stat), df=n - 1))

print("T 검정 통계량:", round(t_stat, 4))
print("p-value:", round(p_value, 4))


from scipy.stats import ttest_1samp

sample = [102, 98, 101, 105, 100, 99, 97]
t_stat, p_val = ttest_1samp(sample, popmean=100)

print("T 검정 통계량:", round(t_stat, 4))
print("p-value:", round(p_val, 4))

#  1. 단일 표본 t-검정 (1-sample t-test)
# 모집단 평균과 표본 평균이 다른지 검정

from scipy.stats import ttest_1samp

# 표본 데이터
sample = [102, 98, 101, 105, 100, 99, 97]

# 검정: 모평균이 100인가?
t_stat, p_val = ttest_1samp(sample, popmean=100)

print("단일 표본 t-검정")
print("t 통계량:", round(t_stat, 4))
print("p-value:", round(p_val, 4))

#✅ 2. 독립표본 t-검정 (2-sample independent t-test)
#두 독립된 집단의 평균 차이를 검정

from scipy.stats import ttest_ind

# 예: 두 그룹 점수
group1 = [88, 92, 85, 91, 87]
group2 = [82, 79, 88, 84, 80]

# 등분산 가정 o → equal_var=True
t_stat, p_val = ttest_ind(group1, group2, equal_var=True)

print("독립표본 t-검정")
print("t 통계량:", round(t_stat, 4))
print("p-value:", round(p_val, 4))

#✅ 3. 대응표본 t-검정 (paired t-test)
#같은 대상의 전/후 비교 (ex. 치료 전/후 체중)

from scipy.stats import ttest_rel

# 운동 전/후 체중
before = [70, 68, 72, 69, 71]
after  = [68, 67, 70, 68, 70]

t_stat, p_val = ttest_rel(before, after)

print("대응표본 t-검정")
print("t 통계량:", round(t_stat, 4))
print("p-value:", round(p_val, 4))

# ✅ 단측 검정 (1-tailed test)
# 위의 예제들은 모두 양측검정. 단측검정은 p-value를 2로 나누고 방향성 확인.

#예: 평균이 100보다 크다는 가설 → p = (1 - cdf(t))

# 단측검정 예시 (단일표본, 평균이 100보다 크다)
from scipy.stats import ttest_1samp

sample = [102, 98, 101, 105, 100, 99, 97]

t_stat, p_val_two_sided = ttest_1samp(sample, popmean=100)

# 단측 검정: 평균이 100보다 **크다**
if t_stat > 0:
    p_val_one_tailed = p_val_two_sided / 2
else:
    p_val_one_tailed = 1 - p_val_two_sided / 2

print("단측 검정 (H1: 평균 > 100)")
print("t 통계량:", round(t_stat, 4))
print("단측 p-value:", round(p_val_one_tailed, 4))





0.5*0.8 + 0.3*0.5 + 0.2*0.9



0.75**2 - 0.25**2
((1.5**2)/4) - (1/4)

x = np.array([1, 2, 3])
prob = np.array([0.2, 0.5, 0.3])

E_X = np.sum(x * prob)
E_X2 = np.sum((x**2) * prob)
theoretical_var = E_X2 - E_X**2

1 - ((0.5**4)/4)
15/16
7/8
from scipy.special import comb
#nCr 계산
(1 - (comb(20,1)*(0.02**1) * (0.98**19) + comb(20,0)*(0.02**0) * (0.98**20))) + (comb(20,1)*(0.02**1) * (0.98**19) + comb(20,0)*(0.02**0) * (0.98**20))


from scipy.stats import uniform
uniform.pdf(2, loc=0, scale=4)
uniform.cdf(3, loc=0, scale=4) - uniform.cdf(1, loc=0, scale=4)

from scipy.stats import expon

X1 = expon(scale=0.5)
X2 = expon(scale=1/3)
X2.mean()
X2.var()

from scipy.stats import norm

norm.cdf(4.95, loc=5, scale=0.05)
norm.pdf(6, loc=8, scale=2)
norm.ppf(0.90, loc=32, scale=6)
norm.ppf(0.10, loc=32, scale=6)

0.22 * 0.85

x = [21, 12, 24, 18, 25, 28, 22, 22, 29, 14, 20, 45, 16, 18, 15, 17, 23, 55, 19, 26]
import numpy as np
data = np.array([21, 12, 24, 18, 25, 28, 22, 22, 29, 14, 20, 45, 16, 18, 15, 17, 23, 55, 19, 26])
sorted_data = np.sort(data) # 데이터 정렬
minimum = np.min(sorted_data) # 최소값과 최대값
maximum = np.max(sorted_data)
median = np.median(sorted_data) # 중앙값
lower_half = sorted_data[sorted_data < median] # 중앙값보다 크거나, 작은 데이터들 필터
upper_half = sorted_data[sorted_data > median]
q1 = np.median(lower_half) # 1사분위수와 3사분위수
q3 = np.median(upper_half)
print("최소값:", minimum, "제 1사분위수:", q1, "중앙값:", median, "제 3사분위수:", q3, "최대값:", maximum)

q1 = np.quantile(sorted_data, 0.25)
q3 = np.quantile(sorted_data, 0.75)
iqr = q3 - q1
q1 - 1.5 * iqr,
sorted_data[sorted_data > q3 + 1.5 *iqr]

X = norm(loc=3, scale=2)
q1 = np.quantile(X, 0.25)
q3 = np.quantile(X, 0.75)

X.ppf(0.75) -X.ppf(0.25) 
X.ppf(0.25) -X.ppf(0.75) 


df = pd.read_csv('./quiz3/datasetSalaries.csv')

df.head()
df.info()

sample = df["salary"]
n = len(sample)
x_bar = sum(sample) / n
s = (sum((x - x_bar)**2 for x in sample) / (n - 1))**0.5  # 표본 표준편차
mu = 50221  # 모집단 평균

# T 검정 통계량
t_stat = (x_bar - mu) / (s / (n ** 0.5))

# p값 (양측 검정)
p_value = 2 * (1 - t.cdf(abs(t_stat), df=n - 1))

print("T 검정 통계량:", round(t_stat, 4))
print("p-value:", round(p_value, 4))

line_a = [2011, 2005, 1998, 2003, 2008, 2001, 2006]
line_b = [1985, 1991, 1988, 1992, 1986, 1990, 1987]
line_c = [2020, 2024, 2019, 2026, 2023, 2025, 2022]

stat_m, p_m = shapiro(line_a)
stat_m, p_m = shapiro(line_b)
stat_m, p_m = shapiro(line_c)

drug_a = np.array([142.9, 140.6, 144.7, 144.0, 142.4, 146.0, 149.1, 150.4])

drug_b = np.array([139.1, 136.4, 147.3, 139.4, 143.0, 142.2, 142.2, 147.9])


from scipy.stats import ttest_rel

# 등분산 가정 o → equal_var=True
t_stat, p_val = ttest_ind(drug_a, drug_b, equal_var=False)

print("독립표본 t-검정")
print("t 통계량:", round(t_stat, 4))
print("p-value:", round(p_val, 4))

t_stat, p_val = ttest_rel(drug_a, drug_b)

print("대응표본 t-검정")
print("t 통계량:", round(t_stat, 4))
print("p-value:", round(p_val, 4))

from scipy.stats import ttest_1samp
aa = drug_b - drug_a
t_statistic, p_value = ttest_1samp(aa, 0, alternative='greater')
print("t-statistic:", t_statistic, "p-value:", p_value)







[1.2, 0.9, 1.5, 2.1, 0.7, 0.8, 1.8, 2.2, 1.0, 1.3, 2.5, 2.0, 1.1, 1.6, 0.6]

from scipy.stats import anderson, norm
import scipy.stats as sp
from statsmodels.distributions.empirical_distribution import ECDF
sample_data = np.array([1.2, 0.9, 1.5, 2.1, 0.7, 0.8, 1.8, 2.2, 1.0, 1.3, 2.5, 2.0, 1.1, 1.6, 0.6])
result = sp.anderson(sample_data, dist='expon') # Anderson-Darling 검정 수행
print('검정통계량',result[0], '임계값:',result[1], '유의수준:',result[2])

ecdf = ECDF(sample_data)
x = np.linspace(min(sample_data), max(sample_data))
y = ecdf(x)
plt.scatter(x, y)
plt.title("Estimated CDF vs. CDF")
k = np.arange(min(sample_data), max(sample_data), 0.1)
plt.plot(k, norm.cdf(k, loc=np.mean(sample_data),
scale=np.std(sample_data, ddof=1)), color='k')
plt.show()

#c*x**3 (0<= x <=1)
#0, otherwise
#P(X>0.5)
1 - (0.5**4)*(1/4)