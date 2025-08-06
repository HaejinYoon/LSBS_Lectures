# 팔머펭귄 데이터의 각 종별 부리길이의 사분위수를 계산하세요
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins

adel = penguins[penguins["species"] == 'Adelie']["bill_length_mm"]
adel = adel.sort_values()

gen2 = penguins[penguins["species"] == 'Gentoo']["bill_length_mm"]
gen2 = gen2.sort_values()

chin = penguins[penguins["species"] == 'Chinstrap']["bill_length_mm"]
chin = chin.sort_values()



def quarter(data, species):
    median = np.median(data)
    lower_half = data[data < median] # 중앙값보다 크거나, 작은 데이터들 필터
    upper_half = data[data > median]
    q1 = np.median(lower_half) # 1사분위수와 3사분위수
    q3 = np.median(upper_half)
    print(species, "--- ",  "Q1:", q1, "Q2:", np.round(median, 1), "Q3:", q3)

quarter(adel, "Adelie")
quarter(gen2, "Gentoo")
quarter(chin, "Chinstrap")

np.quantile(adel, 0.25)
np.quantile(adel, 0.5)
np.quantile(adel, 0.75)

# 이상치 판별하는 방법
# 1Q에서 -1.5*IQR
# 3Q에서 +1.5*IQR
# 구간에서 벗어나는 데이터는 이상치로 판단

data = np.array([155, 126, 10, 82, 115, 140, 73, 92, 110, 134])

#박스플랏 그리기


#데이터
scores = np.array([88, 92, 95, 91, 87, 89, 94, 90, 92, 100, 43])

quarter(scores, "예제")
IQR = 94 - 88
down = 88 - 1.5*IQR
up = 94 + 1.5*IQR

q1 = np.quantile(scores, 0.25)
q3 = np.quantile(scores, 0.75)
iqr = q3 - q1

q3 + 1.5 *iqr

plt.figure(figsize=(6, 4))
plt.boxplot(scores, vert=True)
plt.title("Boxplot of Data")
plt.ylabel("Values")
plt.grid(True)

data = np.array([155, 126, 27, 82, 115, 140, 73, 92, 110, 134])
sorted_data = np.sort(data)
n = len(data)

np.quantile(data, [0.25, 0.5, 0.75])
np.percentile(data, [25, 50, 75])

np.arange(0.01, 1, step=0.01)
data_q = np.quantile(data, np.arange(0.01, 1, step=0.01))# 100등분 한 수


from scipy.stats import norm
data.mean()
data.std(ddof=1)
X = norm(loc = data.mean(), scale=data.std(ddof=1))
X.ppf(0.5) # 중간
X.ppf(0.25) # 1사분위수
X.ppf(0.75) # 3사분위수
norm_q = X.ppf(np.arange(0.01, 1, step=0.01))

data_q
norm_q


plt.scatter(data_q, norm_q)
plt.plot([25, 200], [25, 200], color='red', label='y=x')
plt.xlabel('data')
plt.ylabel('theory')
plt.show()
#정규분포를 따르지 않는다고 판단
#QQplot이라 부른다

import scipy.stats as sp
sp.probplot(data, dist="norm", plot = plt)
plt.show()

#Shapiro Wilk
import scipy.stats as sp
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55,
3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9])
w, p_value = sp.shapiro(data_x)
print("W:", w, "p-value:", p_value)


from statsmodels.distributions.empirical_distribution import ECDF
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87])
ecdf = ECDF(data_x)
x = np.linspace(min(data_x), max(data_x))
y = ecdf(x)
plt.plot(x,y,marker='o', linestyle='none')
plt.title("Estimated CDF")
plt.xlabel("X-axis")
plt.ylabel("ECDF")
# plt.show()


k = np.arange(min(data_x), max(data_x), 0.1)
plt.plot(k, norm.cdf(k, 
                     loc=np.mean(data_x),
                     scale=np.std(data_x, ddof=1)),
                     color='red')
plt.show()

#K - S test

from scipy.stats import kstest, norm
import numpy as np
sample_data = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87])
# 표본 평균과 표준편차로 정규분포 생성
loc = np.mean(sample_data)
scale = np.std(sample_data, ddof=1) 
 # 정규분포를 기준으로 K-S 검정 수행
result = kstest(sample_data, 'norm', args=(loc, scale))
print("검정통계량:", result.statistic)

result.statistic
result.pvalue

# A- D test
from scipy.stats import anderson, norm
sample_data = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87])

result = sp.anderson(data, dist='norm') # Anderson-Darling 검정 수행
print('검정통계량',result[0], '임계값:',result[1], '유의수준:',result[2])
#0.23은 15%이상 되는 곳의 넓이일 것이다.