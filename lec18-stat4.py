from scipy.stats import t
import numpy as np
data = [4.3, 4.1, 5.2, 4.9, 5.0, 4.5, 4.7, 4.8, 5.2, 4.6]
# 표본 평균
mean = np.mean(data)
# 표본 크기
n = len(data)
# 표준 오차
se = np.std(data, ddof=1) / np.sqrt(n)

# 95% 신뢰구간
round(mean - t.ppf(0.975, n-1) * se, 3); round(mean + t.ppf(0.975, n-1) * se, 3)

# t.interval로 구하기
ci = t.interval(0.95, loc=mean, scale=se, df=n-1)
print("95% 신뢰구간: ", [round(i, 3) for i in ci])




# X ~ N(17, 2,21**2)
# P(X<=14) = 0.087

# H0: mu = 7 vs. HA: mu != 7
x = [4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
11.97, 2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9]
t_value = (np.mean(x)-7) / (np.std(x, ddof=1) / np.sqrt(len(x)))
round(t_value, 3)

# p-value
t.cdf(t_value, df=14) *2

# 유의수준: p-value가 큰지, 작은 지 판단하는 기준
# 따라서 0.2216은 유의 수준보다 크므로, 귀무가설을 기각 하지 못한다.

# p118. 문제 1
# 한 공장에서 생산된 배터리의 평균 수명은 500시간이며, 표준편차는 50시간입니다. 이 배터리에서
# 100개의 표본을 추출했을 때, 표본 평균의 분포에 대한 다음 질문에 답하시오.
# - 표본 평균의 분포는 어떤 분포를 따르나요?
# - 표준오차를 구하시오.
# - 표본 평균이 510시간 이상일 확률을 구하시오.
from scipy.stats import norm

norm.sf(510, loc=500, scale=50/np.sqrt(100))

#2 한 제품의 불량률이 5%인 경우, 이 제품 20개를 무작위로 뽑았을 떄:
# 불량품이 정확히 2개 나올 확률
# 2개 이하로 나올 확률
# 3개 이상 나올 확률
from scipy.stats import binom
Y = binom(n=20, p=0.05)
Y.pmf(2)
Y.cdf(2)
1 - Y.cdf(2) # P(X >= 3)
Y.sf(2) # P(X > 2) 부등호가 안들어감으로 Y.sf(3)은 안됨

# 한 학생의 수학 점수가 평균 75점, 표준편차 8점인 정규분포를 따릅니다.
# - 이 학생의 점수가 85점 이상일 확률을 구하시오.
# - 점수가 70점과 80점 사이에 있을 확률을 구하시오.
# - 상위 10%에 해당하는 점수 기준(컷오프)을 구하시오.

from scipy.stats import norm

norm.sf(85, loc=75, scale=8)
norm.cdf(80, loc=75, scale=8) - norm.cdf(70, loc=75, scale=8)
norm.ppf(0.9, loc=75, scale=8)



# 어느 커피숍에서 판매하는 커피 한 잔의 평균 온도가 75℃라고 주장하고 있습니다. 이 주장에 의문
# 을 가진 고객이 10잔의 커피 온도를 측정한 결과 다음과 같은 값을 얻었습니다.
x = np.array([72.4, 74.1, 73.7, 76.5, 75.3, 74.8, 75.9, 73.4, 74.6, 75.1])
## (72.4, 74.1, 73.7, 76.5, 75.3, 74.8, 75.9, 73.4, 74.6, 75.1)
# 귀무가설과 대립가설을 설정하시오.
# 유의수준 5%에서 단일 표본 t-검정을 수행하시오.
# 결론을 내리고 해석하시오.
x.mean()
x.std(ddof=1)
t.cdf(-1.087, df=9) * 2


# 한 제과점에서 하루 평균 판매되는 케이크의 개수가 50개라고 알려져 있습니다. 최근 데이터에서
# 표본 40일 동안 평균 53개의 케이크가 판매되었고, 표준편차는 8개였습니다.
# 귀무가설과 대립가설을 설정하시오.
# 유의수준 0.05에서 z-검정을 수행하시오.
# p-value를 계산하고, 귀무가설을 기각할 수 있는지 판단하시오.

(53 - 50) / (8 / np.sqrt(40))
norm.sf(2.3717, loc=0, scale=1) * 2




#########################################################################################
from scipy.stats import norm

# 평균(mu)=0, 표준편차(sigma)=1인 정규분포에서 10개 샘플 생성
X = norm(loc=5, scale=3)
sample = X.rvs(size=10000000)
sample.mean()
X.var()

import math
6.22/math.sqrt(20)
9.96/math.sqrt(20) 
X = norm(loc=14,scale=2.227)
X.ppf(0.025)
X.ppf(0.975)

import numpy as np
sample = np.array([14,17,12,14,13,14,16,10,14,15,13,17,12,12,16])
sam_mean = sample.mean()
all_sig = 3
sam_sig = 3/math.sqrt(15)
X = norm(loc=sam_mean, scale=sam_sig)
X.ppf(0.05) # 12
X.ppf(0.95) # 15
X.cdf(15.20)

# 1. Xi ~ 모분포는 균일분포 (3,7)
# 2. i=1,2,...,20
# 3. Xi들에서 표본을 하나씩 뽑은 후 표본 평균을 계산하고,
#  95% 신뢰구간을 계산 ( 모분산값 1번 정보로 사용 )
# 3번의 과정을 1000번 실행해서 ( 1000번의 신뢰구간 발생 ) 각 신뢰구간이 모평균을 포함하고 있는지 체크
from scipy.stats import uniform
X_var = (7-3)**2/12
X = uniform(loc=3, scale=4)
X.std()
ans = []
for i in range(1000):
    samples = X.rvs(20)
    sam_mean = samples.mean()
    sam_std = np.sqrt(X_var)/np.sqrt(20)
    sam_X = norm(loc=sam_mean,scale=sam_std)
    sam_1 = sam_X.ppf(0.025)
    sam_2 = sam_X.ppf(0.975)
    if (sam_1 <= 5)&(sam_2>=5)==True:
        ans.append(True)
    else:
        ans.append(False)
len(ans)
np.sum(ans)

# 시각화
# 이론적인 표본 평균의 분포의 pdf를 그리고, 모평균을 빨간색 막대기로 표현
# 3번에서 뽑은 표본들을 x축에 녹색 점들로 표시하고
# 95%의 신뢰구간을 녹색 막대기 2개로 표현
# 표본이 바뀔 때마다 녹색 막대기안에 빨간 막대기가 있는지 확인
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
import matplotlib.font_manager as fm

# 한글 폰트 설정 (예: Windows에서는 'Malgun Gothic', macOS는 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지


# 모분포 정의
X = uniform(loc=3, scale=4)
X_var = (7 - 3) ** 2 / 12
mu = 5  # 모평균
n = 20
sigma = np.sqrt(X_var)
se = sigma / np.sqrt(n)  # 표본평균의 표준오차

# 표본추출 및 신뢰구간 계산 (1회만)
samples = X.rvs(n)
sample_mean = samples.mean()

# 신뢰구간 계산
z = norm.ppf(0.975)
ci_lower = sample_mean - z * se
ci_upper = sample_mean + z * se

# x축 범위 및 표본평균 분포 정의
x = np.linspace(3.5, 6.5, 1000)
pdf = norm(loc=mu, scale=se).pdf(x)

# 시각화
plt.figure(figsize=(10, 6))

# 1. 이론적인 표본평균 분포의 PDF
plt.plot(x, pdf, label='표본평균의 분포', color='blue')

# 2. 모평균 (빨간색 세로선)
plt.axvline(x=mu, color='red', linestyle='--', label='모평균 (5)')

# 3. 표본 평균 (녹색 점)
plt.plot(sample_mean, 0, 'go', markersize=8, label='표본평균')

# 4. 신뢰구간 양 끝 (초록 세로 막대기)
plt.vlines([ci_lower, ci_upper], ymin=0, ymax=norm(loc=mu, scale=se).pdf(sample_mean), 
           color='green', linestyle='-', label='95% 신뢰구간')

plt.title('표본평균 분포 및 신뢰구간 시각화')
plt.xlabel('값')
plt.ylabel('밀도')
plt.legend()
plt.grid(True)
plt.show()

#########################
# 모분포 정의
X = uniform(loc=3, scale=4)
X_var = (7 - 3) ** 2 / 12
mu = 5  # 모평균
n = 20
sigma = np.sqrt(X_var)
se = sigma / np.sqrt(n)  # 표본평균의 표준오차

# 표본추출 및 신뢰구간 계산 (1회만)
samples = X.rvs(n)
sample_mean = samples.mean()

# 신뢰구간 계산
Z = norm(loc=sample_mean,scale=se)
ci_lower = Z.ppf(0.025)
ci_upper = Z.ppf(0.975)

# x축 범위 및 표본평균 분포 정의
x = np.linspace(3.5, 6.5, 1000)
pdf = norm(loc=mu, scale=se).pdf(x)

# 시각화
plt.figure(figsize=(10, 6))

# 1. 이론적인 표본평균 분포의 PDF
plt.plot(x, pdf, label='표본평균의 분포', color='blue')

# 2. 모평균 (빨간색 세로선)
plt.axvline(x=mu, color='red', linestyle='--', label='모평균 (5)')

# 3. 표본 평균 (녹색 점)
plt.plot(sample_mean, 0, 'go', markersize=8, label='표본평균')

# 4. 신뢰구간 양 끝 (초록 세로 막대기)
plt.vlines([ci_lower, ci_upper], ymin=0, ymax=norm(loc=mu, scale=se).pdf(sample_mean), 
           color='green', linestyle='-', label='95% 신뢰구간')

plt.title('표본평균 분포 및 신뢰구간 시각화')
plt.xlabel('값')
plt.ylabel('밀도')
plt.legend()
plt.grid(True)
plt.show()


x = np.array([14,17,12,14,13,14,16,10,14,15,13,17,12,12,16])
n = len(X)
sd = 3 / np.sqrt(n)

X = norm(loc=x.mean(),scale=sd)
z_05 = norm.ppf(0.05,loc=0,scale=1)
x.mean() + z_05 * 3/np.sqrt(n)
x.mean() - z_05 * 3/np.sqrt(n)

# t 분포와 표준 정규분포 비교
# t 분포는 자유도에 따라서 그래프 모양이 변함
# t 값이 작을수록 평균에 값이 몰리기보다 다른 사이드 값이 나오는 빈도가 높아짐
# 자유도는 계속 커질 수 있지만 표준 정규분포보다 높게 올라가지 않고 일치하게 된다.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# x 축 범위 설정
x = np.linspace(-5, 5, 500)

# 표준 정규분포 PDF
pdf_norm = norm.pdf(x)

# 자유도 5인 t-분포 PDF
df = 100
pdf_t = t.pdf(x, df=df)

# 시각화
plt.plot(x, pdf_norm, label='표준 정규분포', color='red')
plt.plot(x, pdf_t, label=f't-분포 (자유도={df})', linestyle='--', color='blue')

plt.title('표준 정규분포 vs t-분포 PDF')
plt.xlabel('x')
plt.ylabel('확률밀도 (PDF)')
plt.legend()
plt.grid(True)
plt.show()

# 신뢰구간
data = [4.3,4.1,5.2,4.9,5.0,4.5,4.7,4.8,5.2,4.6]
from scipy.stats import t
import numpy as np
mean = np.mean(data)
n = len(data)
se= np.std(data,ddof=1)/np.sqrt(n)
mean - t.ppf(0.975,loc=mean,scale=se,df=n-1)
t.interval(0.975,loc=mean,scale=se,df=n-1)
9.89/np.sqrt(20)

from scipy.stats import norm
X = norm(loc=17,scale=2.21)
X.cdf(14)

X = norm(loc=0,scale=1)
X.cdf(-1.357)

# 문제 1
# # 본포 = 정규분포
se = 50/np.sqrt(100)
X_bar = norm(loc=500,scale=se)
1-X_bar.cdf(510)  

# 문제 2
from scipy.stats import binom
binom.pmf(2,20,0.05)
binom.cdf(2,20,0.05)
1-binom.cdf(2,20,0.05)

(20*19)/2 * 0.05**2 * (1-0.05)**18

# 문제 3
X=norm(loc=75,scale=8)
1- X.cdf(85)
X.sf(85)

X.sf(70)-X.sf(80)

X.ppf(0.9)

# 문제 4
# 귀무가설 : 한잔의 평균 온도가 75도이다
# 대립가설 : 한잔의 평균 온도가 75도가 아니다
from scipy.stats import t
data = [72.4,74.1,73.7,76.5,75.3,74.8,75.9,73.4,74.6,75.1]
len(data)
sam_std = np.std(data,ddof=1)
sam_mean = np.mean(data)
T_STATS = (sam_mean-75)/(sam_std/np.sqrt(10))
t.cdf(T_STATS,df=9)*2

X_bar = norm(loc=53-50,scale=8/np.sqrt(40))

z_stats = (53-50)/(8/np.sqrt(40))
(1 - norm(loc=0,scale=1).cdf(abs(z_stats))) * 2
