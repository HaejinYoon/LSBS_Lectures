import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

X = bernoulli(p=0.3)
X.mean()
X.var()

#이항분포
from scipy.stats import binom

X = binom(n=5, p=0.3)
X.mean()
X.pmf(0)
X.pmf(1)
X.pmf(2)
X.pmf(3)
X.pmf(4)
X.pmf(5)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False 

# 이항분포 설정
n = 5
p = 0.3
X = binom(n=n, p=p)

# 확률질량함수 계산
x = np.arange(0, n + 1)
pmf = X.pmf(x)

# 시각화
plt.figure(figsize=(8, 5))
markerline, stemlines, baseline = plt.stem(x, pmf, basefmt=" ")
plt.title(f'이항분포 (n={n}, p={p})의 확률질량함수')
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.xticks(x)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

from scipy.special import comb
#nCr 계산
comb(5,3)
# P(X=3)
comb(5, 3) * (0.3**3) * (0.7**2)

# 앞면이 나올 확률이 0.6인 동전을 10번 던져서 앞면이 5번 나올 확률은?
# X ~ B(n=10, p=0.6)
# P(X = 5)=?

comb(10, 5)*(0.6**5)*(0.4**5)

X = binom(n=5, p=0.3)
X.mean()
X.var()

Y = bernoulli(p=0.3)
sum(Y.rvs(size=5)) # 이항분포 B(5, 0.3)이 된다


# [연습문제] 베르누이분포와 이항분포, 포아송분포

#1
# 0.6, 0.4

#2
# E(X) = 0.8,
X = bernoulli(p=0.8)
X.mean()
X.var()

#3
n = 3
p = 0.2
X = binom(n=n, p=p)
X.pmf(1)
comb(3, 0.2) *(0.2)*(0.8**2)

#4
n = 7
p = 0.5
# 누적 확률 P(X <= 4)
cdf_4 = binom.cdf(4, n, p)

# P(X >= 5) = 1 - P(X <= 4)
prob = 1 - cdf_4
print(f"P(X >= 5) = {prob:.4f}")


#5
n = 6
p = 0.3
X = binom(n=n, p=p)
X.mean()
X.var()

#6
from scipy.stats import poisson

# λ = 4
mu = 2

# P(X = 3)
prob = poisson.pmf(3, mu)
print(f"P(X = 3) = {prob:.4f}")

#7
from scipy.stats import poisson

# λ = 4
mu = 4

# P(X = 3)
prob = poisson.cdf(2, mu)
print(f"P(X <= 2) = {prob:.4f}")

#8
prob = poisson(5)
prob.mean()
prob.var()

#9
X = binom(n=10, p=0.6)
X.cdf(4) - X.cdf(3)

#10
from scipy.stats import poisson

# λ = 3.5
mu = 3.5

# P(X <= 2)
prob = poisson.cdf(2, mu)
print(f"P(X <= 2) = {prob:.4f}")


#11
# 베르누이 분포 설정
p = 0.7
X = bernoulli(p)

# x값 (가능한 값은 0과 1)
x = [0, 1]
pmf = X.pmf(x)

# 시각화
plt.figure(figsize=(6, 4))
plt.bar(x, pmf, width=0.3, color='orange', edgecolor='black')
plt.xticks(x)
plt.title(f'베르누이 분포 (p={p})의 확률질량함수')
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, 1.1)
plt.show()


#15
visits = [0, 1, 2, 0, 3, 1, 4, 2, 2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 4, 2] # 평균이 기대값과 비슷하다는 것에 의하여
lambda_hat = np.mean(visits)
lambda_hat
X = poisson(lambda_hat)
X.pmf(0) # 한명도 방문하지 않을 확률

# 하루동안 5명 이상 방문할 확률은?
1-X.cdf(4)
