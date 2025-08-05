import numpy as np

# 확률변수 X의 가능한 값과 그에 대한 확률
#0 - 지지하지 않는다.
#1 - 지지한다.
values = np.array([0, 1, 2])
probs = np.array([0.36, 0.48, 0.16])

# 예시: 10000개의 난수 생성
X = np.random.choice(values, size=10, p=probs)
X.sum()
X.mean()
X

exp_X = np.sum(values * probs)
exp_X

# 생성된 확률분포 확인
unique, counts = np.unique(X, return_counts=True)
empirical_probs = counts / len(X)

# 출력
for val, p_emp in zip(unique, empirical_probs):
    print(f"X = {val} → 경험적 확률: {p_emp:.3f}")

values = np.array([1, 2, 3, 4])
probs = np.array([0.1, 0.3, 0.2, 0.4])

# 예시: 10000개의 난수 생성
X = np.random.choice(values, size=300, p=probs)
X.sum()
X.mean()
X

exp_X = np.sum(values * probs)
exp_X

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False 
# 값과 확률
values = np.array([1, 2, 3, 4])
probs = np.array([0.1, 0.3, 0.2, 0.4])

# 표본 추출
X = np.random.choice(values, size=300, p=probs)

# 이론적 기대값
exp_X = np.sum(values * probs)
print(f"이론적 기대값: {exp_X:.2f}")
print(f"표본 평균: {X.mean():.2f}")

# 히스토그램 그리기
plt.figure(figsize=(6, 4))
plt.hist(X, bins=np.arange(0.5, 4.6, 1), align='mid', rwidth=0.6, color='skyblue', edgecolor='black', density=True)
plt.xticks(values)
plt.xlabel("value")
plt.ylabel("count")
plt.title("hist")

# 이론적 확률 추가 (빨간 수직선)
plt.vlines(values, ymin=0, ymax=probs, color='red', linestyle='-', linewidth=2, label='이론적 확률')

plt.axvline(exp_X, color='red', linestyle='--', label=f'기대값 E[X] = {exp_X:.2f}')
plt.axvline(X.mean(), color='green', linestyle='--', label=f'표본 평균 = {X.mean():.2f}')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()



values = np.array([1, 2, 3, 4])
probs = np.array([0.1, 0.3, 0.2, 0.4])
E_X = np.sum(values * probs)
X = np.random.choice(values, size=300, p=probs)
X.var(ddof=1)
np.sum((X - X.mean())**2) / (300-1)
# (X - E_X)**2 # 확률변수

# 확률변수가 갖는 값과 확률은?
np.sum((values - E_X)**2 * probs)




values = np.array([1, 2, 3, 4])
probs = np.array([0.1, 0.3, 0.2, 0.4])
E_X = np.sum(values * probs)
X = np.random.choice(values, size=500, p=probs)

# np.sum((X - X.mean())**2) / (300-1)
# np.sum((X - X.mean())**2) / (300)

var_n1 = X.var(ddof=1)
var_n =X.var()

# 위의 var_n1과 var_n을 1000번씩 발생 시킨 후, 각각의 히스토그램을 그려보세요.
# 각 히스토그램의 중심점(표본평균)을 초록색 막대기로 표시
# 이론 분산값(1.09)을 빨간 막대기로 표시

import numpy as np
import matplotlib.pyplot as plt

# 값과 확률
values = np.array([1, 2, 3, 4])
probs = np.array([0.1, 0.3, 0.2, 0.4])

# 이론 기대값 및 분산
E_X = np.sum(values * probs)
E_X2 = np.sum((values**2) * probs)
theoretical_var = E_X2 - E_X**2  # = 1.09

# 분산들을 저장할 리스트
var_n1_list = []
var_n_list = []

n1 = np.array(var_n1_list)
n = np.array(var_n_list)

n - n1

# 시뮬레이션 (1000번)
for _ in range(1000):
    X = np.random.choice(values, size=30, p=probs)
    var_n1_list.append(X.var(ddof=1))  # 표본분산 (unbiased)
    var_n_list.append(X.var(ddof=0))   # 모분산 (biased)

# 히스토그램 시각화
plt.figure(figsize=(12, 5))

# 불편분산 히스토그램
plt.subplot(1, 2, 1)
plt.hist(var_n1_list, bins=30, color='skyblue', edgecolor='black')
plt.axvline(np.mean(var_n1_list), color='green', linestyle='--', linewidth=2, label=f'표본평균: {np.mean(var_n1_list):.3f}')
plt.axvline(theoretical_var, color='red', linestyle='-', linewidth=2, label='이론 분산: 1.09')
plt.title("불편분산 (ddof=1)")
plt.xlabel("분산값")
plt.ylabel("도수")
plt.legend()
plt.grid(True, alpha=0.4)

# 모분산 히스토그램
plt.subplot(1, 2, 2)
plt.hist(var_n_list, bins=30, color='lightcoral', edgecolor='black')
plt.axvline(np.mean(var_n_list), color='green', linestyle='--', linewidth=2, label=f'표본평균: {np.mean(var_n_list):.3f}')
plt.axvline(theoretical_var, color='red', linestyle='-', linewidth=2, label='이론 분산: 1.09')
plt.title("모분산 (ddof=0)")
plt.xlabel("분산값")
plt.ylabel("도수")
plt.legend()
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()


from scipy.stats import uniform

# X ~균일 분포 U(a, b)
a = 2 # 하한
b = 4 # 상한
X = uniform(loc=a, scale=b - a)
X.mean() # (a + b) / 2
X.var() # (b -a)^2 / 12
x = X.rvs(size=100)
x

# P(X <= 3.5)의 값
X.cdf(3.5)

# (3.2 - 2.1) * 0.5
#P(2,1 < X <= 3.2)
X.cdf(3.2) - X.cdf(2.1)
X.cdf(3.1)



# 문제. 지수분포를 따르는 확률변수 X를 만들어보세요
# scale = 0.5
# X ~ exp(theta = 0.5)

from scipy.stats import expon

X1 = expon(scale=0.5)
X2 = expon(scale=3)

x1 = X1.rvs(size=100)
x2 = X2.rvs(size=100)
sum(x1 <= 2)
sum(x2 <= 2)

# 1. 확률밀도함수(pdf)를 -1에서 10사이의 범위에서 그려보세요

# x 값 범위
x = np.linspace(0, 10, 500)
pdf = X1.pdf(x)

pdf2 = X2.pdf(x)
# 그래프
plt.figure(figsize=(7, 4))
plt.plot(x, pdf, color='blue', label='PDF of Exp(θ=0.5)')
plt.plot(x, pdf2, color='red', label='PDF of Exp(θ=3)')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='gray', linestyle='--')
plt.title("지수분포 확률밀도함수 (PDF)")
plt.xlabel("x")
plt.ylabel("밀도 (f(x))")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

X1.cdf(6) - X1.cdf(2)
X2.cdf(6) - X2.cdf(2)

#평균
X1.mean()
X2.mean()

X1.var()
X2.var()

X1.ppf(0.2)
X2.ppf(0.2)



# 정규분포(loc = 0, scale = 1) 확률변수 X1을 만들어보세요.
# 1. 평균계산
# 2. 분산계산
# 3. pdf 그려보기 - 특징유추

from scipy.stats import norm

# 정규분포 N(0, 1)
X1 = norm(loc=0, scale=1)

X1.mean()
X1.var()


# x 범위 설정
x = np.linspace(-4, 4, 500)
pdf = X1.pdf(x)

# PDF 그래프
plt.figure(figsize=(8, 4))
plt.plot(x, pdf, color='darkblue', label='PDF of N(0, 1)')
plt.axvline(0, color='red', linestyle='--', label='mean = 0')
plt.axvline(1, color='gray', linestyle='--', alpha=0.6)
plt.axvline(-1, color='gray', linestyle='--', alpha=0.6)
plt.axvline(2, color='gray', linestyle='--', alpha=0.6)
plt.axvline(-2, color='gray', linestyle='--', alpha=0.6)

plt.title("표준 정규분포의 확률밀도함수 (PDF)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 정규분포(loc = 2, scale = 3) 확률변수 X2를 만들어보세요.

# 4. loc, scale 패러미터의 효과 알아내기
# 5. P(X <= 3) 값은?
# 6. P(1.5 < X < 5) 값은?
# 7. 표본 300개를 뽑아서 히스토그램을 그린 후, pdf와 겹쳐서 그려보기
# 8. X2에서 나올 수 있는 값들 중, 상위 10%에 해당하는 값은?

X2 = norm(loc=2, scale=3)
X2.cdf(3)
X2.cdf(5) - X2.cdf(1.5)

x2 = np.linspace(-2, 6, 300)
pdf2 = X2.pdf(x2)

x2 = X2.rvs(size=300)
pdf2 = X2.pdf(x2)
plt.figure(figsize=(8, 4))
# plt.hist(x2, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='표본 히스토그램')
# 이론적 PDF
plt.plot(x, pdf, color='darkblue', label='PDF of N(0, 1)')
plt.plot(x2, pdf2, color='orange', label='PDF of N(2, 3)')
# plt.axvline(0, color='red', linestyle='--', label='mean = 0')
# plt.axvline(1, color='gray', linestyle='--', alpha=0.6)
# plt.axvline(-1, color='gray', linestyle='--', alpha=0.6)
# plt.axvline(2, color='gray', linestyle='--', alpha=0.6)
# plt.axvline(-2, color='gray', linestyle='--', alpha=0.6)

plt.title("표준 정규분포의 확률밀도함수 (PDF)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

X2.ppf(0.9)



# 표준 정규분포
#-1에서 1사이 값이 나올 확률
X1 = norm(loc=0, scale=1)

X1.cdf(1) - X1.cdf(-1)
X1.cdf(2) - X1.cdf(-2)

# X ~ N(2, 3^2)에서 (2 -3, 2+3) 사이에 값이 나올 확률

X2.cdf(5) - X2.cdf(-1)
X2.cdf(8) - X2.cdf(-4)

values = np.array([0, 1, 2, 3])
probs = np.array([1/8, 1/8, 1/4, 1/2])

exp_X = np.sum(values * probs)
# 확률변수 X의 가능한 값과 그에 대한 확률
exp_X2 = np.sum((values**2) * probs)
exp_X2 - exp_X**2














































