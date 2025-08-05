import numpy as np
import pandas as pd

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5])  # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"])  # 문자형 벡터 생성
c = np.array([True, False, True, True])  # 논리형 벡터 생성
print("Numeric Vector:", a)


type(a)
a[2:4]
a[0]=[1,3,5]

d = np.array(["q", 2])
d

a + 3
a * 20

b = np.array([6, 7, 8, 9, 10])

a + b
a**2

a.cumsum()

np.arange(1, 10, 0.5)

np.arange(4, 10, step=0.5)
a= np.arange(4, 10, step=0.5)
len(a)
vec_a = np.arange(7, 1000, 7)
sum(vec_a)
vec_a.sum()
vec_a.cumsum()


# pip install palmerpenguins
# 데이터로드
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins
penguins = penguins.dropna()

vec_m = np.array(penguins["body_mass_g"])
vec_m.shape
vec_m.max()
vec_m.min()
vec_m.argmax()
vec_m[163]
vec_m.argmin()
vec_m.mean()

# 평균이 4.2kg이라면 평균몸무게보다 작은 펭귄들은 몇마리인가
sum(vec_m<4200)
# 3kg 이상인 펭귄
sum(vec_m>=3000)

sum((vec_m>=3000) & (vec_m<4200)) 