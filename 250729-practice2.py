import numpy as np

#[연습문제]확률질량함수, 확률밀도함수, 누적분포함수

#1
values = np.array([0, 1, 2, 3])
probs = np.array([0.1, 0.3, 0.4, 0.2])

E_X = np.sum(values * probs)
E_X

cdf_at_2 = probs[values <= 2].sum()
print("F(2) =", cdf_at_2)

#2
values = np.array([0, 1, 2, 3])

#PX=2 : 0.8-0.4 = 0.4, PX>2 : 1.0 -0.8 = 0.2

#5

#6
values = np.array([1, 2, 3])
probs = np.array([0.2, 0.5, 0.3])

E_X6 = np.sum(values * probs)
E_X6
F_2 = probs[0] + probs[1]
F_2

#8
# 4/5 - 1/5 = 3/5

#9
values = np.array([0, 1, 2, 3])
probs = np.array([0.1, 0.2, 0.4, 0.3])

(0.1 + 0.2 + 0.4 + 0.3) - (0.1 + 0.2)

#10
0.1
0.2
0.4
0.3
(0.1 + 0.2 + 0.4) - (0.1 + 0.2)