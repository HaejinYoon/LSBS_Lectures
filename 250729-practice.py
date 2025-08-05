import numpy as np

#[연습문제]기대값과 분산
#1
X = np.array([1, 2, 3])
P = np.array([0.2, 0.5, 0.3])

E = np.sum(X * P )
E

#2
V = np.sum(((X - E)**2) * P)
V

#3
Y = 2*X +3
Y

E2 = np.sum(Y * P )
E2

#4
V2 = np.sum(((Y - E2)**2) * P)
V2


#5
X3 = np.array([0, 1, 2, 3])
P3 = np.array([0.1, 0.3, 0.4, 0.2])

E3 = np.sum(X3 * P3)
E3
V3 = np.sum(((X3 - E3)**2) * P3)
V3

#8
EX8 = 5
EY8 = 3
#E(2X-Y+4) = 11

#9

#10
#p = 0.3
X10 = np.array([1, 2, 3])
P10 = np.array([0.3, 0.3, 0.4])
E10 = np.sum(X10 * P10 )
E10
V10 = np.sum(((X10 - E10)**2) * P10)
V10

#11
X11 = np.array([1, 2, 4])
X112 = X11**2
P11 = np.array([0.2, 0.5, 0.3])
E11 = np.sum(X11 * P11 )
E11
E112 = np.sum(X112 * P11 )
E112
V11 = np.sum(((X11 - E11)**2) * P11)
V11



#[연습문제]확률질량함수, 확률밀도함수, 누적분포함수

#1
values = np.array([0, 1, 2, 3])
probs = np.array([0.1, 0.3, 0.4, 0.2])

E_X = np.sum(values * probs)


















