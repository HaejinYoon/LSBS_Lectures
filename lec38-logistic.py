import pandas as pd
import numpy as np
admission_data = pd.read_csv("./data/admission.csv")
print(admission_data.head())
print(admission_data.shape)

p_hat = admission_data['admit'].mean()
print(np.round(p_hat / (1 - p_hat), 3))

unique_ranks = sorted(admission_data['rank'].unique())
print(unique_ranks)

grouped_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean'))
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
print(grouped_data)

odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)

import pandas as pd
import numpy as np
admission_data = pd.read_csv("./data/admission.csv")
admission_data

print(admission_data.shape)

# 입학이 허가될 확률의 오즈
p_hat = admission_data['admit'].mean() # 대학에 입학할 확률
print(np.round(p_hat / (1 - p_hat), 3))

'''
입학할 확률에 대한 오즈는 0.465
=> 입학에 실패할 확률의 46%
=> 즉, 오즈가 1을 기준으로 낮을 경우, 해당 확률이 발생하기 어렵다.
'''

# 범주형 변수를 사용한 오즈 계산
unique_ranks = sorted(admission_data['rank'].unique())
print(unique_ranks)

grouped_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')) # 입학할 확률 

grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit']) # 입학할 오즈 => 1을 기준으로 

'''
1등급 학생들이 입학에 성공할 확률은 입학에 실패할 확률보다 18% 더 높음

나머지 등급의 학생들은 입학할 확률이 입학에 실패할 확률보다 더 낮다.

* 오즈 최솟값 = 0
* 오즈 최댓값 = 무한대 (일어날 확률이 100%인 경우)

* 오즈가 1보다 더 크면 성공할 확률이 실패할 확률보다 더 크다는 것 
'''

# 로그오즈 => -무한대 ~ 무한대까지 선형적으로 연결 가능 


# 로지스틱 회귀계수 예측

odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)

import statsmodels.formula.api as smf
model = smf.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary())

# y_hat = 0.6327 - 0.5675 * X
# 1.5등급 학생의 입학 확률은?
odds_15=np.exp(0.6327-0.5675 * 1.5)
odds_15 / (odds_15 + 1)

# 시각화
import matplotlib.pyplot as plt
# 회귀선 계산
x = odds_data['rank']
y = odds_data['log_odds']
coefficients = np.polyfit(x, y, 1)
poly_eq = np.poly1d(coefficients) # 넘파이 회귀직선 계수 계산
# 회귀선 그리기
plt.scatter(odds_data['rank'], odds_data['log_odds'], label='Data Points')
plt.plot(x, poly_eq(x), color='red', label='Regression Line')
# 그래프 레이블 설정
plt.xlabel('Rank')
plt.ylabel('Log Odds')
plt.title('Scatter Plot with Regression Line')
plt.legend()
plt.show()


# 회귀분석 수행하기 
admission_data = pd.read_csv("./data/admission.csv")
import statsmodels.formula.api as smf
admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')

model = smf.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()
print(model.summary())

# gpa 계수가 0.7753 => 무슨의미?
np.exp(0.7753)
# 2.17 => x(gpa)가 1 증가할 시, 오즈가 2.17배가 된다
np.exp(-0.5614)