import pandas as pd
import numpy as np
odors = ['Lavender', 'Rosemary', 'Peppermint']
minutes_lavender = [10, 12, 11, 9, 8, 12, 11, 10, 10, 11]
minutes_rosemary = [14, 15, 13, 16, 14, 15, 14, 13, 14, 16]
minutes_peppermint = [18, 17, 18, 16, 17, 19, 18, 17, 18, 19]
anova_data = pd.DataFrame({
'Odor': np.repeat(odors, 10),
'Minutes': minutes_lavender + minutes_rosemary + minutes_peppermint
})

anova_data

# ANOVA 검정
# H0: mu_L = mu_R = mu_P
anova_data.groupby(['Odor']).describe()
# 3 표준편차까지는 커버가 된다
# L, P가 차이가 너무 난다.

from scipy.stats import f_oneway

 # 각 그룹의 데이터를 추출
lavender = anova_data[anova_data['Odor'] == 'Lavender']['Minutes']
rosemary = anova_data[anova_data['Odor'] == 'Rosemary']['Minutes']
peppermint = anova_data[anova_data['Odor'] == 'Peppermint']['Minutes']

 # 일원 분산분석(One-way ANOVA) 수행
f_statistic, p_value = f_oneway(lavender, rosemary, peppermint)
print(f'F-statistic: {f_statistic}, p-value: {p_value}')

1 - (1 - 0.05) **3

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(
endog=anova_data['Minutes'],
groups=anova_data['Odor'],
alpha=0.05)
print(tukey)

from scipy.stats import norm

lavender_mu = 9
rosemary_mu = 7
papermint_mu = 6
X = norm(loc = 0 , scale=2)
X.rvs(10)

x_1 = lavender_mu + X.rvs(10)
x_1
x_r = rosemary_mu + X.rvs(10)
x_r
x_p = papermint_mu + X.rvs(10)
x_p


#p20

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Minutes ~ C(Odor)',data=anova_data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
# 267 = SSG,  32.9 = SSE

model.resid

import matplotlib.pyplot as plt
plt.scatter(model.fittedvalues, model.resid)
plt.show()

import scipy.stats as sp
W, p = sp.shapiro(model.resid)
print(f'검정통계량: {W:.3f}, 유의확률: {p:.3f}')

#bartlet
from scipy.stats import bartlett
groups = ['Lavender', 'Rosemary', 'Peppermint']
grouped_residuals = [model.resid[anova_data['Odor'] == group] for group in groups]
test_statistic, p_value = bartlett(*grouped_residuals)
print(f"검정통계량: {test_statistic}, p-value: {p_value}")
