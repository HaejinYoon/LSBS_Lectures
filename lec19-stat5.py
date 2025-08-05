import numpy as np
from scipy.stats import ttest_1samp


sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]

t_statistic, p_value = ttest_1samp(sample, popmean=10, alternative='two-sided') # 양측 검정

t_statistic
p_value

# 유의수준 5% 하에서 통계적 판단은?
# 기각하지 못한다

import pandas as pd

sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]
gender = ["Female"]*7 + ["Male"]*5

my_tab2 = pd.DataFrame({"score": sample, "gender": gender})
print(my_tab2)

from scipy.stats import ttest_ind

male_score = my_tab2[my_tab2['gender'] == 'Male']["score"]
female_score = my_tab2[my_tab2['gender'] == 'Female']["score"]
ttest_ind(male_score, female_score, equal_var=True, alternative='greater') #greater의 기준은 첫 번째 인자


t_statistic, p_value = ttest_ind(male_score, female_score, # 단측 검정 (큰 쪽)
equal_var=True, alternative='greater')
print("t-statistic:", t_statistic, "p-value:", p_value)


import numpy as np

before = np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15])
after = np.array([10.52, 14.83, 13.03, 16.46, 10.84, 12.45])


x = after - before

ttest_1samp(x, popmean=0, alternative='greater')