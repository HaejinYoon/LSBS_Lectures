import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import seaborn as sns
#[연습문제] 분포 비교 방법 이해하기

#1

filename = "./tf_test/problem5_32.csv"
dat = pd.read_csv(filename)
dat
male_salary = dat[dat["Gender"] == "Male"]["Salary"]
Female_salary = dat[dat["Gender"] == "Female"]["Salary"]
male_percentiles = np.array([sp.percentileofscore(male_salary, value, kind='rank') for value in male_salary])
female_percentiles = np.array([sp.percentileofscore(Female_salary, value, kind='rank') for value in Female_salary])
theory_male = sp.norm.ppf(male_percentiles/100, np.mean(male_salary), np.std(male_salary))
theory_female = sp.norm.ppf(female_percentiles/100, np.mean(Female_salary), np.std(Female_salary))

plt.figure(figsize=(4, 3))
plt.scatter(theory_male, male_salary, color='red')
plt.scatter(theory_female, Female_salary, color='blue')
plt.plot([0, 9000], [0, 9000], 'k')
plt.title('QQplot')
plt.xlim(4000)
plt.ylim(4000)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

f_statistic, p_value = f_oneway(male_salary, Female_salary)
print(f'F-statistic: {f_statistic}, p-value: {p_value}')

#2
filename3 = "./tf_test/heart_disease.csv"
dat3 = pd.read_csv(filename3)
dat3
dat3.info()
dat3 = dat3.dropna()

yes = dat3[dat3["target"] == 'yes']["chol"]
no = dat3[dat3["target"] == 'no']["chol"]
yes_percentiles = np.array([sp.percentileofscore(yes, value, kind='rank') for value in yes])
no_percentiles = np.array([sp.percentileofscore(no, value, kind='rank') for value in no])
theory_yes = sp.norm.ppf(yes_percentiles/100, np.mean(yes), np.std(yes))
theory_no = sp.norm.ppf(no_percentiles/100, np.mean(no), np.std(no))

plt.figure(figsize=(4, 3))
plt.scatter(theory_yes, yes, color='red')
plt.scatter(theory_no, no, color='blue')
plt.plot([0, 200], [0, 200], 'k')
plt.title('QQplot')
# plt.xlim(4000)
# plt.ylim(4000)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()
f_statistic, p_value = f_oneway(yes, no)
print(f'F-statistic: {f_statistic}, p-value: {p_value}')



#3
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
dat.head()

out_yes = dat[dat["Outcome"] == 1]["BMI"]
out_no = dat[dat["Outcome"] == 0]["BMI"]

yes_percentiles = np.array([sp.percentileofscore(out_yes, value, kind='rank') for value in out_yes])
no_percentiles = np.array([sp.percentileofscore(out_no, value, kind='rank') for value in out_no])
theory_yes = sp.norm.ppf(yes_percentiles/100, np.mean(out_yes), np.std(out_yes))
theory_no = sp.norm.ppf(no_percentiles/100, np.mean(out_no), np.std(out_no))

plt.figure(figsize=(4, 3))
plt.scatter(theory_yes, out_yes, color='red')
plt.scatter(theory_no, out_no, color='blue')
plt.plot([0, 60], [0, 60], 'k')
plt.title('QQplot')
# plt.xlim(4000)
# plt.ylim(4000)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

f_statistic, p_value = f_oneway(out_yes, out_no)
print(f'F-statistic: {f_statistic}, p-value: {p_value}')

#4
filename4 = "./tf_test/problem5_44.csv"
dat4 = pd.read_csv(filename4)
dat4
dat4.info()
data = dat4.iloc[:, 0].values

data.mean()
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import anderson, norm
result = sp.anderson(data, dist='norm') # Anderson-Darling 검정 수행
print('검정통계량',result[0], '임계값:',result[1], '유의수준:',result[2])

ecdf = ECDF(data)
x = np.linspace(min(data), max(data))
y = ecdf(x)
plt.scatter(x, y)
plt.title("Estimated CDF vs. CDF")
k = np.arange(min(data), max(data), 0.1)
plt.plot(k, norm.cdf(k, loc=np.mean(data),
scale=np.std(data, ddof=1)), color='k')
plt.show()


# [연습문제] 확률 심화

# 문제1번


# 문제4번

