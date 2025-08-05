# [연습문제] t-검정, F-검정 이해하기

import pandas as pd
import numpy as np



filename = "./tf_test/problem5_27.csv"
dat = pd.read_csv(filename)
dat

# 1. 귀무가설과 대립가설을 설정하시오.
# H0 = up - down != 0
# H1 = up - down = 0

# 2. 가설에 대한 검정 통계량과 유의 확률을 구하고, 귀무가설 기각 여부를 판단하시오.

dat['up'] - dat['down']
dat['diff'] = dat['up'] - dat['down']
dat['diff'].mean()
dat['diff'].std(ddof=1)


n= len(dat)
sample_mean = dat['diff'].mean()
sample_std = dat['diff'].std(ddof=1)
t_value = sample_mean / (sample_std / np.sqrt(n))
t_value


#2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


filename2 = "./tf_test/problem5_32.csv"
dat2 = pd.read_csv(filename2)
dat2.info()
dat2.head()
dat2

# 귀무가설과 대립가설을 설정하시오.
H0 = "성별에 따른 급여 차이가 없다."
H1 = "성별에 따른 급여 차이가 있다."

# 성별과 급여의 차이를 알아보기 위해 데이터를 시각화하시오.

plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Salary', data=dat2)
plt.title('성별에 따른 급여 분포')
plt.grid(True)
plt.show()


male_salary = dat2[dat2['Gender'] == 'Male']['Salary']
female_salary = dat2[dat2['Gender'] == 'Female']['Salary']

male_salary.mean()
female_salary.mean()
# 성별에 따른 급여의 차이를 시각화




# 그룹 간 분산의 차이가 있는지 판단하시오. 큰 그룹 분산 < 작은 그룹 분산 * 1.5 이면 같다고 판단
male_salary.var() < 1.5* female_salary.var()
#차이가 있다.
var_male = male_salary.var(ddof=1)
var_female = female_salary.var(ddof=1)

var_large = max(var_male, var_female)
var_small = min(var_male, var_female)

print(f"Male 분산: {var_male:.2f}")
print(f"Female 분산: {var_female:.2f}")
print(f"분산비: {var_large / var_small:.2f}")
# 등분산 여부 판단
equal_var = var_large < var_small * 1.5
if equal_var:
    print("등분산으로 간주할 수 있음")
else:
    print("등분산으로 간주할 수 없음")


# 가설에 대한 검정 통계량과 유의 확률을 구하고, 귀무가설 기각 여부를 판단하시오.
from scipy.stats import ttest_ind
t_statistic, p_value = ttest_ind(dat2)

# 4. 독립표본 t-검정
t_statistic, p_value = ttest_ind(male_salary, female_salary, equal_var=equal_var)

print(f"\nt-통계량: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# 5. 결론 도출
alpha = 0.05
if p_value < alpha:
    print("➤ 귀무가설을 기각합니다 → 성별에 따른 급여 평균에 유의미한 차이가 있습니다.")
else:
    print("➤ 귀무가설을 기각할 수 없습니다 → 성별에 따른 급여 평균 차이는 통계적으로 유의하지 않습니다.")


filename3 = "./tf_test/heart_disease.csv"
dat3 = pd.read_csv(filename3)
dat3
dat3.info()




#4
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
dat.head()

# 당뇨병이 있는 사람과 없는 사람 간의 평균 BMI에 차이가 있는지 검정하시오.

# 귀무가설과 대립가설을 설정하시오.
    # H0: 당뇨병 유무에 따른 BMI 평균 차이가 없다.
    # H1: 당뇨병 유무에 따른 BMI 평균 차이가 있다.

# 변수 간 관계를 알아보기 위해 데이터를 시각화하시오.
dat['Outcome'] = dat['Outcome'].astype('category')
plt.figure(figsize=(8, 6))
sns.boxplot(x='Outcome', y='BMI', data=dat)
plt.xlabel('당뇨병 유무 (0: 없음, 1: 있음)')
plt.ylabel('BMI')
plt.title('당뇨병 유무에 따른 BMI 분포')
plt.grid(True)
plt.show()




# 그룹 간 분산의 차이가 있는지 판단하시오. 큰 그룹 분산 < 작은 그룹 분산 * 1.5 이면 같다고 판단

# 당뇨병이 있는 그룹과 없는 그룹의 BMI 분산 계산
diabetes_no = dat[dat['Outcome'] == 0]['BMI']
diabetes_yes = dat[dat['Outcome'] == 1]['BMI']
var_no = diabetes_no.var(ddof=1)
var_yes = diabetes_yes.var(ddof=1)
print(f"당뇨병 없는 그룹 BMI 분산: {var_no:.2f}")
print(f"당뇨병 있는 그룹 BMI 분산: {var_yes:.2f}")
# 등분산 여부 판단  
equal_var = var_no < 1.5 * var_yes
if equal_var:
    print("등분산으로 간주할 수 있음")
else:
    print("등분산으로 간주할 수 없음")

# 가설에 대한 검정 통계량과 유의 확률을 구하고, 대립가설 채택 여부를 판단하시오.

from scipy.stats import ttest_ind
t_statistic, p_value = ttest_ind(diabetes_no, diabetes_yes, equal_var=equal_var)
print(f"\nt-통계량: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# 결론 도출
alpha = 0.05
if p_value < alpha:
    print("➤ 귀무가설을 기각합니다 → 당뇨병 유무에 따른 BMI 평균에 유의미한 차이가 있습니다.") 
else:
    print("➤ 귀무가설을 기각할 수 없습니다 → 당뇨병 유무에 따른 BMI 평균 차이는 통계적으로 유의하지 않습니다.")



# [연습문제] 균일분포와 지수분포
#1
from scipy.stats import expon
# λ = 0.5일 때 scale = 1/λ
lambda_val = 0.5
x = 2
# CDF 계산
prob = expon.cdf(x, scale=1/lambda_val)
print(f"P(X <= {x}) =", prob)

#2
lambda_val = 2
x=1

prob = expon.cdf(x, scale=1/lambda_val)
1 - prob

#3
lambda_val = 3
