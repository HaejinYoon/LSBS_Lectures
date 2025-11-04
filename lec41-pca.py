import pandas as pd
from palmerpenguins import load_penguins

# 1. 데이터 불러오기 및 전처리
penguins = load_penguins()
penguins = penguins.dropna()
penguins = penguins[penguins["species"] == "Adelie"]
df = penguins[["bill_length_mm", 
               "bill_depth_mm", 
               "body_mass_g"]]
df

import seaborn as sns
sns.pairplot(df, kind="scatter",
             diag_kind="hist")

df.corr()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_data = pd.DataFrame(scaled_data, columns=df.columns)

scaled_data

from sklearn.decomposition import PCA
pca = PCA(n_components=3)

pca_array = pca.fit_transform(scaled_data)

my_pca = pd.DataFrame(pca_array,
                    index = scaled_data.index,
                    columns=["pc1", "pc2", "pc3"])
my_pca
my_pca.corr()
my_pca.cov()

my_pca["pc1"].var(ddof=1)
my_pca["pc2"].var(ddof=1)
my_pca["pc3"].var(ddof=1)
#========================================
# 다중공선성 문제를 없앴다
#========================================

pca.explained_variance_.round(3) # 아이겐벨류
x_to_pc = pd.DataFrame(pca.components_,
                    columns=scaled_data.columns,
                    index=['pca1','pca2','pca3']).round(3)
x_to_pc
pca.components_

# 어떻게 PC들을 만들었나?
# 1) 스케일된 데이터의 공분산행렬 계산
# 2) 행렬을 분해(아이겐벨류 디컴포지션) 적용
# => 아이겐벨류, 아이겐벡터 두 개 결과값이 나옴

scaled_data.cov(ddof=1)

from numpy import linalg
import numpy as np
eig_values, eig_vectors = linalg.eig(scaled_data.cov(ddof=1))
np.sqrt(eig_values[0] / eig_values[2])
eig_vectors

# PC1 = 0.548 * 부리길이 ....

my_pca
scaled_data.iloc[0,:]

import matplotlib.pyplot as plt
def biplot(score, coeff, pcax, pcay, labels=None):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[pca1, i], coeff[pca2, i],color='r',alpha=0.5)
        if labels is None:
            plt.text(coeff[pca1, i]* 1.15, coeff[pca2, i] * 1.15,
            "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[pca1, i]* 1.15, coeff[pca2, i] * 1.15,
            labels[i], color='g', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid()

biplot(pca_array, pca.components_, 1, 2,
labels=scaled_data.columns)
plt.show()

x_to_pc


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
# 1. 불균형 데이터 생성 (정상: 0, 이상: 1)
X, y = make_classification(n_samples=1000, n_features=5, 
                           n_informative=2, n_redundant=0, 
                           weights=[0.95, 0.05], # 95% : 5%
                           random_state=42)
print("클래스 분포:", pd.Series(y).value_counts())
# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 3. 로지스틱 회귀 모형 적합
model = LogisticRegression()
model.fit(X_train, y_train)
# 4. 예측
y_pred = model.predict(X_test)
# 5. 평가
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("혼동 행렬:\n", cm)
print(f"정확도(Accuracy): {acc:.3f}")
print(f"민감도(Recall): {recall:.3f}")





from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# ----------------------------
# [오버샘플링] RandomOverSampler 적용
# ----------------------------

ros = RandomOverSampler(random_state=42)

X_res_over, y_res_over = ros.fit_resample(X_train, y_train)

model_over = LogisticRegression()
model_over.fit(X_res_over, y_res_over)
y_pred_over = model_over.predict(X_test)

print("\n[RandomOverSampler 결과]")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_over))
print("정확도:", accuracy_score(y_test, y_pred_over))
print("민감도:", recall_score(y_test, y_pred_over))

# ----------------------------
# [언더샘플링] RandomUnderSampler 적용
# ----------------------------

under = RandomUnderSampler(random_state=42)
X_res_under, y_res_under = under.fit_resample(X_train, y_train)
model_under = LogisticRegression()
model_under.fit(X_res_under, y_res_under)
y_pred_under = model_under.predict(X_test)

print("\n[RandomUnderSampler 결과]")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_under))
print("정확도:", accuracy_score(y_test, y_pred_under))
print("민감도:", recall_score(y_test, y_pred_under))