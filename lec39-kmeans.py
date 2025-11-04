import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/USArrests.csv')
print(df.head(2))

from sklearn.preprocessing import StandardScaler
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), columns = numeric_data.columns)
print(df_trans.head(2))

from sklearn.cluster import KMeans # K-평균 군집분석 불러오기
kmeans = KMeans(n_clusters = 4, random_state = 1)
labels = kmeans.fit_predict(df_trans)
print(labels)

# 팔머 펭귄 데이터를 bill_length, bill_depth 변수사용
# k-means 알고리즘으로 3개 그룹으로 분류해보세요.

from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
peng_length_depth = penguins[["bill_length_mm", "bill_depth_mm"]]

penguins_trans = pd.DataFrame(stdscaler.fit_transform(peng_length_depth), columns = peng_length_depth.columns)

kmeans = KMeans(n_clusters = 3, random_state = 1)
labels = kmeans.fit_predict(penguins_trans)
print(labels)

penguins["labels"] = labels
penguins

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=penguins,
                x='bill_depth_mm', y='bill_length_mm',
                hue='labels', palette='deep',
                edgecolor='w', s=50)


sns.scatterplot(data=penguins,
                x='bill_depth_mm', y='bill_length_mm',
                hue='species', palette='deep',
                edgecolor='w', s=50)


# k-means
import numpy as np

x1 = np.array([2, 2, 3, 5, 6, 6])
x2 = np.array([2, 4, 3, 5, 4, 5])

# 1) 랜덤하게 그룹의 중심을 선택함.
gr_1 = np.array([3, 6], dtype=float)
gr_2 = np.array([5, 2], dtype=float)

# 2) 각 데이터 포인트에서 각 그룹 중심까지 거리 계산
# gr_1까지 거리
gr_1_dist = np.sqrt((x1 - gr_1[0])**2 + (x2 - gr_1[1])**2)

# gr_2까지 거리
gr_2_dist = np.sqrt((x1 - gr_2[0])**2 + (x2 - gr_2[1])**2)

labels = (gr_1_dist > gr_2_dist) + 1

# 3) 그룹 중심점 업데이트
# gr_1 중심점 업데이트
gr_1[0] = x1[labels == 1].mean()
gr_1[1] = x2[labels == 1].mean()

# gr_2 중심점 업데이트
gr_2[0] = x1[labels == 2].mean()
gr_2[1] = x2[labels == 2].mean()

# 4) 2, 3번 반복

# 2) 각 데이터 포인트에서 각 그룹 중심까지 거리 계산
# gr_1까지 거리
gr_1_dist = np.sqrt((x1 - gr_1[0])**2 + (x2 - gr_1[1])**2)

# gr_2까지 거리
gr_2_dist = np.sqrt((x1 - gr_2[0])**2 + (x2 - gr_2[1])**2)

labels = (gr_1_dist > gr_2_dist) + 1

# 3) 그룹 중심점 업데이트
# gr_1 중심점 업데이트
gr_1[0] = x1[labels == 1].mean()
gr_1[1] = x2[labels == 1].mean()

# gr_2 중심점 업데이트
gr_2[0] = x1[labels == 2].mean()
gr_2[1] = x2[labels == 2].mean()

# 특징)
# 1. 적용가능변수가 2개로 한정된것은 아니다.
# 2. 변수 스케일링이 필요한 이유 => k-means 알고리즘 적용시 계산하는 거리의 각 변수별 영향력을 공정하게 만들기 위해서.
# 3. 이상치에 민감하게 반응=> 거리를 구할 떄 평균을 사용하기 때문. => 중심점 구할 때 중앙값을 사용
# 4. 초기 그룹 중심점에 따라 레이블 결과/수렴속도가 차이남.
# 5. 거리 함수가 다양할 수 있겠다.
# - 유클리디안
# - 맨해탄 거리
# 6. 

str = 'a'
str.repl

