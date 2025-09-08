from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
import pandas as pd
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins_a=penguins.loc[penguins["species"] != "Adelie",]


x = penguins_a[["bill_length_mm", "bill_depth_mm"]]
y = penguins_a["species"]

# x, y 평면상에 좌표를 찍는데
# bill_length, bill_depth에 점을 표시
# 단, 겐투는 사각형, 친스트랩은 동그라미로 표시
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=penguins_a,
    x="bill_length_mm", 
    y="bill_depth_mm",
    style="species",   # 종에 따라 모양 다르게
    hue="species",     # 종에 따라 색도 다르게
    s=80,              # 점 크기
    edgecolor="k"
)
plt.title("Penguins (Gentoo vs Chinstrap)")
plt.grid(True)
plt.show()


x = penguins_a[["bill_length_mm", "bill_depth_mm"]]
y = penguins_a["species"]

left_x = x.loc[x["bill_length_mm"] <= 50, : ]
right_x = x.loc[x["bill_length_mm"] > 50, : ]
left_y = y.loc[x["bill_depth_mm"] <= 50]
right_y = y.loc[x["bill_depth_mm"] > 50]

#지니인덱스
# 특정 노드에 Gentoo만 있는 경우!
# 특정 노드에 Chinstrap만 있는 경우!
# 1 - (p_g^2 + p_c^2)
# 1 - (1**2 + 0**2) = 1 - 1 = 0
# 1 - (0**2 + 1**2) = 1 - 1 = 0

# 특정 노드에 100개의 데이터 -> Gentoo 50개, Chinstrap 50개
# 1 - (p_g^2 + p_c^2)
# 1 - (0.5**2 + 0.5**2) = 0.5

# bill_depth 16.5기준으로 지니 인덱스 구해보기

#==========================================================================================
# 내가 한거
#==========================================================================================

upper_y = y.loc[x["bill_depth_mm"] >= 16.5]
lower_y = y.loc[x["bill_depth_mm"] < 16.5]

p_g_u = sum(upper_y == "Gentoo")
p_c_u =sum(upper_y == "Chinstrap")
1 - ((p_g_u / len(upper_y))**2 + (p_c_u / len(upper_y))**2)

p_g_l = sum(lower_y == "Gentoo")
p_c_l = sum(lower_y == "Chinstrap")
1 - ((p_g_l / len(lower_y))**2 + (p_c_l / len(lower_y))**2)

#==========================================================================================
# 지피티가 해준거
#==========================================================================================

def gini_of_labels(labels: pd.Series) -> float:
    """단일 노드의 지니 불순도: 1 - Σ p_c^2"""
    p = labels.value_counts(normalize=True)
    return 1.0 - (p**2).sum()

# 1) 분할 마스크 (bill_depth_mm ≤ 16.5 vs > 16.5)
thr = 16.5
mask_left = x["bill_depth_mm"] <= thr   # Left: 깊이 ≤ 16.5
mask_right = ~mask_left                 # Right: 깊이 > 16.5

# 2) 각 노드 지니
gini_left = gini_of_labels(y[mask_left])
gini_right = gini_of_labels(y[mask_right])

# 3) 가중 평균(분할 지니)
n_total = len(y)
w_left = mask_left.sum() / n_total
w_right = mask_right.sum() / n_total
gini_split = w_left * gini_left + w_right * gini_right

# 4) 참고용: 분할 전(부모) 지니와 분할 후 클래스 분포 출력
gini_parent = gini_of_labels(y)
ct = pd.crosstab(y, pd.Series(mask_left, name=f"bill_depth_mm <= {thr}"))

print(f"Parent Gini: {gini_parent:.4f}")
print(f"Left  (<= {thr}) size={mask_left.sum()}  Gini: {gini_left:.4f}")
print(f"Right (>  {thr}) size={mask_right.sum()} Gini: {gini_right:.4f}")
print(f"Weighted Gini after split@{thr}: {gini_split:.4f}\n")
print("Class counts by side:")
print(ct)

#==========================================================================================
#
#==========================================================================================

from sklearn.tree import DecisionTreeClassifier
import numpy as np

dct = DecisionTreeClassifier(criterion="gini")
dct.get_params()

dct_params = {'max_depth' : np.arange(1, 8),
                  'ccp_alpha': np.linspace(0, 1, 5)}
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)

# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                              param_grid=dct_params, 
                              cv = cv, 
                              scoring='accuracy')
dct_search.fit(x, y)
dct_search.best_params_

dct_search.predict(x)
dct_search.predict_proba(x)