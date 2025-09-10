import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/open/train.csv')
test_df = pd.read_csv('./data/open/test.csv')
train_df["target"]
# train_df.info()
#train_df = train_df.select_dtypes(include=['number'])

# 칼럼 선택
num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop("target")
# cat_columns = train_df.select_dtypes(include=['object']).columns

# 결측치 채우기 (간단히 처리)
# train_df = train_df.dropna()
# from sklearn.impute import SimpleImputer
# freq_impute = SimpleImputer(strategy='most_frequent')
# mean_impute = SimpleImputer(strategy='mean')

# train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
# train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
# freq_impute.statistics_

# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# onehot = OneHotEncoder(handle_unknown='ignore', 
#                        sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")

# train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])

# train_df_all = pd.concat([train_df_num, train_df_cat], axis = 1)

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df_num
y_train = train_df['target']

from sklearn.linear_model import ElasticNet
# alpha == lamda (패널티 가중치 패러미터)

elastic = ElasticNet()
elasticnet = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}
# 파라미터 확인 
ElasticNet().get_params()

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)

# 그리드서치
elastic_search = GridSearchCV(estimator=elasticnet, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

elastic_search.fit(X_train, y_train)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(elastic_search.cv_results_))

# best prameter
print(elastic_search.best_params_)

# 교차검증 best score 
print(-elastic_search.best_score_)

# elastic = ElasticNet(alpha=0.1, 
#                      l1_ratio=0.75)
# elastic.fit(X_train, y_train)
# 최종 예측 
elastic.fit(X_train, y_train)

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
# test_df = test_df.select_dtypes(include=['number'])
# test_df = test_df.fillna(train_df.mean())

# test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
# test_df[num_columns] = mean_impute.transform(test_df[num_columns])

# test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])

# test_df_all = pd.concat([test_df_num, test_df_cat], axis = 1)

# 예측
#y_pred = elastic.predict(test_df)
y_pred = elastic_search.predict(test_df_num)
submit = pd.read_csv('./data/open/sample_submission.csv')
submit["target"]=y_pred.round(0)

# CSV로 저장
submit.to_csv('./data/open/elasticnet_grid.csv', index=False)


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ===== 1. 데이터 불러오기 =====
train = pd.read_csv("./data/open/train.csv")
test = pd.read_csv("./data/open/test.csv")
sample_submission = pd.read_csv("./data/open/sample_submission.csv")

# ===== 2. Feature/Target 분리 =====
X = train.drop(columns=["ID", "target"])
y = train["target"]

X_test = test.drop(columns=["ID"])  # 제출용 테스트 데이터 (ID 제외)

# ===== 3. 모델 정의 =====
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

# ===== 4. 학습 =====
rf.fit(X, y)

# ===== 5. 예측 =====
preds = rf.predict(X_test)

# ===== 6. 제출 파일 생성 =====
submission = sample_submission.copy()
submission["target"] = preds

submission.to_csv('./data/open/rf-dacon.csv', index=False)









# rf_grid_train_and_submit.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

# ===== 1) 데이터 로드 =====
train = pd.read_csv("./data/open/train.csv")
test = pd.read_csv("./data/open/test.csv")
sample_submission = pd.read_csv("./data/open/sample_submission.csv")

# ===== 2) 피처/타깃 분리 =====
# (안전하게 X_* 컬럼만 사용)
feature_cols = [c for c in train.columns if c.startswith("X_")]
X = train[feature_cols]
y = train["target"]
X_test = test[feature_cols]  # test에도 동일한 컬럼 사용

# ===== 3) 하이퍼파라미터 그리드 설정 =====
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 12, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}

base_rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

# ===== 4) 그리드서치 수행 =====
grid.fit(X, y)
print("Best Params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# ===== 5) 최적 파라미터로 전체 학습 =====
best_params = grid.best_params_
rf_best = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
)
rf_best.fit(X, y)

# (선택) 최적 모델로 교차검증 점수 재확인
cv_scores = cross_val_score(rf_best, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print("CV Accuracy (re-check) mean:", cv_scores.mean())
print("CV Accuracy (re-check) std :", cv_scores.std())

# ===== 6) 테스트 예측 & 제출파일 생성 =====
preds = rf_best.predict(X_test)

submission = sample_submission.copy()
submission["target"] = preds.astype(int)  # 정수형 보장
out_path = "./data/open/rf-dacon-grid.csv"
submission.to_csv(out_path, index=False)
print(f"✅ Saved submission to: {out_path}")

# (선택) 피처 중요도 저장 — 분석 참고용
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_best.feature_importances_
}).sort_values("importance", ascending=False)
fi_path = "./data/open/rf_feature_importances.csv"
fi.to_csv(fi_path, index=False)
print(f"ℹ️ Saved feature importances to: {fi_path}")






# rf_fast_random_search_submit.py
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

t0 = time.time()

# ===== 1) 데이터 로드 =====
train = pd.read_csv("./data/open/train.csv")
test = pd.read_csv("./data/open/test.csv")
sample_submission = pd.read_csv("./data/open/sample_submission.csv")

# ===== 2) 피처/타깃 분리 =====
feature_cols = [c for c in train.columns if c.startswith("X_")]
X = train[feature_cols]
y = train["target"]
X_test = test[feature_cols]

# ===== 3) 빠른 랜덤 서치 설정 (5~10분 목표) =====
# - 탐색 시 n_estimators는 200으로 고정(속도 개선)
# - 3-Fold CV 사용
base_rf = RandomForestClassifier(
    n_estimators=200,  # 탐색 단계에서는 낮게 고정
    random_state=42,
    n_jobs=-1
)

param_distributions = {
    "max_depth": [None, 10, 16, 24, 32],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, 0.7],
    # 부가적으로 조금의 가중치 변화도 시도해볼 수 있음(균등분포면 영향 작음)
    # "bootstrap": [True, False],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rnd = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_distributions,
    n_iter=60,              # 탐색 시도 수 (시간에 맞춰 조절 가능: 40~80)
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("▶ RandomizedSearchCV 탐색 시작...")
rnd.fit(X, y)
print("Best Params:", rnd.best_params_)
print("Best CV Accuracy (search):", rnd.best_score_)

# ===== 4) 최적 파라미터로 전체 재학습 =====
# - 최종 학습에서는 트리 수를 늘려 성능을 조금 더 끌어올림 (예: 400)
best_params = rnd.best_params_.copy()
rf_best = RandomForestClassifier(
    n_estimators=400,   # 최종 학습에서만 늘림(시간 여유가 더 있으면 500~800)
    random_state=42,
    n_jobs=-1,
    **best_params
)
print("▶ 최적 하이퍼파라미터로 전체 재학습...")
rf_best.fit(X, y)

# (선택) 교차검증으로 재확인
cv_scores = cross_val_score(rf_best, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print("CV Accuracy (re-check) mean:", cv_scores.mean())
print("CV Accuracy (re-check) std :", cv_scores.std())

# ===== 5) 테스트 예측 & 제출 파일 생성 =====
preds = rf_best.predict(X_test)
submission = sample_submission.copy()
submission["target"] = preds.astype(int)

out_path = "./data/open/rf-dacon-fast.csv"
submission.to_csv(out_path, index=False)
print(f"✅ Saved submission to: {out_path}")

# ===== 6) (선택) 피처 중요도 저장 — 분석/특성선택 참고용 =====
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_best.feature_importances_
}).sort_values("importance", ascending=False)
fi_path = "./data/open/rf_feature_importances_fast.csv"
fi.to_csv(fi_path, index=False)
print(f"ℹ️ Saved feature importances to: {fi_path}")

print(f"⏱️ Elapsed: {time.time() - t0:.1f} sec")



# rf_preprocess_fast_random_submit.py
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer

t0 = time.time()

# ===== 1) 데이터 로드 =====
train = pd.read_csv("./data/open/train.csv")
test = pd.read_csv("./data/open/test.csv")
sample_submission = pd.read_csv("./data/open/sample_submission.csv")

# ===== 2) 피처/타깃 분리 =====
feature_cols = [c for c in train.columns if c.startswith("X_")]
X = train[feature_cols].copy()
y = train["target"].copy()
X_test = test[feature_cols].copy()

# ===== 3) 전처리: 중복/상수/고상관 정리 =====
# 3-1) 중복행 제거(피처 기준)
dup_mask = X.duplicated(keep="first")
removed_dup = int(dup_mask.sum())
if removed_dup:
    X = X.loc[~dup_mask].reset_index(drop=True)
    y = y.loc[~dup_mask].reset_index(drop=True)

# 3-2) 상수(또는 거의 상수) 특성 제거
# (분할에 의미 없는 특성 제거)
nunique = X.nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index.tolist()
X.drop(columns=constant_cols, inplace=True)
X_test.drop(columns=[c for c in constant_cols if c in X_test.columns], inplace=True)

# 3-3) 고상관 특성 제거 (선택: 임계치 0.995)
def drop_high_corr(df, threshold=0.995):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop

high_corr_cols = drop_high_corr(X, threshold=0.995)
X.drop(columns=high_corr_cols, inplace=True)
X_test.drop(columns=[c for c in high_corr_cols if c in X_test.columns], inplace=True)

# 남은 피처 목록 저장
kept_features = X.columns.tolist()

# ===== 4) 결측치 대비: median impute (train/test 동일 규칙) =====
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=kept_features)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=kept_features)

# ===== 5) 빠른 랜덤서치 (5~10분 목표) =====
base_rf = RandomForestClassifier(
    n_estimators=200,   # 탐색 단계는 낮게 고정(속도)
    random_state=42,
    n_jobs=-1
)

param_distributions = {
    "max_depth": [None, 10, 16, 24, 32],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, 0.7],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rnd = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_distributions,
    n_iter=60,              # 시간에 맞춰 40~80 사이 조절
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("▶ RandomizedSearchCV 탐색 시작...")
rnd.fit(X_imp, y)
print("Best Params:", rnd.best_params_)
print("Best CV Accuracy (search):", rnd.best_score_)

# ===== 6) 최적 파라미터로 전체 재학습 (트리 수↑) =====
best_params = rnd.best_params_.copy()
rf_best = RandomForestClassifier(
    n_estimators=400,   # 최종 학습에서만 증가 (여유되면 500~800)
    random_state=42,
    n_jobs=-1,
    **best_params
)
print("▶ 최적 하이퍼파라미터로 전체 재학습...")
rf_best.fit(X_imp, y)

# (선택) CV 재확인
cv_scores = cross_val_score(rf_best, X_imp, y, cv=cv, scoring="accuracy", n_jobs=-1)
print("CV Accuracy (re-check) mean:", cv_scores.mean())
print("CV Accuracy (re-check) std :", cv_scores.std())

# ===== 7) 예측 & 제출 파일 저장 =====
preds = rf_best.predict(X_test_imp)
submission = sample_submission.copy()
submission["target"] = preds.astype(int)

out_path = "./data/open/rf-dacon-fast-prep.csv"
submission.to_csv(out_path, index=False)
print(f"✅ Saved submission to: {out_path}")

# ===== 8) 참고 아웃풋: 드롭한 컬럼/중요도 저장 =====
pd.DataFrame({
    "dropped_constant": constant_cols
}).to_csv("./data/open/dropped_constant_cols.csv", index=False)

pd.DataFrame({
    "dropped_high_corr": high_corr_cols
}).to_csv("./data/open/dropped_highcorr_cols.csv", index=False)

fi = pd.DataFrame({
    "feature": kept_features,
    "importance": rf_best.feature_importances_
}).sort_values("importance", ascending=False)
fi.to_csv("./data/open/rf_feature_importances_after_prep.csv", index=False)

print(f"⏱️ Elapsed: {time.time() - t0:.1f} sec")

