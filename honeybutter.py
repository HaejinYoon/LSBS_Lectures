import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 집가격 데이터 불러오세요!
honeybutter_df = pd.read_csv('./data/honeybutter.csv')
honeybutter_df.info()
honeybutter_df["Conv_speed"].hist()
honeybutter_df["paddle_speed"].hist()
honeybutter_df['Oil_temperature'].mean()

# Convert 'time' column to datetime and create derived features
honeybutter_df['time'] = pd.to_datetime(honeybutter_df['time'])

# Extract common time-based features
honeybutter_df['year'] = honeybutter_df['time'].dt.year
honeybutter_df['month'] = honeybutter_df['time'].dt.month
honeybutter_df['day'] = honeybutter_df['time'].dt.day
honeybutter_df['hour'] = honeybutter_df['time'].dt.hour
honeybutter_df['minute'] = honeybutter_df['time'].dt.minute
honeybutter_df['weekday'] = honeybutter_df['time'].dt.weekday  # Monday=0, Sunday=6
honeybutter_df['is_weekend'] = honeybutter_df['weekday'].isin([5,6]).astype(int)

from sklearn.model_selection import train_test_split

# 독립변수(X), 종속변수(y) 분리
X = honeybutter_df.drop(columns=["time", "label"])  # time은 제거, 파생변수들은 포함
y = honeybutter_df["label"]

# 원하는 비율로 분리 (예: train 70%, test 30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify=y → label 비율 유지
)

def split_train_val_test(
    data: pd.DataFrame,
    label_col: str,
    drop_cols: list[str] = None,
    ratios: tuple = (0.6, 0.2, 0.2),
    random_state: int = 42,
    stratify: bool = True,
):
    """
    DataFrame을 (train, val, test)로 분할합니다.
    - ratios: (train, val, test) 세 비율의 합이 1이어야 합니다.
    - stratify=True이면 label 분포를 각 세트에서 유지합니다.
    - drop_cols: 학습에서 제외할 컬럼(예: 'time' 원본) 리스트
    """
    if drop_cols is None:
        drop_cols = []
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    X = data.drop(columns=[label_col] + drop_cols)
    y = data[label_col]

    train_ratio, val_ratio, test_ratio = ratios
    temp_ratio = val_ratio + test_ratio

    # 1) 전체에서 train vs temp(val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    # 2) temp을 다시 val vs test로 (비율 유지)
    val_fraction_of_temp = val_ratio / temp_ratio if temp_ratio > 0 else 0.0
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_fraction_of_temp),
        random_state=random_state,
        stratify=y_temp if stratify else None,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# 사용 예시: 6:2:2로 분할, 'time'은 드랍(파생변수만 사용)
X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
    honeybutter_df, label_col="label", drop_cols=["time"], ratios=(0.6, 0.2, 0.2), random_state=42, stratify=True
)

# 칼럼 선택
num_columns = X_train.select_dtypes(include=['number']).columns

honeybutter_df

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler().set_output(transform="pandas")
X_train_scaled = std_scaler.fit_transform(X_train[num_columns])

X_train_scaled

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(oob_score=True)
rf_grid = {
    # 트리 개수: 안정적인 학습을 위해 200~500 사이 권장
    "n_estimators": [200, 300, 500],

    # 리프 노드 최소 샘플 수: 너무 작으면 과적합, 2~10 사이가 적절
    "min_samples_leaf": [1, 2, 4, 8],

    # 분할 시 최소 샘플 수: 리프와 함께 조정 (데이터 크기 고려)
    "min_samples_split": [2, 5, 10, 20],

    # 특성 사용 비율: 기본값은 sqrt(#features). 비율로 지정도 가능
    "max_features": [0.3, 0.5, 0.7, "sqrt"],

    # 최대 리프 노드 수: 제한 없거나 중간 크기 추천
    "max_leaf_nodes": [None, 128, 256, 512],

    # 최대 깊이: 너무 깊으면 과적합, 없으면 자동 확장됨
    "max_depth": [None, 10, 20, 30],
}


# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)

# 그리드서치
rf_search = GridSearchCV(estimator=rf, 
                              param_grid=rf_grid, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

rf_search.fit(X_train_scaled, y_train)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(rf_search.cv_results_))

# best prameter
print(rf_search.best_params_)

# 교차검증 best score 
print(-rf_search.best_score_)

# {'max_depth': None, 'max_features': 0.3, 'max_leaf_nodes': 512, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
# 0.035711108071135436