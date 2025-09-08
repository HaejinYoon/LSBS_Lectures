import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 타이타닉 데이터 불러오세요!
train_df = pd.read_csv('./data/titanic/train.csv')
test_df = pd.read_csv('./data/titanic/test.csv')
#train_df = train_df.select_dtypes(include=['number'])

# 칼럼 선택
num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop("Survived")
cat_columns = train_df.select_dtypes(include=['object']).columns
cat_columns = cat_columns.drop(["Name", "Ticket", "Cabin"])

# 각 칼럼별 결측지 몇 개가 있을까
train_df.isna().sum(axis=0)

# 결측치 채우기 (간단히 처리)
from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')

train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
freq_impute.statistics_

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
onehot = OneHotEncoder(handle_unknown='ignore', 
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")

train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])

train_df_all = pd.concat([train_df_num, train_df_cat], axis = 1)
train_df_all

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df_all
y_train = train_df["Survived"]

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
dct_search.fit(X_train, y_train)
dct_search.best_params_

#테스트 데이터 채우기
test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])

test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])

test_df_all = pd.concat([test_df_num, test_df_cat], axis = 1)

dct_search.predict(test_df_all)

y_pred = dct_search.predict(test_df_all)
submit = pd.read_csv('./data/titanic/gender_submission.csv')
submit["Survived"]=y_pred

# CSV로 저장
submit.to_csv('./data/titanic/base-model.csv', index=False)

#==========================================================================================
#
#==========================================================================================

# 타이타닉 데이터 불러오세요!
train_low_data = pd.read_csv('./data/titanic/train.csv')
test_low_data = pd.read_csv('./data/titanic/test.csv')
#train_df = train_df.select_dtypes(include=['number'])

# test_low_data PassengerId를 따로 저장(후에 predict를 위해)
test_passenger_ids = test_low_data['PassengerId']

train_low_data['Sex_clean'] = train_low_data['Sex'].astype('category').cat.codes
test_low_data['Sex_clean'] = test_low_data['Sex'].astype('category').cat.codes

# 결측치 확인
print(train_low_data['Embarked'].isnull().sum())  # train 데이터 결측치 확인
print(test_low_data['Embarked'].isnull().sum())   # test 데이터 결측치 확인
print(train_low_data['Embarked'].value_counts())  # Embarked 값 분포 확인

# 결측치 처리
train_low_data.loc[:, 'Embarked'] = train_low_data['Embarked'].fillna('S')
test_low_data.loc[:, 'Embarked'] = test_low_data['Embarked'].fillna('S')

# 범주형 데이터를 숫자형으로 인코딩
train_low_data['Embarked_clean'] = train_low_data['Embarked'].astype('category').cat.codes
test_low_data['Embarked_clean'] = test_low_data['Embarked'].astype('category').cat.codes

train_low_data

train_low_data['Family']=1+train_low_data['SibSp']+train_low_data['Parch']
test_low_data['Family']=1+test_low_data['SibSp']+test_low_data['Parch']

#train_low_data['Solo']=train_low_data['Family']==1
#test_low_data['Solo']=test_low_data['Family']==1

# 'Solo' 열 생성 (Family가 1이면 1, 아니면 0)
train_low_data['Solo'] = train_low_data['Family'].apply(lambda x: 1 if x == 1 else 0)
test_low_data['Solo'] = test_low_data['Family'].apply(lambda x: 1 if x == 1 else 0)

# 'Solo' 열 확인
print(train_low_data['Solo'])

train_low_data['FareBin']=pd.qcut(train_low_data['Fare'],5)
test_low_data['FareBin']=pd.qcut(test_low_data['Fare'],5)
print(train_low_data['FareBin'].value_counts())

train_low_data['Fare_clean']=train_low_data['FareBin'].astype('category').cat.codes
test_low_data['Fare_clean']=test_low_data['FareBin'].astype('category').cat.codes
train_low_data['Fare_clean'].value_counts

train_low_data['Title'] = train_low_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_low_data['Title'] = test_low_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

#replace이후 list이외 others
train_low_data['Title']=train_low_data['Title'].replace(['Lady', 'Countess','Capt', 
                            'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer',  'Dona'], 'Other')

test_low_data['Title']=test_low_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                            'Major', 'Rev', 'Sir', 'Jonkheer',  'Dona'], 'Other')

print(train_low_data['Title'].value_counts())
print(test_low_data['Title'].value_counts())

# 각 Title의 개수 출력
title_counts_train = train_low_data['Title'].value_counts()
title_counts_test = test_low_data['Title'].value_counts()

# "Other"의 갯수 확인
other_count_train = title_counts_train.get('Other', 0)  # "Other"가 없으면 0 반환
other_count_test = title_counts_test.get('Other', 0)    # "Other"가 없으면 0 반환

print("Train Data 'Other' Count:", other_count_train)
print("Test Data 'Other' Count:", other_count_test)

# 'Mlle', 'Ms', 'Mme' 호칭을 각각 'Miss', 'Miss', 'Mrs'로 변경
train_low_data['Title'] = train_low_data['Title'].replace('Mlle', 'Miss')
train_low_data['Title'] = train_low_data['Title'].replace('Ms', 'Miss')
train_low_data['Title'] = train_low_data['Title'].replace('Mme', 'Mrs')
test_low_data['Title'] = test_low_data['Title'].replace('Ms', 'Miss')

# 각 Title의 개수 출력
print(train_low_data['Title'].value_counts())
print(test_low_data['Title'].value_counts())

train_low_data['Title_clean'] = train_low_data['Title'].astype('category').cat.codes
test_low_data['Title_clean'] = test_low_data['Title'].astype('category').cat.codes

# 나이 결측치 대체
train_low_data['Age'] = train_low_data['Age'].fillna(train_low_data.groupby("Title")["Age"].transform("median"))
test_low_data['Age'] = test_low_data['Age'].fillna(test_low_data.groupby("Title")["Age"].transform("median"))

#train_low_data
train_low_data.loc[   train_low_data['Age'] <= 10, 'Age_clean'] = 0
train_low_data.loc[(  train_low_data['Age'] > 10) & (  train_low_data['Age'] <= 16), 'Age_clean'] = 1
train_low_data.loc[(  train_low_data['Age'] > 16) & (  train_low_data['Age'] <= 20), 'Age_clean'] = 2
train_low_data.loc[(  train_low_data['Age'] > 20) & (  train_low_data['Age'] <= 26), 'Age_clean'] = 3
train_low_data.loc[(  train_low_data['Age'] > 26) & (  train_low_data['Age'] <= 30), 'Age_clean'] = 4
train_low_data.loc[(  train_low_data['Age'] > 30) & (  train_low_data['Age'] <= 36), 'Age_clean'] = 5
train_low_data.loc[(  train_low_data['Age'] > 36) & (  train_low_data['Age'] <= 40), 'Age_clean'] = 6
train_low_data.loc[(  train_low_data['Age'] > 40) & (  train_low_data['Age'] <= 46), 'Age_clean'] = 7
train_low_data.loc[(  train_low_data['Age'] > 46) & (  train_low_data['Age'] <= 50), 'Age_clean'] = 8
train_low_data.loc[(  train_low_data['Age'] > 50) & (  train_low_data['Age'] <= 60), 'Age_clean'] = 9
train_low_data.loc[   train_low_data['Age'] > 60, 'Age_clean'] = 10

#test_low_data
test_low_data.loc[ test_low_data['Age'] <= 10, 'Age_clean'] = 0
test_low_data.loc[(test_low_data['Age'] > 10) & (test_low_data['Age'] <= 16), 'Age_clean'] = 1
test_low_data.loc[(test_low_data['Age'] > 16) & (test_low_data['Age'] <= 20), 'Age_clean'] = 2
test_low_data.loc[(test_low_data['Age'] > 20) & (test_low_data['Age'] <= 26), 'Age_clean'] = 3
test_low_data.loc[(test_low_data['Age'] > 26) & (test_low_data['Age'] <= 30), 'Age_clean'] = 4
test_low_data.loc[(test_low_data['Age'] > 30) & (test_low_data['Age'] <= 36), 'Age_clean'] = 5
test_low_data.loc[(test_low_data['Age'] > 36) & (test_low_data['Age'] <= 40), 'Age_clean'] = 6
test_low_data.loc[(test_low_data['Age'] > 40) & (test_low_data['Age'] <= 46), 'Age_clean'] = 7
test_low_data.loc[(test_low_data['Age'] > 46) & (test_low_data['Age'] <= 50), 'Age_clean'] = 8
test_low_data.loc[(test_low_data['Age'] > 50) & (test_low_data['Age'] <= 60), 'Age_clean'] = 9
test_low_data.loc[ test_low_data['Age'] > 60, 'Age_clean'] = 10

train_low_data['Cabin'].str[:1].value_counts()

mapping = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'T': 7
}

train_low_data['Cabin_clean'] = train_low_data['Cabin'].str[:1]
train_low_data['Cabin_clean'] = train_low_data['Cabin_clean'].map(mapping)

test_low_data['Cabin_clean'] = test_low_data['Cabin'].str[:1]
test_low_data['Cabin_clean'] = test_low_data['Cabin_clean'].map(mapping)

train_low_data["Cabin_clean"] = train_low_data.groupby("Pclass")["Cabin_clean"].transform(lambda x: x.fillna(x.median()))
test_low_data["Cabin_clean"] = test_low_data.groupby("Pclass")["Cabin_clean"].transform(lambda x: x.fillna(x.median()))

print(train_low_data['Cabin_clean'].value_counts())
print(test_low_data['Cabin_clean'].value_counts())

train_low_data.info()
test_low_data.info()

num_columns = train_low_data.select_dtypes(include=['number']).columns
num_columns = num_columns.drop("Survived")
cat_columns = train_low_data.select_dtypes(include=['object']).columns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
onehot = OneHotEncoder(handle_unknown='ignore', 
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")

train_df_cat = onehot.fit_transform(train_low_data[cat_columns])
train_df_num = std_scaler.fit_transform(train_low_data[num_columns])

train_df_all = pd.concat([train_df_num, train_df_cat], axis = 1)
train_df_all

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df_all
y_train = train_low_data["Survived"]

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
dct_search.fit(X_train, y_train)
dct_search.best_params_

#테스트 데이터 채우기
test_low_data[num_columns] = mean_impute.transform(test_low_data[num_columns])
test_low_data[cat_columns] = freq_impute.transform(test_low_data[cat_columns])

test_df_cat = onehot.transform(test_low_data[cat_columns])
test_df_num = std_scaler.transform(test_low_data[num_columns])

test_df_all = pd.concat([test_df_num, test_df_cat], axis = 1)

dct_search.predict(test_df_all)

y_pred = dct_search.predict(test_df_all)
submit = pd.read_csv('./data/titanic/gender_submission.csv')
submit["Survived"]=y_pred

# CSV로 저장
submit.to_csv('./data/titanic/more-preprocessing-model.csv', index=False)

#==========================================================================================
# 범주형 변수 인코딩하기
#==========================================================================================

import pandas as pd 
import numpy as np
dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')
y = dat.grade
X = dat.drop(['grade'], axis = 1)
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X,  
                   y, 
                   test_size = 0.2,                     
                   random_state = 0, 
                   shuffle = True, 
                   stratify = None 
                   )

#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
train_X6 = train_X.copy()
test_X6 = test_X.copy()
train_X6_cat = train_X6.select_dtypes('object')
test_X6_cat = test_X6.select_dtypes('object')

#labelencoder = LabelEncoder()
ordinalencoder = OrdinalEncoder().set_output(transform = 'pandas')
train_X6_cat = ordinalencoder.fit_transform(train_X6_cat)
test_X6_cat = ordinalencoder.transform(test_X6_cat)
print(train_X6_cat.head(2))

# 훈련 데이터
train_data = pd.DataFrame({
    'job': ['Doctor', 'Engineer', 'Teacher', 'Nurse']
})
# 테스트 데이터
test_data = pd.DataFrame({
    'job': ['Doctor', 'Lawyer', 'Teacher', 'Scientist']
})

# # OrdinalEncoder 설정
# oe = OrdinalEncoder()
# # 훈련 데이터 변환
# train_data['job_encoded'] = oe.fit_transform(train_data[['job']])
# test_data['job_encoded'] = oe.transform(test_data[['job']])

# OrdinalEncoder 설정
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# 훈련 데이터로 인코더 학습
oe.fit(train_data[['job']])

# 훈련 데이터 변환
train_data['job_encoded'] = oe.transform(train_data[['job']])
# 테스트 데이터 변환 (훈련 데이터에 없는 직업은 -1로 인코딩됨)
test_data['job_encoded'] = oe.transform(test_data[['job']])

from sklearn.preprocessing import OneHotEncoder
train_X7 = train_X.copy()
test_X7 = test_X.copy()
train_X7_cat = train_X7.select_dtypes('object')
test_X7_cat = test_X7.select_dtypes('object')

onehotencoder = OneHotEncoder(sparse_output = False, 
                        handle_unknown = 'ignore').set_output(transform = 'pandas')
                        
train_X7_cat = onehotencoder.fit_transform(train_X7_cat)
test_X7_cat = onehotencoder.transform(test_X7_cat)
print(train_X7_cat.head())