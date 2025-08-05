import pandas as pd

## [실습] 날짜형, 문자형 다루기 연습 - 자전거 시스템의 대여기록
df = pd.read_csv('./data/bike_data.csv')
print(df.head())

df = df.astype({'datetime' : 'datetime64[ns]', 'weather' : 'int64', 
                'season' : 'object', 'workingday' : 'object', 
                'holiday' : 'object'})
#1
df_s1 = df[df["season"]==1]
df_s1
df_s1['hour'] = df_s1['datetime'].dt.hour
df_s1.groupby('hour')['count'].max()

df_s1.loc[df_s1['count'].idxmax()]["hour"]



df_sub = df.loc[df.season == 1, ]
# 시간 정보 추출
df_sub.loc[:, 'hour'] = df_sub['datetime'].dt.hour
# count가 가장 큰 hour 찾기
max_count_hour = df_sub.loc[df_sub['count'].idxmax(), 'hour']
max_count = df_sub['count'].max()
print(f"count가 가장 큰 hour는 {max_count_hour}시이며, 대여량은 {max_count}입니다.")

#2 각 계절별 평균 대여량
df.groupby('season')['count'].mean()

#3 특정 달 총 대여량
df_m = df[df['datetime'].dt.month == 1]
df_m["count"].sum()

#4 
df
df['newdate'] = df["datetime"].dt.date
a = df.groupby('newdate')[['count']].sum()
max_date = a.idxmax()
max_count = a.max()

#5
df["newHour"] = df["datetime"].dt.hour
df.groupby("newHour", as_index=False)["count"].mean()

#6
df['newDay'] = df["datetime"].dt.weekday
df.groupby("newDay")["count"].sum()

#7
df_melted = pd.melt(df, 
                    id_vars=['datetime', 'season'],
                    value_vars=['casual', 'registered'],
                    var_name='user_type', 
                    value_name='user_count')
df_melted

#8
df_melted.groupby(['season', 'user_type'])[["user_count"]].mean().reset_index()

pd.set_option('display.max_columns', None) # 전체 칼럼 정보 프린트 옵션
df = pd.read_csv('./data/logdata.csv')
df

#1 숫자만 추출

df_extracted_n = df['로그'].str.extractall(r'(\d+)')
df_extracted_n

#2 시간 정보

df_time = df['로그'].str.extract(r'(\d+:\d+:\d+)')
df_time

# 3 한글 정보
df_Kor = df['로그'].str.extract(r'([가-힣]+)')
df_Kor

#4 특수 문자 제거
df_special = df['로그'].str.replace(r'([^a-zA-Z0-9가-힣\s])', '',regex=True)
df_special

#5 Amount 평균
df['이름'] = df['로그'].str.extract(r'([가-힣]+)')
df['Amount'] = df['로그'].str.extract(r'()')
df['로그'].str.extract(r'(Amount:\s + \d+)')

