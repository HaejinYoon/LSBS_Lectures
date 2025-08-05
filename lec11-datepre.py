import pandas as pd
import numpy as np


data = {
    'date': ['2024-01-01 12:34:56', '2024-02-01 23:45:01', '2024-03-01 06:07:08', '2021-04-01 14:15:16'],
    'value': [100, 201, 302, 404]
}

df = pd.DataFrame(data)
df

df.info()

df['date'] = pd.to_datetime(df['date'])
print(df.dtypes)


pd.to_datetime('02-01-2024')

pd.to_datetime('02-2024-01', format='%m-%Y-%d')

# Y: 네자리 연도
# y:  2자리 연도
# m: 월 정보
# M : 분 정보

df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.hour
df['date'].dt.minute

df['date'].dt.day_name()
df['date'].dt.weekday

current_date = pd.to_datetime('2025-07-21')
(current_date - df['date']).dt.days #타임델타에 관해 찾아보기

date_range = pd.date_range(start='2021-01-01', end='2021-12-10', freq='D')
date_range

df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date2'] = pd.to_datetime(
    dict(
        year=df['date'].year,
        month=df['date'].month, 
        day=df['date'].day
        )
)







