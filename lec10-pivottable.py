import pandas as pd
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}
df = pd.DataFrame(data)
print(df)
df

df_melted = pd.melt(df, 
                    id_vars=['Date'],
                    value_vars=['Temperature', 'Humidity'],
                    var_name='측정요소', 
                    value_name='측정값')
print(df_melted.head(6))
df_melted

# 원래 형식으로 변환
# Date, Temperature, Humidity
di_pivot = pd.pivot_table(
    df_melted,
    index="Date",
    columns="측정요소",
    values="측정값",
    aggfunc="sum"
).reset_index()
di_pivot.columns.name = None
di_pivot

# pivot은 오류
di_pivot = pd.pivot(
    df_melted,
    index="Date",
    columns="측정요소",
    values="측정값"
).reset_index()

df = pd.read_csv('./data/dat.csv')
print(df.head())
df.columns
df = df.rename(columns = {'Dalc' : 'dalc', 'Walc' : 'walc'})

df.info()
df.loc[:, ['famrel', 'dalc']].astype({'famrel' : 'object', 'dalc' : 'float64'})

def classify_famrel(famrel):
    if famrel <= 2:
        return 'Low'
    elif famrel <= 4:
        return 'Medium'
    else:
        return 'High'

classify_famrel(3)
classify_famrel(4)
classify_famrel(5)
df['famrel']

df=df.assign(famrel = df['famrel'].apply(classify_famrel))

print(df.select_dtypes('number').head(2))
print(df.select_dtypes('object').head(2))

import numpy as np
def standardize(x):
    return ((x - np.nanmean(x))/np.std(x))

vec_a=np.arange(5)
vec_a

standardize(vec_a)
vec_a.std()
(vec_a-2)/1.414

df_std = df.select_dtypes('number').apply(standardize, axis=0) # axis = 여부 찾아보기
df_std.mean(axis=0)
df_std.std(axis=0)

index_f = df.columns.str.startswith('f')

df.loc[:, index_f].head()

# 예제 데이터 생성
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [90, 85, 88]
})

df.to_csv("data.csv", index=False)























































