import pandas as pd
# 데이터 프레임 생성
df = pd.DataFrame({
    'col1': ['one', 'two', 'three', 'four', 'five'],
    'col2': [6, 7, 8, 9, 10]
})
print(df)
df.shape
df['col1'][1]
df['col1']
df['col2']



#series
data = [10, 20, 30]
df_s = pd.Series(data, index=['one', 'two', 'three'], 
                 name = 'count')
print(df_s)
df_s.shape
df_s[0]

# 데이터 프레임 생성
my_df = pd.DataFrame({
    'name': ['issac', 'bomi'],
    'birthmonth': [5, 4]
})
print(my_df)
my_df
my_df.info()

# 130 ~ 179page

url = "https://bit.ly/examscore-csv"
mydata = pd.read_csv(url)
print(mydata.head())
print(mydata)

mydata.shape
mydata["gender"].head()
mydata["gender"].tail(10)

mydata[["midterm", "final"]].head()
mydata.columns

mydata[mydata["midterm"] > 15].head()
mydata[mydata["midterm"] > 39]

#iloc - indexing
mydata.iloc[:, 1]
mydata.iloc[1:5, 2]
mydata.iloc[1:4, :3]

mydata.iloc[:, 1].head()
mydata.iloc[:, [1]].head() # 데이터 프래임을 유지하고 싶다면 []사용
mydata.iloc[:, [1]].squeeze() # 시리즈로 바꿔주는 함수 sqeeze()

mydata.iloc[:, [1, 0, 1]].head() #중복해서 가져올 수 있다

mydata.loc[mydata['midterm'] <= 15].head()

#라벨 인덱싱 Loc()
# True False 필터링 가능
mydata.loc[:, "midterm"]
mydata.loc[:, 2] # 칼럼을 숫자 기준으로 접근을 허용하지 않는다
mydata.loc[1:4, "midterm"]

mydata[mydata['midterm'] <= 15].head()
mydata.loc[mydata['midterm'] <= 15, "gender"].head()
mydata.loc[mydata['midterm'] <= 15, ["gender", "student_id"]].head()


mydata['midterm'].isin([28, 38, 52])
#중간고사 점수 28, 30, 52인 애들의 기말고사 점수와 성별정보

mydata.loc[mydata['midterm'].isin([28, 38, 52]), ["final", "gender"]]

# .iloc[]?
import numpy as np
check_index = np.where(mydata['midterm'].isin([28, 38, 52]))[0]
mydata.iloc[check_index, [3, 1]]

# 일부 데이터를 NA로 설정
mydata.iloc[0, 1] = np.nan
mydata.iloc[4, 0] = np.nan

mydata.head()

mydata["gender"].isna().sum()

mydata.dropna() #2개 빠짐

#1번
mydata["student_id"].isna().head()
#2번
vec_2 = ~mydata["std-id"].isna()

#3번
vec_3 = ~mydata["gender"].isna()

mydata[vec_2 & vec_3]

mydata['total'] = mydata['midterm'] + mydata['final']
print(mydata.iloc[0:3, [3, 4]])

mydata = pd.concat([mydata, (mydata['total'] / 2).rename('average')], axis=1)

mydata["average^2"] = mydata["average"]**2
mydata

del mydata["average^2"]

mydata.rename(columns={"student_id" : "std-id"}, inplace=True)

mydata


df1 = pd.DataFrame({
'A': ['A0', 'A1', 'A2'],
'B': ['B0', 'B1', 'B2']
})
df2 = pd.DataFrame({
'A': ['A3', 'A4', 'A5'],
'B': ['B3', 'B4', 'B5']
})

result = pd.concat([df1, df2], axis=1)
result = pd.concat([df1, df2], ignore_index=True)
result

df4 = pd.DataFrame({
'A': ['A2', 'A3', 'A4'],
'B': ['B2', 'B3', 'B4'],
'C': ['C2', 'C3', 'C4']
})

pd.concat([df1, df4], join="inner")
pd.concat([df1, df4], join="outer")

df = pd.read_csv('./data/penguins.csv')
df

#1 Q1 bill_lengthmm, bill_depth_mm, flipper_length_mm, body_mass-g 중에서 결측치가 하나라도 있는 행은 몇 개인가요?
bl = df["bill_length_mm"].isna()
bd = df["bill_depth_mm"].isna()
fl = df["flipper_length_mm"].isna()
bm = df["body_mass_g"].isna()
len(df[bl | bd | fl | bm])

df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].isna().sum()

df.iloc[:, 2:6].shape[0]
df.iloc[:, 2:6].dropna().shape[0]

#2 몸무게가 4000g이상 5000g 이하인 펭귄은 몇 마리?
((df['body_mass_g']>=4000) & (df["body_mass_g"]<=5000)).sum()
#NaN도 똑같이 작동한다
df['body_mass_g'].between(4000, 5000).sum()


#3 펭귄 종(spiecies)별로 평균 부리 길이는?
# sp = set(df["species"])
# df['species'].isin(sp)

df.groupby("species").mean(numeric_only=True)["bill_length_mm"]
df.groupby("species").mean(numeric_only=True)["flipper_length_mm"]

#최대 부리길이
df.groupby("species")["bill_length_mm"].max()


df["species"].unique()
df.loc[df["species"] == "Adelie", "bill_length_mm"].mean()
df.loc[df["species"] == "Chinstrap", "bill_length_mm"].mean()
df.loc[df["species"] == "Gentoo", "bill_length_mm"].mean()

#4 성별이 결측치가 아닌 데이터 중, 성별 비율은 각각 몇 퍼센트인가요?
sx = ~df["sex"].isna()
ss = df[sx]
male = (ss['sex'] == 'Male').sum()/len(ss) * 100
female = (ss['sex'] == 'Female').sum()/len(ss) * 100

#5 섬 별로 평균 날개 길이(flipper_length_mm)가 가장 긴 섬은 어디인가요
mean_vec = df.groupby("island")["flipper_length_mm"].mean(numeric_only=True)
mean_vec.index[mean_vec.argmax()]
df.mean(numeric_only=True)


df.describe()

# 'bill_length_mm' 열을 기준으로 데이터 프레임 정렬
sorted_df = df.sort_values(by='bill_length_mm')
sorted_df = df.sort_values(by=['bill_length_mm', "flipper_length_mm"])
sorted_df = df.sort_values(by='bill_length_mm', ascending=False)
sorted_df = df.sort_values(by=['bill_length_mm', "flipper_length_mm"], ascending=[True, False])
print(sorted_df.head())
























