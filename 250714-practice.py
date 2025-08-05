# [연습문제] 벡터와 넘파이 전체 연습 문제

#11
#| echo: false
import numpy as np
import pandas as pd
# 행렬 A 생성
A = np.array([[3, 5, 7],
              [2, 3, 6]])
print("행렬 A:\n", A)

#12
#| echo: false
# 재현성을 위해 난수 시드 설정
np.random.seed(2023)
# 행렬 B 생성
B = np.random.choice(range(1, 11), 20, replace=True).reshape(5, 4)
print("행렬 B:\n", B)

B[1]
result = np.array([[B[1], B[3], B[4]]])
result

#13
B = np.random.choice(range(1, 11), 20, replace=True).reshape(5, 4)
B
B[:, 3]
np.where(B[:, 2] > 3)

#14
row_sums = np.sum(B, axis=1)
print("각 행별 합계:\n", row_sums)

result = np.array(B[np.where(row_sums > 20)])
result

#15
result = np.where(np.mean(B, axis=0) >= 5)[0]
result

#16
np.where(np.sum(B>7, axis=1))

#17
x = np.array([1, 2, 3, 4, 5])  
y = np.array([2, 4, 5, 4, 5])  # 종속 변수

print("x 벡터:", x)
print("y 벡터:", y)

top = np.sum((x - x.mean())*(y - y.mean()))
bot = np.sum((x - x.mean())**2)
answer = top / bot

#18??
X = np.array([[2, 4, 6],
              [1, 7, 2],
              [7, 8, 12]])
y = np.array([[10],
              [5],
              [15]])

########################################################################################
#판다스 알아보기 챕터 하단 연습 문제 1, 2, 3, 4, 6, 10, 11
#1 
df = pd.read_csv('./data/grade.csv')
print(df.head())

df.info()

#2
df.loc[df["midterm"] >= 85]

#3
df["final"].sort_values().head()
df["final"].sort_values(ascending=False).head()

#4
df.groupby("gender")[["midterm", "final"]].mean()

#6
df.iloc[df["assignment"].argmax()]
df.iloc[df["assignment"].argmin()]
#idxmax를 사용해도 된다


df[df["assignment"].argmin()]

#10
df = pd.read_csv('./data/grade.csv')
df["average"] = (df["midterm"] + df["final"]+ df["assignment"])/3
df
df.groupby(["gender"]).mean(["assignment", "average", "final", "midterm"])

df_melted = pd.melt(df,
                     id_vars=["student_id", "gender"], 
                     value_vars=["assignment", "average", "final", "midterm"], 
                     var_name='variable', 
                     value_name='score')
df_melted = pd.melt(df.reset_index(), id_vars=["index"], value_vars=["assignment", "average", "final", "midterm"], var_name='variable', value_name='score')

df_melted.groupby(["gender","variable"]).mean(["assignment", "average", "final", "midterm"])
df_melted
df_pivot_table = df_melted.pivot_table(index= 'gender', columns='variable', values='score')
df_pivot_table
#11
df["average"] = (df["midterm"] + df["final"]+ df["assignment"])/3
df.loc[df["average"].idxmax(), ['name', 'average']]

max_avg_student_idx = df['average'].idxmax()
df.loc[max_avg_student_idx, ['name', 'average']]


### 수업
#average
df["average"] = (df["midterm"] + df["final"]+ df["assignment"])/3
df

df.groupby("gender")[["midterm", "final", "assignment", "average"]].mean()

result = df.iloc[:, 2:].groupby("gender", as_index=False).mean()
result = pd.melt(result,
        id_vars='gender',
        var_name='variable',
        value_name='score',
        )
result.sort_values(["gender", 'variable'])











data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}
df = pd.DataFrame(data)
print(df)

df_melted = pd.melt(df, 
                    id_vars=['Date'],
                    value_vars=['Temperature', 'Humidity'],
                    var_name='Variable', 
                    value_name='Value')
print(df_melted)

df_pivoted = df_melted.pivot(index='Date', 
                             columns='Variable', 
                             values='Value').reset_index()

df_melted2 = pd.melt(df.reset_index(), 
                    id_vars=['index'],
                    value_vars=['Temperature', 'Humidity'],
                    var_name='Variable', 
                    value_name='Value')
print(df_melted2)







df = pd.DataFrame({
    'Date': ['2024-07-01', '2024-07-02'],
    'Temperature': [10, 20],
    'Humidity': [60, 65]
})
df_melted = pd.melt(df,
                    id_vars=['Date'],
                    value_vars=['Temperature', 'Humidity'],
                    var_name='Variable', 
                    value_name='numers')

df_melted