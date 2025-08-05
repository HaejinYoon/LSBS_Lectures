import pandas as pd

df = pd.read_csv('./data/penguins.csv')
df.info()

df.describe()
df.sort_values("bill_length_mm")

# groupby()
result = df.groupby("species")["bill_length_mm"].mean()
result
#idxmax()
#result.index[result.values.argmax()]
result.idxmax()

df["bill_length_mm"]

#merge()
# 예제 데이터 프레임 생성
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
# 두 데이터 프레임 병합
merged_df = pd.merge(df1, df2, on='key', how='inner')
print(merged_df)

merged_df = pd.merge(df1, df2, on='key', how='outer')


mid_df1 = pd.DataFrame({'std_id': [1, 2, 4, 5], 'midterm': [10, 20, 40, 30]})
fin_df2 = pd.DataFrame({'std_id': [1, 2, 3, 6], 'final': [4, 5, 7, 2]})
#중간 & 기말 둘다 본 학생들의 데이터
pd.merge(mid_df1, fin_df2, on='std_id', how='inner')

#학생들 전체의 중간 & 기말 데이터
pd.merge(mid_df1, fin_df2, on='std_id', how='outer')

#std_id student id
midfin_df = pd.merge(mid_df1, fin_df2, left_on='student_id', right_on='std_id', how='inner')
del midfin_df['std_id']


#merge how = "left"
#가장 빈번하게 사용됨
left_df = pd.merge(mid_df1, fin_df2, on='std_id', how='left')
left_df

#melt
wide_df = pd.DataFrame({
    '학생' : ['철수', '영희', '민수'],
    '수학' : [90, 80, 70],
    '영어' : [85, 95, 75]
})
wide_df

long_df = pd.melt(wide_df, 
        id_vars='학생', #기준 칼럼
        var_name='과목',
        value_name='점수'
        )
long_df

result_df = long_df.pivot_table(
    index="학생",
    columns="과목",
    values="점수"
).reset_index()
result_df.columns.name = None
result_df

#id_vars가 한 개인 경우
w_df = pd.DataFrame({
    '반' : ['A', 'B', 'C'],
    '1월' : [20, 18, 22],
    '2월' : [19, 20, 21],
    '3월' : [21, 17, 23]
})
long_w_df = pd.melt(w_df,
                    id_vars='반',
                    var_name='월',
                    value_name='출석일수'
                    )
long_w_df


#id_vars가 두 개인 경우
w_df = pd.DataFrame({
    '학년': [1, 1, 2],
    '반' : ['A', 'B', 'C'],
    '1월' : [20, 18, 22],
    '2월' : [19, 20, 21],
    '3월' : [21, 17, 23]
})
long_w_df = pd.melt(w_df,
                    id_vars=['학년', '반'],
                    var_name='월',
                    value_name='출석일수'
                    )
long_w_df

df3 = pd.DataFrame({
    '학생' : [ '철수', '영희', '민수'],
    '국어' : [90, 80, 85],
    '수학' : [70, 90, 75],
    '영어' : [88, 92, 79],
    '학급' : ['1반', '1반', '2반']
})

#특정 칼럼만 뽑아오기
pd.melt(df3,
        id_vars=['학급', '학생'],
        var_name='언어 과목',
        value_vars=['국어', '영어'],
        value_name='성적'
        )






























