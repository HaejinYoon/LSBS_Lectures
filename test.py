a=1
print("a는 ",a)

import pandas as pd
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

df_melted2 = pd.melt(df.reset_index(), 
                    id_vars=['index'],
                    value_vars=['Temperature', 'Humidity'],
                    var_name='Variable', 
                    value_name='Value')
df_melted2

df_pivot_table = df_melted.pivot_table(index='Date', 
                                       columns='Variable', 
                                       values='Value').reset_index()
df_pivot_table

df = pd.read_csv('./data/grade.csv')
df
df[df["midterm"]>=85]
df["final"].sort_values(ascending=False)
df.sort_values(by='final', ascending=False)

import sqlite3
# DB 파일 연결 (없으면 자동 생성됨)
conn = sqlite3.connect("./data/penguins.db")

import pandas as pd
# SELECT 쿼리 결과를 DataFrame으로 읽기
df = pd.read_sql_query("SELECT * FROM penguins;", conn)
df
print(df.head())

# 예시 DataFrame
df2 = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [25, 30]
})
# SQLite 테이블로 저장
df2.to_sql("people", conn, if_exists="replace", index=False)

conn.close()