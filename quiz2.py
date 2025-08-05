import pandas as pd
import numpy as np

df = pd.read_csv('./quiz2/problem1.csv')
df2 = pd.read_csv('./quiz2/problem2.csv')
df2_1 = pd.read_csv('./quiz2/problem2_1.csv')
df2_2 = pd.read_csv('./quiz2/problem2_2.csv')
df2_3 = pd.read_csv('./quiz2/problem2_3.csv')
df
df2
df2_1
df2_2
df2_3

#1
df.info()
df.isna()
df.shape[0]

#2
out = sum(df["퇴거여부"]== "퇴거")
df.shape[0] - out

#3
df1_1 = df.groupby(["아파트 이름", '성별'])["보증금(원)"].mean()
aa = np.array([
    abs(df1_1.iloc[0] - df1_1.iloc[1]),
    abs(df1_1.iloc[2] - df1_1.iloc[3]),
    abs(df1_1.iloc[4] - df1_1.iloc[5]),
    abs(df1_1.iloc[6] - df1_1.iloc[7]),
    abs(df1_1.iloc[8] - df1_1.iloc[9])
    ])
aa.argmax()
pd.melt(df1_1,
        id_vars=["아파트 이름"],
        value_vars=["남성", "여성"],
        var_name="성별",
        value_name="보증금(원)"
    )


#4
df.iloc[df["월세(원)"].idxmax()]

#5
df.groupby('층')["거주자 수"].mean().idxmax()

#6
df[(df["계약구분"] =='유효') & (df["재계약횟수"]>=5)].groupby("평형대")['대표나이'].mean().idxmax()


#7
df["계약자고유번호", "겨주연도"]
print(df)
df.info()
df.groupby("계약자고유번호")["거주연도"].idxmax()
sorted_df = df.sort_values(by="거주연도", ascending=False)
sorted_df[sorted_df["거주연도"] >= 2019]
sorted_df.shape

#8
df['아파트 평점'].isna().sum()
df['계약구분'].isna().sum()
df['아파트 평점'] = df['아파트 평점'].dropna()
df.dropna(subset=['계약구분', '아파트 평점'], inplace=True)
df

#9
df.shape[0] - df[(df['퇴거여부'] == '미퇴거') & (df['퇴거연도'].notna())].shape[0]

df['퇴거연도'].notna().sum()

#10
df['중앙값'] = df['재계약횟수'].median()
df['이분변수'] = np.where(df['재계약횟수']>=df['재계약횟수'].median(), '높음', '낮음')
df.groupby('이분변수')['거주개월'].mean()

#11
df.groupby('이분변수')['나이'].median()


#12
df.groupby(["이분변수", '성별'])['성별'].count()
17079/(17079 + 20416)
20416/(17079 + 20416)

18591/(18591 + 30818)
30818/(18591 + 30818)


#13
df2 = pd.read_csv('./quiz2/problem2.csv')
df2.info()
df2.shape
df2 = df2.dropna()
print(df2)

type(df2['a1_1'])
df2.loc[:,['a1_1']] = df2.loc[:,['a1_1']].astype(str)
df2.loc[:,:].str.replace(r'[^a-zA-Z0-9가-힣\s]', '', regex=True)


#14
df2_1 = pd.read_csv('./quiz2/problem2_1.csv')

df2_1
df2_1.info()
ining = df2_1.iloc[:, :19]
score = df2_1.iloc[:, 19:]
score

pd.df2_1
df_pivot = df2_1.pivot_table(
                               index=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9'], 
                            columns=['game_id'], 
                                #  values=, 
                                fill_value=0)
df_pivot
df2_1['game_id']


iningscore = []
for i in score:
    
    iningscore.append(df2_1[i].mean())
iningscore.max()

#15
df2_1 = pd.read_csv('./quiz2/problem2_1.csv')

a1 = []
a2 = []
for j in range(1, 10):
    ining.iloc[:,j].mean() - ining.iloc[:,j].mean()
    # df2_1[j].mean()



#19
df2_3 = pd.read_csv('./quiz2/problem2_3.csv')
df2_3

df2_3['score_index'] = np.where(df2_3['score']>0, 0, 1)

df2_3.groupby('score_index')['ining2_move'].mean()
















