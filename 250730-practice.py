import pandas as pd
import numpy as np

#1
url = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_1.csv"
mydata = pd.read_csv(url)
mydata['총합'] = mydata['금액1'] + mydata['금액2']
mydata

mydata_gender = mydata.groupby(["year", 'gender', '지역코드']).sum().reset_index()
df_pivot = mydata_gender.pivot_table(index=['year', '지역코드'], 
                                columns='gender', 
                                values='총합', 
                                fill_value=0)
df_pivot
df_abs = abs(df_pivot[0] - df_pivot[1])
df_abs.idxmax()
df_abs.max()


#######################

#=========================================================================================================
#2
url2 = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_2.csv"
mydata2 = pd.read_csv(url2)
mydata2

occ = mydata2[mydata2['구분'] == '발생건수'].reset_index()
cat = mydata2[mydata2['구분'] == '검거건수'].reset_index()
cat.iloc[0, 3:]
catch_rate = (cat.iloc[:, 3:] / occ.iloc[:,3:])*100
catch_rate.max()

cat[catch_rate.iloc[:,:]==100].sum().sum()


#=========================================================================================================
#3
url3 = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_3.csv"
mydata3 = pd.read_csv(url3)
mydata3

mydata3.loc[mydata3['평균만족도'].isna(), '평균만족도'] = mydata3['평균만족도'].mean()
mydata3[mydata3['근속연수'].isna()] = round(mydata3['근속연수'].mean())

mydata3.iloc[3]['근속연수']
a= round(mydata3.groupby(['부서', '등급'])['근속연수'].mean())
mydata3['근속연수'].fillna()
a['HR']['A']
a.info()
mydata3['근속연수'] = mydata3.groupby(['부서', '등급'])['근속연수'].transform(
    lambda x: x.fillna(round(x.mean()))
)

mean_tenure = (
    mydata3.groupby(["부서", "등급"])["근속연수"]
    .mean()
    .apply(np.floor)
    .reset_index()
    .rename(columns={"근속연수": "평균근속연수"})
)

mydata3 = mydata3.merge(mean_tenure, on=["부서", "등급"], how="left")
mydata3['근속연수'] = np.where(mydata3['근속연수'].isna(), mydata3['평균근속연수'], mydata3['근속연수'])


A = mydata3.loc[(mydata3['부서'] == "HR") & (mydata3['등급'] == "A"), '근속연수'].mean()
A = mydata3.loc[(mydata3["부서"] == "HR") & (mydata3["등급"] == "A"), "근속연수"].mean()
A

b = mydata3[mydata3['부서']=='Sales']
b1 = b[b['등급']=='B']
B = b1['교육참가횟수'].mean()
A + B

##############################################################

#평균만족도 칼럼은 해당 칼럼의 평균으로 대치합니다.

#idx = df['평균만족도'].isna()
#df.loc[idx, ['평균만족도']] = df['평균만족도'].mean()
mydata3["평균만족도"] = mydata3["평균만족도"].fillna(mydata3["평균만족도"].mean())

mean_tenure = (
    mydata3.groupby(["부서", "등급"])["근속연수"]
    .mean()
    .apply(np.floor)
    .reset_index()
    .rename(columns={"근속연수": "평균근속연수"})
)

#=========================================================================================================
#4
url = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_1.csv"
mydata4 = pd.read_csv(url)
mydata4

mydata4.groupby('대륙')['맥주소비량'].mean().idxmax()

a = mydata4[mydata['대륙'] == 'SA'].sort_values('맥주소비량', ascending=False)
a.groupby('국가')['맥주소비량'].sum().sort_values(ascending=False)

b= mydata4[mydata['국가'] == 'Venezuela']
b['맥주소비량'].mean()

#5
url = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_2.csv"
mydata5 = pd.read_csv(url)
mydata5

mydata5['관광']
mydata5['국가'][1]

mydata5.iloc[1].sum(axis=1)
mydata5.iloc[0, 1:].sum()

country_sum = mydata5.groupby("국가").sum()
(country_sum['관광']/country_sum.sum(axis=1)*100).sort_values(ascending=False)

country_sum.sort_values('관광', ascending=False)

mydata5[mydata5['국가'] == '이스라엘']['공무'].mean()


#=========================================================================================================
#6
import pandas as pd

# 1. 데이터 불러오기
url = 'https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_3.csv'
mydata6 = pd.read_csv(url)

# 2. 필요한 칼럼만 선택
cols = ['CO(GT)', 'NMHC(GT)']
data = mydata6[cols]

# 3. NaN 제거 (옵션: 데이터에 따라 필요)
# data = data.dropna()

# 4. Min-Max 스케일링 직접 계산
scaled_data = (data - data.min()) / (data.max() - data.min())

# 5. 표준편차 계산 + 소수점 셋째 자리 반올림
std_CO = round(scaled_data['CO(GT)'].std(), 3)
std_NMHC = round(scaled_data['NMHC(GT)'].std(), 3)

# 6. 결과 출력
print(f"CO(GT) 표준편차: {std_CO}")
print(f"NMHC(GT) 표준편차: {std_NMHC}")


#=========================================================================================================
#7

url7 = 'https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_1.csv'
mydata7 = pd.read_csv(url7)
mydata7
mydata7.info()

mydata7.loc[:, ['신고일자', '신고시각', '처리일자', '처리시각']] = mydata7.loc[:, ['신고일자', '신고시각', '처리일자', '처리시각']].astype(str)
mydata7['신고일시2'] = mydata7['신고일자'] + mydata7['신고시각']
mydata7['처리일시2'] = mydata7['처리일자'] + mydata7['처리시각']

mydata7['my신고일시'] = pd.to_datetime(mydata7['신고일시2'])
mydata7['my처리일시'] = pd.to_datetime(mydata7['처리일시2'])

mydata7['실제처리시간'] = (mydata7['my처리일시'] - mydata7['my신고일시']).dt.total_seconds()

mydata7.groupby('공장명')['실제처리시간'].mean().sort_values(ascending=False)

cols7 = ['신고일자', '신고시각', '처리일자', '처리시각']
data = mydata7[cols7]
data
datetime_data = pd.to_datetime(data['신고일자'])
pd.to_datetime(mydata7['신고일자'] )
mydata7
new = str(mydata7['신고시각'][0:])


pd.to_datetime(mydata7['신고시각'])


######################################################
mydata7['신고일시'] = pd.to_datetime(mydata7['신고일자'] + mydata7['신고시각'], format='%Y%m%d%H%M%S')
mydata7['처리일시'] = pd.to_datetime(mydata7['처리일자'] + mydata7['처리시각'], format='%Y%m%d%H%M%S')

(mydata7['처리일시'] - mydata7['신고일시']).dt.total_seconds().head(2)
mydata7['처리시간'] = (mydata7['처리일시'] - mydata7['신고일시']).dt.total_seconds()


#=========================================================================================================
#8
url = 'https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_2.csv'
mydata8 = pd.read_csv(url)
mydata8

addr = mydata8['STATION_ADDR1']
mydata8['STATION_ADDR1'] = addr.str.extract(r'([가-힣]+구)')

mydata8.groupby('STATION_ADDR1')['dist'].mean()

#=========================================================================================================
#9
url9 = 'https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_3.csv'
mydata9 = pd.read_csv(url9)
mydata9
mydata9.info()

quarter = []
for i in range(0,8):
    # print(mydata9.iloc[3*i: 3*i+3, 1:].sum(axis=1))
    quarter.append(mydata9.iloc[3*i: 3*i+3, 1:].sum(axis=1).mean())
quarter

total = []
for j in range(0,8):
    total.append(mydata9.iloc[3*j: 3*j+3, 1:].sum(axis=1).sum())
total

mydata9.iloc[6:9, 1:].sum(axis=1)


####################################################################
mydata9['총판매량'] = mydata9[['제품A', '제품B', '제품C', '제품D', '제품E']].sum(axis = 1)
mydata91 = mydata9.copy()

mydata91[['연도', '월']] = mydata91['기간'].str.split('_', expand = True)
print(mydata91[['연도', '월']].head())

mydata91['월'] = mydata91['월'].str.replace('월', '').astype(int)

mydata91['분기'] = pd.cut(mydata91['월'], bins=[0, 3, 6, 9, 12], labels=[1, 2, 3, 4], right=True)
print(mydata91.head())

# 분기별 총판매량의 월평균 계산
quarterly_avg = mydata91.groupby(['연도', '분기'])['총판매량'].mean().reset_index()