import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

lcd_df = pd.read_csv('./geodata/seoul_bike.csv')
lcd_df

lcd_df["LCD거치대수"]

fig = px.scatter_mapbox(
    lcd_df,
    lat="lat",
    lon="long",
    size="LCD거치대수",
    color="자치구",
    hover_name="대여소명", # 마우스 오버 시 표시한 텍스트
    hover_data={"lat": False, "long": False, "LCD거치대수": True, "자치구": True, 'text': False},
    text="text",
    zoom=11,
    height=650,
)
# carto-positron : 무료, 지도 배경 스타일 지정
fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

import pandas as pd
pd.set_option('display.max_columns', None)
import geopandas as gpd
gdf = gpd.read_file("./geodata/seoul/TL_SCCO_SIG_W.shp")

gdf.head()

import json
with open('./geodata/seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)
print(geojson_data.keys())

geojson_data['features'][0]['properties']
geo_list = geojson_data['features'][20]["geometry"]["coordinates"]
len(geo_list[0])

# 산점도 X, Y 좌표
geo_list[0]
np.array(geo_list[0]).shape

x=np.array(geo_list[0])[:, 0]
y=np.array(geo_list[0])[:, 1]

fig = go.Figure()
fig
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name="Map Plot"
    )
)

lcd_df = pd.read_csv('./geodata/seoul_bike.csv')
print(lcd_df.head())

agg_df = (lcd_df
    .groupby("자치구",
    as_index=False)["LCD거치대수"]
    .sum())

agg_df.columns = ["자치구", "LCD합계"]
# 컬럼 이름을 GeoJSON과 맞추기
agg_df = agg_df.rename(columns={"자치구": "SIG_KOR_NM"})
print(agg_df.head(2))

fig = px.choropleth_mapbox(
    agg_df,
    geojson=geojson_data,
    locations="SIG_KOR_NM",
    featureidkey="properties.SIG_KOR_NM",
    color="LCD합계",
    color_continuous_scale="Blues",
    mapbox_style="carto-positron",
    center={"lat": 37.5665, "lon": 126.9780},
    zoom=10,
    opacity=0.7,
    title="서울시 자치구별 LCD 거치대 수"
)
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()

import requests
from tqdm import tqdm

bank = pd.read_excel('./geodata/iM뱅크_영업지점.xlsx')
addresses = bank
bank.info()
KAKAO_API_KEY = "e1bfc7ab682fa7a1cde39d6fece2ac2a"
url = "https://dapi.kakao.com/v2/local/search/address.json"
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# 3. 주소 → 좌표 변환
coords = []
for addr in tqdm(addresses, desc="주소 변환 중"):
    print(addr)
    res = requests.get(url, headers=headers, params={"query": addr})
    if res.status_code == 200 and res.json()["documents"]:
        doc = res.json()["documents"][0]
        coords.append({
            "주소": addr,
            "위도": float(doc["y"]),
            "경도": float(doc["x"])
        })
    else:
        coords.append({"주소": addr, "위도": None, "경도": None})

df = pd.DataFrame(coords)
print(df)

import requests

KAKAO_API_KEY = "e1bfc7ab682fa7a1cde39d6fece2ac2a"
url = "https://dapi.kakao.com/v2/local/search/address.json"
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
params = {"query": "대구광역시 중구 동성로2가 88"}

res = requests.get(url, headers=headers, params=params)
print(res.status_code)
print(res.json())

fig = px.scatter_mapbox(
    df,
    lat="위도",
    lon="경도",
    hover_name="주소", # 마우스 오버 시 표시한 텍스트
    zoom=11,
    height=650,
)
# carto-positron : 무료, 지도 배경 스타일 지정
fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

import requests
import time

# 카카오 REST API 키
KAKAO_API_KEY = "e1bfc7ab682fa7a1cde39d6fece2ac2a"

def get_lat_lon(address):
    """주소를 입력받아 위도, 경도 반환 (실패 시 None, None)"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}
    
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        data = res.json()
        if data['documents']:
            y = float(data['documents'][0]['y'])  # 위도
            x = float(data['documents'][0]['x'])  # 경도
            return y, x
    return None, None

# bank DataFrame에 위도/경도 컬럼 추가
bank['위도'] = None
bank['경도'] = None

for idx, row in bank.iterrows():
    lat, lon = get_lat_lon(row['주소'])
    bank.at[idx, '위도'] = lat
    bank.at[idx, '경도'] = lon
    time.sleep(0.3)  # API 호출 제한 방지 (초당 3회 이하)

print(bank.head())

fig = px.scatter_map(
    bank,
    lat="위도",
    lon="경도",
    hover_name="지점", # 마우스 오버 시 표시한 텍스트
    zoom=11,
    height=650,
)
# carto-positron : 무료, 지도 배경 스타일 지정
fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

import pandas as pd
import requests
import time
import plotly.express as px

# ===== 1. 카카오 API 세팅 =====
KAKAO_API_KEY = "e1bfc7ab682fa7a1cde39d6fece2ac2a"

def get_lat_lon(address):
    """주소 → (위도, 경도)"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        data = res.json()
        if data['documents']:
            y = float(data['documents'][0]['y'])  # 위도
            x = float(data['documents'][0]['x'])  # 경도
            return y, x
    return None, None

# ===== 2. 주소에서 '동' 추출 =====
def extract_dong(address):
    # '동', '가' 등으로 끝나는 행정동 추출
    parts = address.split()
    for p in parts:
        if p.endswith(('구', '군')):
            return p
    return None

# ===== 3. 위도/경도 및 동 정보 추가 =====
bank['구'] = bank['주소'].apply(extract_dong)
bank['위도'] = None
bank['경도'] = None

for idx, row in bank.iterrows():
    lat, lon = get_lat_lon(row['주소'])
    bank.at[idx, '위도'] = lat
    bank.at[idx, '경도'] = lon
    time.sleep(0.3)  # API 제한 방지

# ===== 4. Plotly 시각화 =====
fig = px.scatter_map(
    bank,
    lat="위도",
    lon="경도",
    color="구",  # 동별 색상
    hover_name="지점",
    hover_data={"주소": True, "전화번호": True},
    zoom=11,
    height=700
)

fig.update_layout(
    mapbox_style="open-street-map",  # 무료 지도 스타일
    title="대구은행 지점 위치 (동별 색상)"
)

fig.show()

bank.to_csv("대구은행_지점_위경도.csv", index=False, encoding="utf-8-sig")


import pandas as pd
import requests
import time

# ==== 설정 부분 ====
INPUT_FILE = "국가철도공단_대구_지하철_주소데이터_20230901.csv"  # 원본 CSV 파일명
OUTPUT_FILE = "대구_지하철_주소_좌표추가.csv"                # 결과 CSV 파일명
API_KEY = "e1bfc7ab682fa7a1cde39d6fece2ac2a"                # 카카오 REST API 키
ADDRESS_COLUMN = "도로명주소"                               # 도로명 주소가 들어있는 컬럼명
# ===================

# CSV 읽기 (EUC-KR 인코딩)
df = pd.read_csv(INPUT_FILE, encoding="euc-kr")

# 위도, 경도 컬럼 추가
df["위도"] = None
df["경도"] = None

headers = {"Authorization": f"KakaoAK {API_KEY}"}

for idx, addr in enumerate(df[ADDRESS_COLUMN]):
    try:
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        params = {"query": addr}
        res = requests.get(url, headers=headers, params=params)
        
        if res.status_code == 200:
            result = res.json()
            if result["documents"]:
                df.at[idx, "경도"] = result["documents"][0]["x"]
                df.at[idx, "위도"] = result["documents"][0]["y"]
                print(f"[{idx+1}/{len(df)}] 변환 완료: {addr}")
            else:
                print(f"[{idx+1}/{len(df)}] 좌표 없음: {addr}")
        else:
            print(f"[{idx+1}/{len(df)}] 요청 실패: {addr} (HTTP {res.status_code})")

        time.sleep(0.1)  # API 호출 제한 방지 (초당 10회 이내)

    except Exception as e:
        print(f"[{idx+1}/{len(df)}] 오류 발생: {addr} - {e}")

# CSV 저장 (UTF-8 with BOM으로 저장하면 엑셀에서 한글 깨짐 방지)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUTPUT_FILE}")