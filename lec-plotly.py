import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 데이터 생성
df = pd.DataFrame({'x': [1, 2, 3, 4],
'y': [1, 4, 9, 16]})
# 산점도 (Scatter Plot) 생성
fig = px.scatter(
df,
x='x', # X축 데이터 설정
y='y', # Y축 데이터 설정
title='Scatter Plot Example', # 그래프 제목 설정
size_max=10 # 마커 크기 최대값 설정
)

fig.show() # 그래프 출력


df = pd.DataFrame({'x': [1, 2, 3, 4],
'y': [1, 4, 9, 16]})

fig = px.line(
df,
x='x', # X축 데이터 설정
y='y', # Y축 데이터 설정
title='Line Plot Example', # 그래프 제목 설정
markers=False # 마커(점) 추가
)

fig.show() # 그래프 출력

# 데이터 생성
df = pd.DataFrame({'x': [1, 2, 3, 4],
'y': [1, 4, 9, 16]})
# 산점도 (Scatter Plot) 생성
fig = px.scatter(
df,
x='x', # X축 데이터 설정
y='y', # Y축 데이터 설정
title='Scatter Plot Example', # 그래프 제목 설정
size_max=10 # 마커 크기 최대값 설정
)

# 레이아웃 설정
fig.update_layout(
title='업데이트된 산점도',
xaxis_title='X 값',
yaxis_title='Y 값 (제곱)',
width=600,
height=400,
template='plotly_dark',
legend_title='범례',
paper_bgcolor='darkred',
plot_bgcolor='black',
legend=dict(x=0.5, y=1)
)

fig.show() # 그래프 출력


x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])
fig = go.Figure()
fig
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            color='blue',
            symbol='square',
            size=10,
            line=dict(width=3,
                      color='red')
        ),
        name="Line Plot"
    )
)

fig.update_layout(
    title="Go로 그린 라인 플랏",
)

fig.add_annotation(
    x=2,
    y=25,
    text="Important Point",
    showarrow=True,
    arrowhead=5,
    font=dict(color="blue", size=14)
)
fig.show()

t = np.arange(0, 5, 0.5)
fig = go.Figure()
styles = ['solid', 'dash', 'dot', 'dashdot']
colors = ['blue', 'green', 'orange', 'red']
for i, dash_style in enumerate(styles):
    fig.add_trace(
        go.Scatter(
            x=t,
            y=t + i,
            mode='lines',
            name=f'dash="{dash_style}"',
            line=dict(
                dash=dash_style,
                width=3,
                color=colors[i]
            )
        )
    )
fig.show()

# 엑셀 파일 불러오기
file_path = "./data/천안 도소매 매출.csv"
df = pd.read_csv(file_path)

# 데이터 확인
df.head(20)
df

import pandas as pd

filename = "./data/천안 도소매 매출.csv"

# CP949로 읽기
dat = pd.read_csv(filename, encoding='cp949')
dat


# 산점도 (Scatter Plot) 생성
fig = px.histogram(
dat,
x='행정동', # X축 데이터 설정
y='도소매 매출', # Y축 데이터 설정
title='천안시 행정동 별 도소매 매출', # 그래프 제목 설정
# size_max=10 # 마커 크기 최대값 설정
)

fig.show() # 그래프 출력

# pip install ccxt
import ccxt
import pandas as pd 
binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", '1d')
df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.index, # X축: 날짜
        y=df['close'], # Y축: 종가
        mode='lines+markers', # 선 + 마커 같이 표시
        marker=dict(
            color='blue', # 마커 색상
            size=6, # 마커 크기
            symbol='circle' # 원형 마커
        ),
        line=dict(
            color='blue', # 선 색상
            width=2, # 선 두께
            dash='solid' # 실선 스타일
        ),
        name="BTC/USDT Closing Price"
    )
)
fig.show()